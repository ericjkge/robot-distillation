"""First-of-k sampling for the teacher's collection (concise episodes): each turn starts k samples of the same prompt
and keeps the first to finish with a usable action; the others are aborted on their server.

The kept turn is an ordinary sample of the teacher's, so a collected episode has the same format as one collected
with k = 1, and its thinking is shorter: in closed loop (research/pear_distill_compression/DIRECTION.md, E6) first-of-2
kept the teacher's success (26/30 vs 27/30 paired episodes) at 0.59x the executed thinking tokens.

`race` is the selection rule. `FirstOfKClient` wraps verl's LLMServerClient for one agent loop: its k requests share
the turn's sticky request id, so they land on one server, which also holds their common prompt's prefix cache. The
agent loop that installs it is opd.first_k_loop.FirstOfKAgentLoop."""

from __future__ import annotations

import asyncio
import json
import os
import time
from collections.abc import Awaitable, Callable
from typing import Any
from uuid import uuid4

FIRST_K_ENV = "OPD_FIRST_K"


def first_k() -> int:
    """The k of the run: OPD_FIRST_K (set by scripts/collect.sh from FIRST_K), 1 when unset."""
    k = int(os.environ.get(FIRST_K_ENV, "1"))
    if k < 1:
        raise ValueError(f"{FIRST_K_ENV} must be >= 1, got {k}")
    return k


async def race(
    samples: list[Awaitable[Any]], is_valid: Callable[[Any], Awaitable[bool]], abort: Callable[[int], Awaitable[None]]
) -> tuple[int, Any, int]:
    """Runs the samples concurrently and returns (index, output, valid count among the finished) of the first that
    finishes with is_valid(output); abort(i) is awaited for every sample still running then. With no valid sample, the
    first to finish is returned (the caller's parse then ends the episode as it would with k = 1)."""
    tasks = [asyncio.ensure_future(sample) for sample in samples]
    index = {task: i for i, task in enumerate(tasks)}
    pending, first, valid = set(tasks), None, 0
    winner = None
    try:
        while pending and winner is None:
            done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
            # several may finish in one wakeup: the lowest index breaks the tie, deterministically
            for task in sorted(done, key=index.__getitem__):
                output = task.result()
                first = first or (index[task], output)
                if await is_valid(output):
                    valid += 1
                    if winner is None:
                        winner = (index[task], output)
    finally:
        if pending:
            await asyncio.gather(*(abort(index[task]) for task in pending), return_exceptions=True)
            # drain: an aborted request returns (an aborted output) once its server has dropped it
            await asyncio.gather(*pending, return_exceptions=True)
    i, output = winner or first
    return i, output, valid


class FirstOfKClient:
    """verl's LLMServerClient for one agent loop, whose generate keeps the first of k samples with a usable action.
    vLLM only (PEAR's rollout): it calls the server actors' generate and abort_request directly."""

    def __init__(
        self, inner, k: int, is_valid: Callable[[list[int]], Awaitable[bool]], log_path: str | None = None
    ) -> None:
        self.inner, self.k, self.is_valid, self.log_path = inner, k, is_valid, log_path

    def __getattr__(self, name: str):
        return getattr(self.inner, name)

    async def generate(
        self,
        request_id,
        *,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
        image_data=None,
        video_data=None,
        audio_data=None,
        mm_processor_kwargs=None,
        **kwargs: Any,
    ):
        if self.k == 1:
            return await self.inner.generate(
                request_id,
                prompt_ids=prompt_ids,
                sampling_params=sampling_params,
                image_data=image_data,
                video_data=video_data,
                audio_data=audio_data,
                mm_processor_kwargs=mm_processor_kwargs,
                **kwargs,
            )

        kwargs.pop("mm_processor_output", None)  # SGLang's payload; vLLM takes the raw frames
        priority = kwargs.pop("priority", 0)
        extra = {"priority": priority} if priority else {}
        if audio_data is not None:
            extra["audio_data"] = audio_data
        if mm_processor_kwargs:
            extra["mm_processor_kwargs"] = mm_processor_kwargs

        # one acquire per sample under the turn's sticky id: the load balancer counts every in-flight request and
        # routes all k to the server that holds the episode's prefix cache
        acquired = [
            await self.inner._acquire_server(
                request_id, prompt_ids=prompt_ids, sampling_params=sampling_params, image_data=image_data
            )
            for _ in range(self.k)
        ]
        vllm_ids = [uuid4().hex for _ in range(self.k)]
        start = time.monotonic()
        try:
            samples = [
                server.generate.remote(
                    request_id=vllm_id,
                    prompt_ids=prompt_ids,
                    sampling_params=sampling_params,
                    image_data=image_data,
                    video_data=video_data,
                    **extra,
                    **kwargs,
                )
                for (_, server), vllm_id in zip(acquired, vllm_ids, strict=True)
            ]

            async def abort(i: int) -> None:
                # the prefix cache is kept: the winner's continuation reuses it next turn
                await acquired[i][1].abort_request.remote(vllm_ids[i], reset_prefix_cache=False)

            async def valid(output) -> bool:
                return getattr(output, "stop_reason", None) not in ("aborted", "abort") and await self.is_valid(
                    output.token_ids
                )

            winner, output, n_valid = await race(samples, valid, abort)
        finally:
            for server_id, _ in acquired:
                self.inner._release_server(server_id, request_id=request_id)

        global_steps = output.extra_fields.get("global_steps")
        output.extra_fields.setdefault("min_global_steps", global_steps)
        output.extra_fields.setdefault("max_global_steps", global_steps)
        if self.log_path:
            line = {
                "k": self.k,
                "winner": winner,
                "tokens": len(output.token_ids),
                "valid": n_valid,
                "seconds": round(time.monotonic() - start, 2),
            }
            with open(self.log_path, "a") as f:
                f.write(json.dumps(line) + "\n")
        return output
