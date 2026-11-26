"""opd.first_k: the first-of-k selection rule, and the client that runs it on verl's servers (faked here)."""

import asyncio
import json
from types import SimpleNamespace

import pytest

from opd.first_k import FirstOfKClient, first_k, race


def run(coro):
    return asyncio.run(coro)


async def after(seconds: float, value):
    await asyncio.sleep(seconds)
    return value


async def truthy(output) -> bool:
    return bool(output)


def test_the_first_valid_sample_wins_and_the_rest_are_aborted():
    aborted = []

    async def abort(i):
        aborted.append(i)

    winner, output, valid = run(race([after(0.05, "slow"), after(0.01, "fast"), after(0.1, "slower")], truthy, abort))
    assert (winner, output, valid) == (1, "fast", 1)
    assert sorted(aborted) == [0, 2]


def test_an_invalid_first_finisher_is_skipped():
    winner, output, _ = run(race([after(0.01, ""), after(0.03, "ok")], truthy, lambda i: asyncio.sleep(0)))
    assert (winner, output) == (1, "ok")


def test_with_no_valid_sample_the_first_to_finish_is_returned():
    winner, output, valid = run(race([after(0.03, ""), after(0.01, None)], truthy, lambda i: asyncio.sleep(0)))
    assert (winner, output, valid) == (1, None, 0)


def test_simultaneous_finishers_break_ties_by_index():
    winner, _, _ = run(race([after(0, "a"), after(0, "b")], truthy, lambda i: asyncio.sleep(0)))
    assert winner == 0


def test_first_k_reads_the_environment(monkeypatch):
    monkeypatch.delenv("OPD_FIRST_K", raising=False)
    assert first_k() == 1
    monkeypatch.setenv("OPD_FIRST_K", "2")
    assert first_k() == 2
    monkeypatch.setenv("OPD_FIRST_K", "0")
    with pytest.raises(ValueError):
        first_k()


# a stand-in for one vLLM server actor: generate.remote / abort_request.remote return awaitables, as Ray's do
class FakeServer:
    def __init__(self, delays: dict[int, float], tokens: dict[int, list[int]]):
        self.delays, self.tokens, self.calls, self.aborted = delays, tokens, [], []
        self._events = {}
        self.generate = SimpleNamespace(remote=self._generate)
        self.abort_request = SimpleNamespace(remote=self._abort)

    async def _generate(self, request_id, prompt_ids, sampling_params, **kwargs):
        n = len(self.calls)
        self.calls.append({"request_id": request_id, "prompt_ids": prompt_ids, **kwargs})
        self._events[request_id] = event = asyncio.Event()
        try:
            await asyncio.wait_for(event.wait(), self.delays[n])
            return SimpleNamespace(token_ids=[], stop_reason="aborted", extra_fields={})
        except asyncio.TimeoutError:
            return SimpleNamespace(token_ids=self.tokens[n], stop_reason="completed", extra_fields={"global_steps": 3})

    async def _abort(self, request_id, reset_prefix_cache=True):
        assert reset_prefix_cache is False  # the winner's next turn reuses the prompt's prefix cache
        self.aborted.append(request_id)
        self._events[request_id].set()


class FakeInner:
    def __init__(self, server):
        self.server, self.acquired, self.released = server, [], []

    async def _acquire_server(self, request_id, **extra):
        self.acquired.append(request_id)
        return f"s{len(self.acquired)}", self.server

    def _release_server(self, server_id, request_id=None):
        self.released.append(server_id)

    async def generate(self, request_id, **kwargs):
        return "delegated"


async def has_action(token_ids) -> bool:
    return 7 in token_ids  # stand-in for the json-block parse


def test_client_keeps_the_first_sample_with_an_action_and_aborts_the_other(tmp_path):
    server = FakeServer(delays={0: 0.05, 1: 0.01}, tokens={0: [1, 7], 1: [2, 2, 7]})
    inner = FakeInner(server)
    log = tmp_path / "first_k.jsonl"
    client = FirstOfKClient(inner, 2, has_action, str(log))
    output = run(
        client.generate(
            "traj-1",
            prompt_ids=[9, 9],
            sampling_params={"max_tokens": 10},
            image_data=["img"],
            mm_processor_output={"x": 1},
        )
    )
    assert output.token_ids == [2, 2, 7]
    assert output.extra_fields == {"global_steps": 3, "min_global_steps": 3, "max_global_steps": 3}
    assert inner.acquired == ["traj-1", "traj-1"]  # both under the trajectory's sticky id: one server
    assert sorted(inner.released) == ["s1", "s2"]
    assert server.aborted == [server.calls[0]["request_id"]]
    assert server.calls[0]["request_id"] != server.calls[1]["request_id"]  # distinct vLLM requests
    assert all(call["image_data"] == ["img"] and "mm_processor_output" not in call for call in server.calls)
    line = json.loads(log.read_text())
    assert (line["k"], line["winner"], line["tokens"], line["valid"]) == (2, 1, 3, 1)


def test_client_skips_a_first_finisher_without_an_action():
    server = FakeServer(delays={0: 0.03, 1: 0.01}, tokens={0: [7], 1: [1, 1]})
    output = run(FirstOfKClient(FakeInner(server), 2, has_action).generate("t", prompt_ids=[1], sampling_params={}))
    assert output.token_ids == [7]
    assert server.aborted == []


def test_k_1_delegates_to_verls_client():
    client = FirstOfKClient(FakeInner(None), 1, has_action)
    assert run(client.generate("t", prompt_ids=[1], sampling_params={})) == "delegated"
