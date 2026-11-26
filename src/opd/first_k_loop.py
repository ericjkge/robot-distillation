"""PEAR's agent loop with first-of-k sampling (opd.first_k): configs/agent_loop/first_k.yaml registers it under PEAR's
name, so scripts/collect.sh with FIRST_K > 1 swaps it in and changes nothing else."""

from __future__ import annotations

import os

from pear.agent_loop.agent_loop import PearAgentLoop
from pear.agent_loop.tool_parser import NoToolCall

from opd.first_k import FirstOfKClient, first_k


class FirstOfKAgentLoop(PearAgentLoop):
    """Each turn keeps the first of OPD_FIRST_K samples that ends with a usable ```json action (the parse PEAR's loop
    applies to the turn); every turn is logged to $PEAR_LOG_DIR/first_k_<pid>.jsonl."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        log_dir = os.environ.get("PEAR_LOG_DIR")
        log_path = os.path.join(log_dir, f"first_k_{os.getpid()}.jsonl") if log_dir else None
        self.server_manager = FirstOfKClient(self.server_manager, first_k(), self._has_action, log_path)

    async def _has_action(self, token_ids: list[int]) -> bool:
        try:
            await self.tool_parser.parse(token_ids, [self.tool.tool_schema])
        except NoToolCall:
            return False
        return True
