"""Smoke runner for Task 4: simulate a long-running Bash tool.

Emits a `tool_use` content_block_start, then stays silent for 50 seconds
before sending `content_block_stop`. The status bar should keep
"Last activity: 0s/1s ago" the whole time, and `Tool: Bash` should be
visible throughout.
"""

from __future__ import annotations

import asyncio
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import claude_agent_sdk
from claude_agent_sdk import ResultMessage
from claude_agent_sdk.types import StreamEvent


async def _mock_query(prompt, options):
    # A small prelude line so the run is visibly alive.
    yield StreamEvent(
        uuid="evt-pre",
        session_id="sess-long-tool",
        event={"type": "content_block_delta", "delta": {"type": "text_delta", "text": "Starting a long Bash...\n"}},
    )
    await asyncio.sleep(0.5)

    # Tool start
    yield StreamEvent(
        uuid="evt-ts",
        session_id="sess-long-tool",
        event={
            "type": "content_block_start",
            "content_block": {"type": "tool_use", "name": "Bash", "id": "tu-1"},
        },
    )
    # Long silent tool execution
    await asyncio.sleep(50)

    # Tool end
    yield StreamEvent(
        uuid="evt-te",
        session_id="sess-long-tool",
        event={"type": "content_block_stop"},
    )
    await asyncio.sleep(0.5)
    yield StreamEvent(
        uuid="evt-post",
        session_id="sess-long-tool",
        event={"type": "content_block_delta", "delta": {"type": "text_delta", "text": "Bash complete.\n"}},
    )
    # Deliberate idle so the stall warning can surface after ~60s of quiet.
    await asyncio.sleep(70)

    r = MagicMock(spec=ResultMessage)
    r.is_error = False
    r.session_id = "sess-long-tool"
    r.result = "done"
    r.total_cost_usd = 0.01
    r.duration_ms = 1000
    r.num_turns = 1
    yield r


import ralph_tui.orchestrator as _orch_mod
_orch_mod.query = _mock_query
# Prevent HeartbeatWatchdog from killing the stream during our fake long tool.
_orch_mod.SOFT_TIMEOUT_SEC = 10 * 60
_orch_mod.HARD_TIMEOUT_SEC = 20 * 60


def main() -> None:
    from textual.app import App
    from ralph_tui.config import RalphConfig
    from ralph_tui.screens.runner_screen import RunnerScreen

    tmp = tempfile.mkdtemp(prefix="ralph-longtool-")
    proj = Path(tmp) / "proj"
    proj.mkdir()
    (proj / "readme.md").write_text("smoke")
    cfg = RalphConfig(
        project_path=str(proj),
        initial_prompt="run a long bash",
        rerun_prompt="again",
        min_iterations=1,
        max_iterations=1,
        soft_timeout_sec=10 * 60,
        hard_timeout_sec=20 * 60,
    )

    class SmokeApp(App):
        TITLE = "Ralph TUI (long-tool smoke)"
        CSS = "Screen { background: $surface; }"

        def on_mount(self) -> None:
            self.push_screen(RunnerScreen(cfg))

    SmokeApp().run()


if __name__ == "__main__":
    main()
