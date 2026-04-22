"""Run the full ralph-tui TUI against a mocked SDK.

Used for live screen-capture verification of Task 2 (horizontal fill) and
Task 3 (sticky scroll). Streams many long, text-only messages — enough to
overflow the output window so scroll + wrap behavior is observable.
"""

from __future__ import annotations

import asyncio
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import claude_agent_sdk
from claude_agent_sdk import ResultMessage
from claude_agent_sdk.types import StreamEvent


_SAMPLE_LINES = [
    ("Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt "
     "ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation "
     "ullamco laboris nisi ut aliquip ex ea commodo consequat. " * 3),
    ("Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat "
     "nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia "
     "deserunt mollit anim id est laborum. " * 3),
] * 20


async def _mock_query(prompt, options):
    for i, text in enumerate(_SAMPLE_LINES):
        yield StreamEvent(
            uuid=f"evt-{i}",
            session_id="sess-tui-smoke",
            event={"type": "content_block_delta", "delta": {"type": "text_delta", "text": text + "\n\n"}},
        )
        await asyncio.sleep(0.5)
    r = MagicMock(spec=ResultMessage)
    r.is_error = False
    r.session_id = "sess-tui-smoke"
    r.result = "done"
    r.total_cost_usd = 0.01
    r.duration_ms = 1000
    r.num_turns = 1
    yield r


import ralph_tui.orchestrator as _orch_mod
_orch_mod.query = _mock_query


def _build_config() -> "RalphConfig":
    from ralph_tui.config import RalphConfig
    tmp = tempfile.mkdtemp(prefix="ralph-smoke-")
    proj = Path(tmp) / "proj"
    proj.mkdir()
    (proj / "readme.md").write_text("smoke")
    return RalphConfig(
        project_path=str(proj),
        initial_prompt="stream some long lines",
        rerun_prompt="stream more",
        min_iterations=1,
        max_iterations=1,
    )


def main() -> None:
    from textual.app import App
    from ralph_tui.screens.runner_screen import RunnerScreen

    cfg = _build_config()

    class SmokeApp(App):
        TITLE = "Ralph TUI (smoke)"
        CSS = "Screen { background: $surface; }"

        def on_mount(self) -> None:
            self.push_screen(RunnerScreen(cfg))

    SmokeApp().run()


if __name__ == "__main__":
    main()
