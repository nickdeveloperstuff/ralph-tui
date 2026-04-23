"""TUI sibling of plan_usage_limit.py — mounts the real RunnerScreen in a
Textual App so a real tmux capture can verify the plan-usage countdown and
resume flow end-to-end.

Usage:
    .venv/bin/python tests/smoke_fixtures/plan_usage_limit_tui.py

Reset clock is set to the NEXT full minute + 1 minute (so wait is 60-119s,
always at least a minute of countdown to observe).
"""

from __future__ import annotations

import sys
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock

import claude_agent_sdk
from claude_agent_sdk import AssistantMessage, ResultMessage, TextBlock


_CALL_COUNT = [0]


async def _mock_query(prompt, options):
    _CALL_COUNT[0] += 1
    if _CALL_COUNT[0] == 1:
        now = datetime.now()
        # Round UP to the next full minute, then add 1 more minute, so wait
        # is always between 60 and 119 seconds after truncation to HH:MM.
        next_min = now.replace(second=0, microsecond=0) + timedelta(minutes=2)
        reset_str = next_min.strftime("%-I:%M%p").lower()
        a = MagicMock(spec=AssistantMessage)
        a.error = "rate_limit"
        b = MagicMock(spec=TextBlock)
        b.text = f"You've hit your session limit · resets {reset_str}"
        a.content = [b]
        yield a
        r = MagicMock(spec=ResultMessage)
        r.is_error = True
        r.session_id = "sess-tui-plan-usage"
        r.result = ""
        r.total_cost_usd = 0.0
        r.duration_ms = 100
        r.num_turns = 1
        yield r
    else:
        r = MagicMock(spec=ResultMessage)
        r.is_error = False
        r.session_id = "sess-tui-plan-usage"
        r.result = "Smoke run complete."
        r.total_cost_usd = 0.1
        r.duration_ms = 2000
        r.num_turns = 3
        yield r


import ralph_tui.rate_limit as _rl_mod
_rl_mod.BUFFER_MINUTES = 0  # skip the 3-min safety buffer on retry_at

import ralph_tui.orchestrator as _orch_mod
_orch_mod.query = _mock_query
_orch_mod.PLAN_USAGE_TICK_SEC = 3  # coarse-but-visible countdown cadence


def main() -> None:
    from textual.app import App
    from ralph_tui.config import RalphConfig
    from ralph_tui.screens.runner_screen import RunnerScreen

    tmp = tempfile.mkdtemp(prefix="ralph-plan-usage-tui-")
    proj = Path(tmp) / "proj"
    proj.mkdir()
    (proj / "readme.md").write_text("smoke")
    cfg = RalphConfig(
        project_path=str(proj),
        initial_prompt="plan-usage smoke test",
        rerun_prompt="plan-usage rerun",
        min_iterations=1,
        max_iterations=1,
        soft_timeout_sec=600,
        hard_timeout_sec=1800,
    )

    class SmokeApp(App):
        TITLE = "Ralph TUI (plan-usage smoke)"
        CSS = "Screen { background: $surface; }"

        def on_mount(self) -> None:
            self.push_screen(RunnerScreen(cfg))

    SmokeApp().run()


if __name__ == "__main__":
    main()
