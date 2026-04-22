"""Live smoke runner for Task 1 (auto-resume after plan-usage limit).

Monkey-patches the SDK to emit a plan-usage-limit message whose reset is
~90 seconds in the future, then a normal completion. Runs the real
Orchestrator headless so a tmux capture + screencapture PNG can show the
"Plan usage limit hit" countdown followed by "Resumed at" on success.

Usage:
    .venv/bin/python tests/smoke_fixtures/plan_usage_limit.py
"""

from __future__ import annotations

import asyncio
import sys
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock

# Patch BEFORE importing orchestrator
import claude_agent_sdk
from claude_agent_sdk import AssistantMessage, ResultMessage, TextBlock


_CALL_COUNT = [0]


async def _mock_query(prompt, options):
    _CALL_COUNT[0] += 1
    if _CALL_COUNT[0] == 1:
        reset_time = (datetime.now() + timedelta(seconds=45))
        reset_str = reset_time.strftime("%-I:%M%p").lower()
        a = MagicMock(spec=AssistantMessage)
        a.error = "rate_limit"
        b = MagicMock(spec=TextBlock)
        b.text = f"You've hit your session limit · resets {reset_str}"
        a.content = [b]
        yield a
        r = MagicMock(spec=ResultMessage)
        r.is_error = True
        r.session_id = "sess-smoke-001"
        r.result = ""
        r.total_cost_usd = 0.0
        r.duration_ms = 100
        r.num_turns = 1
        yield r
    else:
        r = MagicMock(spec=ResultMessage)
        r.is_error = False
        r.session_id = "sess-smoke-001"
        r.result = "Smoke run complete."
        r.total_cost_usd = 0.1
        r.duration_ms = 2000
        r.num_turns = 3
        yield r


import ralph_tui.rate_limit as _rl_mod
_rl_mod.BUFFER_MINUTES = 0  # smoke: skip the 3-min safety buffer on retry_at

import ralph_tui.orchestrator as _orch_mod
_orch_mod.query = _mock_query
_orch_mod.PLAN_USAGE_TICK_SEC = 3  # coarse-but-visible countdown cadence

from ralph_tui.config import RalphConfig
from ralph_tui.orchestrator import Orchestrator


async def _status(s: str) -> None:
    print(f"[STATUS] {s}", flush=True)


async def _text(t: str) -> None:
    sys.stdout.write(t)
    sys.stdout.flush()


async def _main() -> None:
    with tempfile.TemporaryDirectory() as td:
        proj = Path(td) / "smokeproj"
        proj.mkdir()
        (proj / "readme.md").write_text("smoke")
        cfg = RalphConfig(
            project_path=str(proj),
            initial_prompt="smoke test",
            rerun_prompt="smoke rerun",
            min_iterations=1,
            max_iterations=1,
            soft_timeout_sec=600,
            hard_timeout_sec=1800,
        )
        orch = Orchestrator(cfg, on_status=_status, on_text=_text)
        print("[SMOKE] starting headless orchestrator with mocked SDK", flush=True)
        await orch.run()
        print("\n[SMOKE] finished", flush=True)


if __name__ == "__main__":
    asyncio.run(_main())
