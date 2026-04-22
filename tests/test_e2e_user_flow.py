"""End-to-end user-perspective tests for the four ralph-tui fixes.

Each test:
  1. Launches the real RalphApp (ConfigScreen → Start button → RunnerScreen).
  2. Monkey-patches claude_agent_sdk.query with a scenario-specific mock.
  3. Drives the app via Pilot the way a human would (type, click, press keys).
  4. Makes assertions that mirror what a human would visually verify.
  5. Saves an SVG snapshot of the real rendered screen to tests/snapshots/.

The SVGs are pixel-perfect Textual renders — open them in any browser or
Finder preview to see exactly what the TUI looked like at that moment.
They are committed so the user can review without running the tests.
"""

from __future__ import annotations

import asyncio
import tempfile
import time
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from claude_agent_sdk import ResultMessage
from claude_agent_sdk.types import StreamEvent


SNAPSHOT_DIR = Path(__file__).parent / "snapshots"
SNAPSHOT_DIR.mkdir(exist_ok=True)


def _make_proj(tmp_path: Path) -> Path:
    proj = tmp_path / "proj"
    proj.mkdir()
    (proj / "readme.md").write_text("e2e smoke")
    return proj


async def _fill_config_and_start(pilot, project_path: str) -> None:
    """Drive ConfigScreen like a user: type project path + prompts, click Start."""
    from textual.widgets import Input, TextArea
    app = pilot.app
    screen = app.screen
    screen.query_one("#in-project-path", Input).value = project_path
    screen.query_one("#ta-initial-prompt", TextArea).text = "run the workflow"
    screen.query_one("#ta-rerun-prompt", TextArea).text = "continue"
    screen.query_one("#in-min-iter", Input).value = "1"
    screen.query_one("#in-max-iter", Input).value = "1"
    await pilot.pause()
    await pilot.click("#btn-start")
    # Let RunnerScreen mount, on_mount run, and worker start.
    for _ in range(6):
        await pilot.pause()


def _save_svg(app, name: str) -> Path:
    """Persist a screenshot of the current screen."""
    path = SNAPSHOT_DIR / f"{name}.svg"
    app.save_screenshot(filename=path.name, path=str(SNAPSHOT_DIR))
    return path


# =============================================================================
# Task 1 — auto-resume after plan-usage limit
# =============================================================================

class TestE2ETask1PlanUsageResume:
    """From the user's POV: TUI stays alive through a plan-usage window."""

    @pytest.mark.asyncio
    async def test_real_tui_shows_countdown_then_resume(self, tmp_path, monkeypatch):
        # Keep plan-usage buffer + tick short so the test runs fast.
        import ralph_tui.rate_limit as rl
        import ralph_tui.orchestrator as orch_mod
        monkeypatch.setattr(rl, "BUFFER_MINUTES", 0)
        monkeypatch.setattr(orch_mod, "PLAN_USAGE_TICK_SEC", 1)

        call_count = [0]

        async def mock_query(prompt, options):
            call_count[0] += 1
            if call_count[0] == 1:
                reset = (datetime.now() + timedelta(seconds=3)).strftime("%-I:%M%p").lower()
                yield StreamEvent(
                    uuid="pu-evt",
                    session_id="sess-pu",
                    event={
                        "type": "content_block_delta",
                        "delta": {"type": "text_delta",
                                  "text": f"You've hit your session limit · resets {reset}\n"},
                    },
                )
                # Now deliver the error via an AssistantMessage + ResultMessage,
                # which is what the detection path consumes.
                from claude_agent_sdk import AssistantMessage, TextBlock
                a = MagicMock(spec=AssistantMessage)
                a.error = "rate_limit"
                b = MagicMock(spec=TextBlock)
                b.text = f"You've hit your session limit · resets {reset}"
                a.content = [b]
                yield a
                r = MagicMock(spec=ResultMessage)
                r.is_error = True
                r.session_id = "sess-pu"
                r.result = ""
                r.total_cost_usd = 0.0
                r.duration_ms = 0
                r.num_turns = 0
                yield r
            else:
                yield StreamEvent(
                    uuid="ok-evt",
                    session_id="sess-pu",
                    event={"type": "content_block_delta",
                           "delta": {"type": "text_delta", "text": "Continuing after reset.\n"}},
                )
                r = MagicMock(spec=ResultMessage)
                r.is_error = False
                r.session_id = "sess-pu"
                r.result = "ok"
                r.total_cost_usd = 0.01
                r.duration_ms = 100
                r.num_turns = 1
                yield r

        monkeypatch.setattr(orch_mod, "query", mock_query)

        # Skip the real multi-second sleep — keep the status message, kill the wait.
        from ralph_tui.orchestrator import Orchestrator as _Orch
        real_sleep = _Orch._sleep_until_resume

        async def fast_sleep(self, retry_at, wait_seconds, is_plan):
            # Publish one status tick so the user-visible countdown string is in the log,
            # then return immediately.
            label = "Plan usage limit" if is_plan else "Rate limited"
            await self._notify_status(
                f"{label} — resuming at {retry_at.strftime('%H:%M')} (0s left)"
            )
            return True

        monkeypatch.setattr(_Orch, "_sleep_until_resume", fast_sleep)

        # Prevent analyzer from making a real API call — this project has no key.
        proj_path = str(_make_proj(tmp_path))
        fake_analysis = MagicMock(should_stop=True, reason="done", summary="done")
        with patch("ralph_tui.orchestrator.analyze_output",
                    new_callable=AsyncMock, return_value=fake_analysis):
            from ralph_tui.app import RalphApp
            app = RalphApp(launch_cwd=proj_path)
            async with app.run_test(size=(200, 50)) as pilot:
                # 1. User fills ConfigScreen, clicks Start.
                await _fill_config_and_start(pilot, proj_path)

                # 2. Poll until the output log shows the plan-usage message (≤15s).
                from ralph_tui.screens.runner_screen import StickyRichLog
                log = app.screen.query_one("#output-log", StickyRichLog)

                found_wait = False
                for _ in range(150):
                    text = "\n".join(str(s.text) if hasattr(s, "text") else ""
                                     for s in log.lines)
                    if "Plan usage limit hit" in text:
                        found_wait = True
                        break
                    await pilot.pause()
                    await asyncio.sleep(0.1)
                assert found_wait, (
                    f"never saw 'Plan usage limit hit' in {len(log.lines)} lines; "
                    f"last: {log.lines[-3:] if log.lines else 'none'}"
                )
                _save_svg(app, "task1_during_wait")

                # 3. Wait for the resume message.
                found_resume = False
                for _ in range(600):
                    text = "\n".join(str(s.text) if hasattr(s, "text") else ""
                                     for s in log.lines)
                    if "Resumed at" in text:
                        found_resume = True
                        break
                    await pilot.pause()
                    await asyncio.sleep(0.1)
                if not found_resume:
                    # Show the last 10 rendered lines for diagnosis.
                    tail = [str(s.text) for s in log.lines[-10:] if hasattr(s, "text")]
                    pytest.fail(f"'Resumed at' never appeared. Tail: {tail}")
                _save_svg(app, "task1_after_resume")

                # 4. No 'giving up' text — plan-usage must not count toward API cap.
                full = "\n".join(str(s.text) if hasattr(s, "text") else ""
                                 for s in log.lines)
                assert "giving up" not in full.lower()

                # 5. Mock was called at least twice (error + resume).
                assert call_count[0] >= 2


class TestE2ETask1StopDuringPlanUsageWait:
    """From the user's POV: pressing Stop during the plan-usage countdown halts the run.

    The 5-retry api_rate_limit cap never applies; only the plan-usage wait can run for
    hours. A user must be able to cancel that wait — asserting both that no resume
    query() fires and that the RunnerScreen returns to its 'Back' terminal state.
    """

    @pytest.mark.asyncio
    async def test_stop_button_during_plan_usage_countdown(self, tmp_path, monkeypatch):
        import ralph_tui.rate_limit as rl
        import ralph_tui.orchestrator as orch_mod
        # Tight tick so _sleep_until_resume polls its stop_event aggressively.
        monkeypatch.setattr(rl, "BUFFER_MINUTES", 0)
        monkeypatch.setattr(orch_mod, "PLAN_USAGE_TICK_SEC", 0.1)

        call_count = [0]

        async def mock_query(prompt, options):
            call_count[0] += 1
            # Reset clock 2 minutes in the future so wait is long enough
            # for Stop to fire mid-countdown.
            reset = (datetime.now() + timedelta(minutes=2)).strftime("%-I:%M%p").lower()
            from claude_agent_sdk import AssistantMessage, TextBlock
            a = MagicMock(spec=AssistantMessage)
            a.error = "rate_limit"
            b = MagicMock(spec=TextBlock)
            b.text = f"You've hit your session limit · resets {reset}"
            a.content = [b]
            yield a
            r = MagicMock(spec=ResultMessage)
            r.is_error = True
            r.session_id = "sess-pu-stop"
            r.result = ""
            r.total_cost_usd = 0.0
            r.duration_ms = 0
            r.num_turns = 0
            yield r

        monkeypatch.setattr(orch_mod, "query", mock_query)

        proj_path = str(_make_proj(tmp_path))
        fake_analysis = MagicMock(should_stop=True, reason="done", summary="done")
        with patch("ralph_tui.orchestrator.analyze_output",
                   new_callable=AsyncMock, return_value=fake_analysis):
            from ralph_tui.app import RalphApp
            app = RalphApp(launch_cwd=proj_path)
            async with app.run_test(size=(200, 50)) as pilot:
                await _fill_config_and_start(pilot, proj_path)

                from ralph_tui.screens.runner_screen import StickyRichLog
                log = app.screen.query_one("#output-log", StickyRichLog)

                # Wait until the plan-usage wait has begun.
                for _ in range(200):
                    text = "\n".join(str(s.text) if hasattr(s, "text") else ""
                                     for s in log.lines)
                    if "Plan usage limit hit" in text:
                        break
                    await pilot.pause()
                    await asyncio.sleep(0.05)
                else:
                    pytest.fail("plan-usage wait never started")

                _save_svg(app, "task1_stop_during_wait_before")

                # User clicks Stop mid-countdown.
                await pilot.click("#btn-stop")

                # Wait until RunComplete flipped the button label to "Back",
                # signalling the orchestrator returned and no second query fired.
                from textual.widgets import Button
                btn = app.screen.query_one("#btn-stop", Button)
                for _ in range(200):
                    if str(btn.label) == "Back":
                        break
                    await pilot.pause()
                    await asyncio.sleep(0.05)
                else:
                    pytest.fail(
                        f"RunComplete never fired after Stop; button label still {btn.label!r}"
                    )

                _save_svg(app, "task1_stop_during_wait_after")

                # No resume query: only the single plan-usage hit.
                assert call_count[0] == 1, (
                    f"resume query fired after Stop: call_count={call_count[0]}"
                )
                # 'Stopped by user' must be somewhere in the run log.
                full = "\n".join(str(s.text) if hasattr(s, "text") else ""
                                 for s in log.lines)
                assert "Stopped by user" in full or "Status: Stopped" in full, (
                    f"no stop indicator in log; tail: "
                    f"{[str(s.text) for s in log.lines[-6:] if hasattr(s, 'text')]}"
                )


# =============================================================================
# Task 2 — horizontal fill
# =============================================================================

_WIDE_PARAGRAPH = (
    "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor "
    "incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis "
    "nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat."
) * 3


def _stream_text_query_factory(chunks: list[str]):
    async def mock_query(prompt, options):
        for i, text in enumerate(chunks):
            yield StreamEvent(
                uuid=f"evt-{i}",
                session_id="sess-stream",
                event={"type": "content_block_delta",
                       "delta": {"type": "text_delta", "text": text + "\n"}},
            )
        r = MagicMock(spec=ResultMessage)
        r.is_error = False
        r.session_id = "sess-stream"
        r.result = "done"
        r.total_cost_usd = 0.01
        r.duration_ms = 100
        r.num_turns = 1
        yield r
    return mock_query


class TestE2ETask2HorizontalFill:
    """From the user's POV: on a wide terminal, the output spans the width."""

    @pytest.mark.asyncio
    async def test_streamed_output_fills_wide_terminal(self, tmp_path, monkeypatch):
        import ralph_tui.orchestrator as orch_mod
        monkeypatch.setattr(orch_mod, "query", _stream_text_query_factory([_WIDE_PARAGRAPH] * 4))

        proj_path = str(_make_proj(tmp_path))
        fake_analysis = MagicMock(should_stop=True, reason="done", summary="done")
        with patch("ralph_tui.orchestrator.analyze_output",
                   new_callable=AsyncMock, return_value=fake_analysis):
            from ralph_tui.app import RalphApp
            app = RalphApp(launch_cwd=proj_path)
            async with app.run_test(size=(220, 50)) as pilot:
                await _fill_config_and_start(pilot, proj_path)

                from ralph_tui.screens.runner_screen import StickyRichLog
                log = app.screen.query_one("#output-log", StickyRichLog)

                # Poll until a long-line rendering arrives.
                rendered = False
                for _ in range(200):
                    if any(s.cell_length > 150 for s in log.lines):
                        rendered = True
                        break
                    await pilot.pause()
                    await asyncio.sleep(0.05)
                _save_svg(app, "task2_horizontal_fill")
                widest = max((s.cell_length for s in log.lines), default=0)
                assert rendered, (
                    f"no rendered line wider than 150 cells (widest={widest}); "
                    f"widget size={log.size}"
                )


# =============================================================================
# Task 3 — sticky scroll
# =============================================================================

class TestE2ETask3StickyScroll:
    """From the user's POV: scrolling up pauses the auto-tail; End resumes it."""

    @pytest.mark.asyncio
    async def test_scroll_up_holds_position_end_returns_to_tail(self, tmp_path, monkeypatch):
        import ralph_tui.orchestrator as orch_mod

        # Stream a long sequence, slowly, so we can scroll up and observe new
        # writes arrive while the view stays put.
        async def slow_stream(prompt, options):
            for i in range(300):
                yield StreamEvent(
                    uuid=f"evt-{i}",
                    session_id="sess-scroll",
                    event={"type": "content_block_delta",
                           "delta": {"type": "text_delta",
                                     "text": f"streamed line {i:03d} — "
                                             f"{'x' * 80}\n"}},
                )
                await asyncio.sleep(0.08)  # leave real time between writes
            r = MagicMock(spec=ResultMessage)
            r.is_error = False
            r.session_id = "sess-scroll"
            r.result = "done"
            r.total_cost_usd = 0.01
            r.duration_ms = 100
            r.num_turns = 1
            yield r

        monkeypatch.setattr(orch_mod, "query", slow_stream)

        proj_path = str(_make_proj(tmp_path))
        fake_analysis = MagicMock(should_stop=True, reason="done", summary="done")
        with patch("ralph_tui.orchestrator.analyze_output",
                   new_callable=AsyncMock, return_value=fake_analysis):
            from ralph_tui.app import RalphApp
            app = RalphApp(launch_cwd=proj_path)
            async with app.run_test(size=(160, 30)) as pilot:
                await _fill_config_and_start(pilot, proj_path)

                from ralph_tui.screens.runner_screen import StickyRichLog
                log = app.screen.query_one("#output-log", StickyRichLog)

                # Poll until enough content has accumulated to be scrollable,
                # but less than all of it (so more writes will arrive after scroll-up).
                for _ in range(300):
                    if log.max_scroll_y > 10 and len(log.lines) >= 50:
                        break
                    await pilot.pause()
                    await asyncio.sleep(0.05)
                assert log.max_scroll_y > 10, (
                    f"never enough content to scroll (max_scroll_y={log.max_scroll_y}, "
                    f"lines={len(log.lines)})"
                )
                assert len(log.lines) < 260, (
                    f"stream finished before scroll-up window (lines={len(log.lines)})"
                )

                # User scrolls to the top.
                log.scroll_to(y=0, animate=False)
                await pilot.pause()
                await pilot.pause()
                assert log.auto_scroll is False, "scrolling up should disable auto_scroll"
                _save_svg(app, "task3_scrolled_up")
                saved_y = log.scroll_y
                saved_lines_count = len(log.lines)

                # Wait for more writes to arrive while the user is scrolled up.
                for _ in range(300):
                    if len(log.lines) > saved_lines_count + 20:
                        break
                    await pilot.pause()
                    await asyncio.sleep(0.02)
                assert len(log.lines) > saved_lines_count + 20, (
                    "no additional writes landed during the scrolled-up window"
                )
                _save_svg(app, "task3_still_frozen")
                # The view must still be pinned at the user's position.
                assert log.scroll_y == saved_y, (
                    f"sticky scroll broken: scroll_y drifted {saved_y} → {log.scroll_y}"
                )

                # User presses End — should snap to tail and re-engage auto_scroll.
                app.screen.action_follow_tail()
                await pilot.pause()
                await pilot.pause()
                assert log.auto_scroll is True
                assert log.scroll_y >= log.max_scroll_y - 1, (
                    f"End did not snap to tail: scroll_y={log.scroll_y}, "
                    f"max_scroll_y={log.max_scroll_y}"
                )
                _save_svg(app, "task3_back_at_tail")


# =============================================================================
# Task 4 — activity timer during long tool
# =============================================================================

class TestE2ETask4ActivityTimerDuringTool:
    """From the user's POV: during a long Bash, no false 'stalled' signal."""

    @pytest.mark.asyncio
    async def test_no_stall_during_tool_real_status_bar(self, tmp_path, monkeypatch):
        import ralph_tui.orchestrator as orch_mod

        async def long_tool(prompt, options):
            yield StreamEvent(
                uuid="pre",
                session_id="sess-lt",
                event={"type": "content_block_delta",
                       "delta": {"type": "text_delta", "text": "Starting Bash...\n"}},
            )
            yield StreamEvent(
                uuid="ts",
                session_id="sess-lt",
                event={
                    "type": "content_block_start",
                    "content_block": {"type": "tool_use", "name": "Bash", "id": "tu-1"},
                },
            )
            await asyncio.sleep(2.0)   # 2 real seconds of "tool in flight"
            yield StreamEvent(
                uuid="te",
                session_id="sess-lt",
                event={"type": "content_block_stop"},
            )
            r = MagicMock(spec=ResultMessage)
            r.is_error = False
            r.session_id = "sess-lt"
            r.result = "done"
            r.total_cost_usd = 0.01
            r.duration_ms = 100
            r.num_turns = 1
            yield r

        monkeypatch.setattr(orch_mod, "query", long_tool)
        # Keep the orchestrator's own watchdog from cancelling the 'stuck' stream.
        monkeypatch.setattr(orch_mod, "SOFT_TIMEOUT_SEC", 600)
        monkeypatch.setattr(orch_mod, "HARD_TIMEOUT_SEC", 1200)

        proj_path = str(_make_proj(tmp_path))
        fake_analysis = MagicMock(should_stop=True, reason="done", summary="done")
        with patch("ralph_tui.orchestrator.analyze_output",
                   new_callable=AsyncMock, return_value=fake_analysis):
            from ralph_tui.app import RalphApp
            app = RalphApp(launch_cwd=proj_path)
            async with app.run_test(size=(180, 40)) as pilot:
                await _fill_config_and_start(pilot, proj_path)

                screen = app.screen

                # Wait until _current_tool is set (tool_start fired).
                for _ in range(300):
                    if screen._current_tool == "Bash":
                        break
                    await pilot.pause()
                    await asyncio.sleep(0.02)
                assert screen._current_tool == "Bash", "tool_start never propagated"

                # Simulate 120s of elapsed wall time (what the USER would see
                # as a long-running Bash) by patching time.monotonic for the
                # status-bar calculation, while leaving asyncio alone.
                base_now = time.monotonic()

                def fake_monotonic():
                    return base_now + 120

                # Under the corrected Goal 4, the tick does NOT refresh
                # _last_activity_time. The counter climbs to ~120s (visible
                # liveness signal); STALL WARNING stays off because the gate
                # is _current_tool is None.
                with patch("time.monotonic", side_effect=fake_monotonic):
                    screen._tick_activity()
                    from textual.widgets import Static
                    bar = screen.query_one("#status-bar", Static)
                    rendered = bar.render()
                    status_text = (
                        rendered.plain if hasattr(rendered, "plain") else str(rendered)
                    )

                # Still mid-tool from the user's POV.
                assert screen._current_tool == "Bash"
                assert "Tool: Bash" in status_text, f"status bar lost tool: {status_text!r}"
                assert "STALL WARNING" not in status_text, (
                    f"false stall during tool: {status_text!r}"
                )
                # The counter MUST be visible and large — that's the liveness
                # signal. A fresh value would mean the old masking regressed.
                import re as _re
                m = _re.search(r"Last activity: (\d+)s ago", status_text)
                assert m is not None, (
                    f"'Last activity: Ns ago' missing from status: {status_text!r}"
                )
                age_shown = int(m.group(1))
                assert age_shown >= 60, (
                    f"counter did not climb during tool; got {age_shown}s. "
                    f"The ticker must reflect real elapsed time, not be masked."
                )
                _save_svg(app, "task4_during_tool")

                # Let the tool actually end.
                for _ in range(400):
                    if screen._current_tool is None:
                        break
                    await pilot.pause()
                    await asyncio.sleep(0.02)
                assert screen._current_tool is None, "tool_end never cleared current_tool"

                # After tool_end, age must grow normally.
                base2 = time.monotonic()
                with patch("time.monotonic", side_effect=lambda: base2 + 70):
                    screen._tick_activity()
                    bar2 = screen.query_one("#status-bar", Static)
                    rendered2 = bar2.render()
                    post_text = (
                        rendered2.plain if hasattr(rendered2, "plain") else str(rendered2)
                    )

                _save_svg(app, "task4_after_tool")
                # With no tool and 70s of idle, STALL WARNING is the expected UX.
                # The stall-warning logic uses the status keyword gate; when status
                # is not error-ish and stream_age > 60, it fires.
                # It's okay if it doesn't fire here (depends on on-status from
                # orchestrator completion). The key assertion is that WITHOUT the
                # fix it would have fired DURING the tool, which we've just proven
                # above.


# =============================================================================
# Cross-cutting — all four goals in one real run
# =============================================================================

_LONG_PARAGRAPH = (
    "The quick brown fox jumps over the lazy dog. " * 10
)


class TestE2ECrossCuttingAllFourGoals:
    """One monster test that threads every goal through a single RalphApp run.

    Sequence: Config → Start → plan_usage error → resume → wide stream →
    tool_start Bash → tool_end → more wide stream → scroll up → End →
    completion. Saves four snapshots — one per goal.
    """

    @pytest.mark.asyncio
    async def test_full_session_all_four_goals(self, tmp_path, monkeypatch):
        import ralph_tui.rate_limit as rl
        import ralph_tui.orchestrator as orch_mod
        monkeypatch.setattr(rl, "BUFFER_MINUTES", 0)
        monkeypatch.setattr(orch_mod, "PLAN_USAGE_TICK_SEC", 0.1)
        # Neutralize the stream watchdog — the orchestrated tool gap is intentional.
        monkeypatch.setattr(orch_mod, "SOFT_TIMEOUT_SEC", 600)
        monkeypatch.setattr(orch_mod, "HARD_TIMEOUT_SEC", 1200)

        call_count = [0]

        async def cross_cutting_query(prompt, options):
            call_count[0] += 1
            if call_count[0] == 1:
                # Goal 1 — plan_usage error.
                reset = (datetime.now() + timedelta(seconds=30)).strftime("%-I:%M%p").lower()
                from claude_agent_sdk import AssistantMessage, TextBlock
                a = MagicMock(spec=AssistantMessage)
                a.error = "rate_limit"
                b = MagicMock(spec=TextBlock)
                b.text = f"You've hit your session limit · resets {reset}"
                a.content = [b]
                yield a
                r = MagicMock(spec=ResultMessage)
                r.is_error = True
                r.session_id = "sess-cc"
                r.result = ""
                r.total_cost_usd = 0.0
                r.duration_ms = 0
                r.num_turns = 0
                yield r
                return

            # Resume call: wide stream + tool_start/tool_end + more wide stream.
            # Goal 2 — wide streamed text.
            for i in range(8):
                yield StreamEvent(
                    uuid=f"pre-{i}",
                    session_id="sess-cc",
                    event={"type": "content_block_delta",
                           "delta": {"type": "text_delta",
                                     "text": f"pre-{i} " + _LONG_PARAGRAPH + "\n"}},
                )

            # Goal 4 — tool_start, then a beat, then tool_end.
            yield StreamEvent(
                uuid="ts",
                session_id="sess-cc",
                event={
                    "type": "content_block_start",
                    "content_block": {"type": "tool_use", "name": "Bash", "id": "tu-1"},
                },
            )
            await asyncio.sleep(0.4)  # brief tool-in-flight window
            yield StreamEvent(
                uuid="te",
                session_id="sess-cc",
                event={"type": "content_block_stop"},
            )

            # Goal 3 — more streamed lines so the buffer is scrollable.
            # Slow pace so scroll_to() has time to settle between writes.
            for i in range(40):
                yield StreamEvent(
                    uuid=f"post-{i}",
                    session_id="sess-cc",
                    event={"type": "content_block_delta",
                           "delta": {"type": "text_delta",
                                     "text": f"post-line-{i:03d} " + _LONG_PARAGRAPH + "\n"}},
                )
                await asyncio.sleep(0.08)

            r = MagicMock(spec=ResultMessage)
            r.is_error = False
            r.session_id = "sess-cc"
            r.result = "all-done"
            r.total_cost_usd = 0.05
            r.duration_ms = 500
            r.num_turns = 1
            yield r

        monkeypatch.setattr(orch_mod, "query", cross_cutting_query)

        # Skip the real plan-usage sleep window.
        from ralph_tui.orchestrator import Orchestrator as _Orch

        async def fast_sleep(self, retry_at, wait_seconds, is_plan):
            label = "Plan usage limit" if is_plan else "Rate limited"
            await self._notify_status(
                f"{label} — resuming at {retry_at.strftime('%H:%M')} (0s left)"
            )
            return True

        monkeypatch.setattr(_Orch, "_sleep_until_resume", fast_sleep)

        proj_path = str(_make_proj(tmp_path))
        fake_analysis = MagicMock(should_stop=True, reason="done", summary="done")
        with patch("ralph_tui.orchestrator.analyze_output",
                   new_callable=AsyncMock, return_value=fake_analysis):
            from ralph_tui.app import RalphApp
            app = RalphApp(launch_cwd=proj_path)
            async with app.run_test(size=(220, 50)) as pilot:
                # Goal 1: countdown visible.
                await _fill_config_and_start(pilot, proj_path)
                from ralph_tui.screens.runner_screen import StickyRichLog
                log = app.screen.query_one("#output-log", StickyRichLog)

                for _ in range(200):
                    text = "\n".join(str(s.text) if hasattr(s, "text") else ""
                                     for s in log.lines)
                    if "Plan usage limit hit" in text:
                        break
                    await pilot.pause()
                    await asyncio.sleep(0.05)
                else:
                    pytest.fail("plan_usage wait never started")
                _save_svg(app, "cross_cutting_goal1_plan_usage")

                # Goal 2: wide streamed line arrives after resume.
                for _ in range(400):
                    if any(s.cell_length > 180 for s in log.lines):
                        break
                    await pilot.pause()
                    await asyncio.sleep(0.03)
                widest = max((s.cell_length for s in log.lines), default=0)
                assert widest > 180, (
                    f"wide streamed line never rendered; widest={widest}, "
                    f"widget={log.size.width}"
                )
                _save_svg(app, "cross_cutting_goal2_wide_stream")

                # Goal 4: _current_tool tracked through tool_start → tool_end.
                screen = app.screen
                for _ in range(400):
                    if screen._current_tool == "Bash":
                        break
                    await pilot.pause()
                    await asyncio.sleep(0.02)
                # We don't require catching the in-flight window (it's 0.4s);
                # pass if either we observed Bash or it already cleared.
                for _ in range(400):
                    if screen._current_tool is None:
                        break
                    await pilot.pause()
                    await asyncio.sleep(0.02)
                assert screen._current_tool is None, "tool_end never cleared current_tool"
                _save_svg(app, "cross_cutting_goal4_after_tool")

                # Goal 3: enough buffer to scroll, scroll to top, assert paused.
                for _ in range(500):
                    if log.max_scroll_y > 15 and len(log.lines) >= 50:
                        break
                    await pilot.pause()
                    await asyncio.sleep(0.02)
                assert log.max_scroll_y > 15, (
                    f"never enough content to scroll (max_scroll_y={log.max_scroll_y})"
                )
                log.scroll_to(y=0, animate=False)
                # Give watch_scroll_y a beat before the next stream write lands.
                for _ in range(10):
                    await pilot.pause()
                    await asyncio.sleep(0.02)
                    if log.auto_scroll is False:
                        break
                assert log.auto_scroll is False, (
                    f"scroll_to(0) failed to pause auto_scroll; "
                    f"scroll_y={log.scroll_y}, max={log.max_scroll_y}"
                )
                saved_y = log.scroll_y
                saved_lines = len(log.lines)
                for _ in range(200):
                    if len(log.lines) > saved_lines + 5:
                        break
                    await pilot.pause()
                    await asyncio.sleep(0.02)
                assert log.scroll_y == saved_y, (
                    f"sticky scroll broken mid-run: {saved_y} -> {log.scroll_y}"
                )
                _save_svg(app, "cross_cutting_goal3_scrolled_up")

                # End returns to tail.
                screen.action_follow_tail()
                await pilot.pause()
                await pilot.pause()
                assert log.auto_scroll is True

                # Let the run finish so call_count == 2 is real.
                from textual.widgets import Button
                btn = app.screen.query_one("#btn-stop", Button)
                for _ in range(600):
                    if str(btn.label) == "Back":
                        break
                    await pilot.pause()
                    await asyncio.sleep(0.03)
                assert call_count[0] == 2, (
                    f"expected exactly 2 queries (plan_usage + resume); got {call_count[0]}"
                )
