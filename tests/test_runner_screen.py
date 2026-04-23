"""Tests for RunnerScreen status bar logic and display fixes."""

import time
from unittest.mock import MagicMock, patch

import pytest

from ralph_tui.orchestrator import ActivityEvent, UsageInfo


class FakeRunnerScreen:
    """Minimal stand-in for RunnerScreen to test status bar logic without Textual."""

    def __init__(self):
        self._last_activity_time: float = 0.0
        self._last_tool_time: float = 0.0
        self._current_tool: str | None = None
        self._last_usage: UsageInfo | None = None
        self._stall_warning_shown: bool = False
        self._last_status: str = "Initializing"
        self._log_entries: list[str] = []

    def handle_text(self, text: str) -> None:
        """Mirrors RunnerScreen._on_text logic: text-write resets the timer."""
        self._log_entries.append(text)
        self._last_activity_time = time.monotonic()

    def handle_activity(self, event: ActivityEvent) -> None:
        """Mirrors RunnerScreen._on_activity logic.

        Timer reset is anchored on text-write sites, NOT on every activity
        event. Here we only reset on tool_end because tool_end writes a
        'Done:' log line.
        """
        if event.event_type in ("tool_start", "tool_end"):
            self._last_tool_time = event.timestamp
        if event.event_type == "tool_start":
            self._current_tool = event.tool_name
        elif event.event_type in ("tool_end", "message_stop"):
            self._current_tool = None

        # Clear stale error/stall status when real activity resumes
        if event.event_type in ("text_delta", "tool_start", "message_start"):
            if any(kw in self._last_status.lower() for kw in ("error", "stall", "retry", "no activity")):
                self._last_status = "Running"

        # Log tool_end with timestamp, and reset timer since the Done line is user-visible text
        if event.event_type == "tool_end":
            ts = time.strftime("%H:%M:%S")
            tool = event.tool_name or "unknown"
            self._log_entries.append(f"[{ts} Done: {tool}]")
            self._last_activity_time = time.monotonic()

    def handle_status(self, status: str) -> None:
        self._last_status = status

    def tick_activity(self) -> None:
        """Mirror of RunnerScreen._tick_activity.

        While a tool is in flight, the tool IS the activity: the user-visible
        counter is refreshed every tick so "Last activity: 0s ago" stays on
        screen throughout a long Bash/Read. When no tool is running the
        ticker does nothing; _last_activity_time climbs so STALL WARNING can
        fire on a true idle gap.
        """
        if self._current_tool is not None:
            self._last_activity_time = time.monotonic()

    def build_status_parts(self) -> list[str]:
        """Mirrors RunnerScreen._update_status_bar logic, returns parts list."""
        status = self._last_status
        parts = [f"[{time.strftime('%H:%M:%S')}]"]

        parts.append(f"Status: {status}")

        # Activity age — always render when we have a baseline, including 0s.
        if self._last_activity_time > 0:
            age = int(time.monotonic() - self._last_activity_time)
            parts.append(f"Last activity: {age}s ago")

        # Current tool
        if self._current_tool:
            parts.append(f"Tool: {self._current_tool}")

        # Stall vs writing indicator — only if status doesn't already say stalled/errored
        status_lower = status.lower()
        is_error_or_stall = any(kw in status_lower for kw in ("error", "stall", "retry", "no activity"))

        if not is_error_or_stall and self._last_activity_time > 0:
            stream_age = time.monotonic() - self._last_activity_time
            tool_age = (time.monotonic() - self._last_tool_time) if self._last_tool_time > 0 else 0
            if stream_age > 60 and self._current_tool is None:
                parts.append("[bold red]STALL WARNING[/]")
            elif tool_age > 60 and self._last_tool_time > 0 and self._current_tool is None:
                parts.append(f"[yellow]Writing... {int(tool_age)}s[/]")

        return parts


class TestTextWriteResetsActivityTimer:
    """User contract: the timer resets every time text appears on screen,
    regardless of whether the write was driven by a text_delta, by the
    plan-usage banner, by '[Resumed at ...]', or by the ToolUseBlock
    fallback. Anchoring on text-writes (not on ActivityEvent types) is what
    the production code does."""

    def test_text_write_resets_timer(self):
        screen = FakeRunnerScreen()
        old_time = time.monotonic() - 200
        screen._last_activity_time = old_time

        before = time.monotonic()
        screen.handle_text("hello world")
        after = time.monotonic()

        # Timer moved to "now-ish"; definitely advanced past the stale value.
        assert screen._last_activity_time >= before
        assert screen._last_activity_time <= after

    def test_stall_warning_does_not_reset_timer(self):
        """stall_warning is a meta-event emitted by the watchdog, not by
        Claude, and writes no text; must not reset the timer."""
        screen = FakeRunnerScreen()
        old_time = time.monotonic() - 200
        screen._last_activity_time = old_time

        event = ActivityEvent(timestamp=time.monotonic(), event_type="stall_warning")
        screen.handle_activity(event)

        assert screen._last_activity_time == old_time

    def test_silent_activity_events_do_not_reset_timer(self):
        """message_start / message_delta / message_stop don't render text,
        so they must NOT reset the timer. This is a direct consequence of
        the user's contract."""
        screen = FakeRunnerScreen()
        old_time = time.monotonic() - 100
        for kind in ("message_start", "message_delta", "message_stop"):
            screen._last_activity_time = old_time
            ev = ActivityEvent(timestamp=time.monotonic(), event_type=kind)
            screen.handle_activity(ev)
            assert screen._last_activity_time == old_time, (
                f"{kind} reset the timer; it has no on-screen text"
            )

    def test_tool_end_resets_timer_because_it_writes_done_line(self):
        """tool_end writes '[HH:MM:SS Done: X]' to the log, which is
        user-visible text, so it DOES reset the timer."""
        screen = FakeRunnerScreen()
        screen._last_activity_time = time.monotonic() - 100
        before = time.monotonic()
        screen.handle_activity(ActivityEvent(
            timestamp=time.monotonic(), event_type="tool_end", tool_name="Bash",
        ))
        assert screen._last_activity_time >= before


class TestRealActivityClearsStallStatus:
    """1A: Real activity after stall should clear error/stall status."""

    def test_real_activity_after_stall_clears_stall_status(self):
        screen = FakeRunnerScreen()
        screen._last_status = "No activity for 120s - Claude may be stalled"

        event = ActivityEvent(
            timestamp=time.monotonic(),
            event_type="text_delta",
            text_fragment="output",
        )
        screen.handle_activity(event)

        assert screen._last_status == "Running"

    def test_error_status_cleared_on_tool_start(self):
        screen = FakeRunnerScreen()
        screen._last_status = "Error (unknown) — retrying in 30s"

        event = ActivityEvent(
            timestamp=time.monotonic(),
            event_type="tool_start",
            tool_name="Read",
        )
        screen.handle_activity(event)

        assert screen._last_status == "Running"

    def test_retry_status_cleared_on_message_start(self):
        screen = FakeRunnerScreen()
        screen._last_status = "Retry in 25s — server_error: Internal failure"

        event = ActivityEvent(
            timestamp=time.monotonic(),
            event_type="message_start",
        )
        screen.handle_activity(event)

        assert screen._last_status == "Running"

    def test_normal_status_not_cleared_on_activity(self):
        screen = FakeRunnerScreen()
        screen._last_status = "Running Claude (iteration 3)"

        event = ActivityEvent(
            timestamp=time.monotonic(),
            event_type="text_delta",
            text_fragment="output",
        )
        screen.handle_activity(event)

        # Normal status should not be overwritten
        assert screen._last_status == "Running Claude (iteration 3)"


class TestStallAndWritingMutualExclusive:
    """1A: Stall/writing indicators should not appear when status already says stalled/errored."""

    def test_writing_indicator_not_shown_during_stall_status(self):
        screen = FakeRunnerScreen()
        screen._last_status = "No activity for 120s - Claude may be stalled"
        screen._last_activity_time = time.monotonic() - 30  # Recent activity (< 60s)
        screen._last_tool_time = time.monotonic() - 90      # Tool age > 60s

        parts = screen.build_status_parts()
        writing_parts = [p for p in parts if "Writing" in p]
        assert len(writing_parts) == 0, f"Writing indicator should not appear during stall status: {parts}"

    def test_stall_warning_not_shown_during_error_status(self):
        screen = FakeRunnerScreen()
        screen._last_status = "Error (unknown) — retrying in 30s"
        screen._last_activity_time = time.monotonic() - 120  # Very old

        parts = screen.build_status_parts()
        stall_parts = [p for p in parts if "STALL WARNING" in p]
        assert len(stall_parts) == 0, f"STALL WARNING should not appear during error status: {parts}"

    def test_stall_and_writing_mutually_exclusive(self):
        """When stream_age > 60 and tool_age > 60, only STALL WARNING should appear, not Writing."""
        screen = FakeRunnerScreen()
        screen._last_status = "Running"
        now = time.monotonic()
        screen._last_activity_time = now - 90   # stream_age > 60
        screen._last_tool_time = now - 120       # tool_age > 60

        parts = screen.build_status_parts()
        has_stall = any("STALL WARNING" in p for p in parts)
        has_writing = any("Writing" in p for p in parts)

        assert has_stall, "STALL WARNING should appear when stream_age > 60"
        assert not has_writing, "Writing indicator should not appear when STALL WARNING is shown"


class TestToolEndTimestampsInLog:
    """1B: tool_end events should write timestamps to the output log."""

    def test_tool_end_writes_timestamp_to_log(self):
        screen = FakeRunnerScreen()
        event = ActivityEvent(
            timestamp=time.monotonic(),
            event_type="tool_end",
            tool_name="Edit",
        )
        screen.handle_activity(event)

        assert len(screen._log_entries) == 1
        assert "Done: Edit" in screen._log_entries[0]
        # Should contain a time pattern like [HH:MM:SS]
        assert screen._log_entries[0].startswith("[")

    def test_tool_start_does_not_write_to_log(self):
        screen = FakeRunnerScreen()
        event = ActivityEvent(
            timestamp=time.monotonic(),
            event_type="tool_start",
            tool_name="Read",
        )
        screen.handle_activity(event)

        assert len(screen._log_entries) == 0


class TestStatusBarWallClock:
    """1B: Status bar should include wall-clock time."""

    def test_status_bar_includes_wall_clock(self):
        screen = FakeRunnerScreen()
        screen._last_status = "Running"
        parts = screen.build_status_parts()

        # First part should be the wall clock
        assert parts[0].startswith("[")
        assert ":" in parts[0]  # HH:MM:SS format


class TestErrorRetryCountdown:
    """1C: Error retry should show countdown with actual error message."""

    @pytest.mark.asyncio
    async def test_error_retry_shows_countdown(self, tmp_path):
        """Error retry wait should send countdown status updates."""
        import asyncio
        from ralph_tui.config import RalphConfig
        from ralph_tui.orchestrator import Orchestrator
        from unittest.mock import AsyncMock, patch

        project = tmp_path / "project"
        project.mkdir()
        (project / "main.py").write_text("code")

        cfg = RalphConfig(
            project_path=str(project),
            initial_prompt="go",
            rerun_prompt="again",
            min_iterations=1,
            max_iterations=1,
        )

        status_messages = []

        async def capture_status(s):
            status_messages.append(s)

        orch = Orchestrator(cfg, on_status=capture_status)
        query_calls = []

        async def mock_query(prompt, options):
            query_calls.append(True)
            from claude_agent_sdk import AssistantMessage, ResultMessage, TextBlock

            if len(query_calls) == 1:
                assistant = MagicMock(spec=AssistantMessage)
                assistant.error = "server_error"
                text_block = MagicMock(spec=TextBlock)
                text_block.text = "Internal server error occurred"
                assistant.content = [text_block]
                yield assistant

                result = MagicMock(spec=ResultMessage)
                result.is_error = True
                result.session_id = "sess-err"
                result.result = ""
                result.total_cost_usd = 0.01
                result.duration_ms = 100
                result.num_turns = 1
                yield result
            else:
                result = MagicMock(spec=ResultMessage)
                result.is_error = False
                result.session_id = "sess-ok"
                result.result = "Done"
                result.total_cost_usd = 0.10
                result.duration_ms = 1000
                result.num_turns = 5
                yield result

        with patch("ralph_tui.orchestrator.query", side_effect=mock_query), \
             patch("ralph_tui.orchestrator.asyncio.sleep", new_callable=AsyncMock), \
             patch("ralph_tui.orchestrator.analyze_output", new_callable=AsyncMock) as mock_analyze:
            mock_analyze.return_value = MagicMock(should_stop=True, reason="done", summary="done")
            await orch.run()

        # Should have countdown messages with actual error info
        retry_msgs = [s for s in status_messages if "Retry in" in s or "retry" in s.lower()]
        assert len(retry_msgs) > 0, f"Expected retry countdown messages, got: {status_messages}"

        # Should include actual error type/message
        error_detail_msgs = [s for s in retry_msgs if "server_error" in s]
        assert len(error_detail_msgs) > 0, f"Expected error details in retry messages, got: {retry_msgs}"

    @pytest.mark.asyncio
    async def test_stop_event_interrupts_retry_wait(self, tmp_path):
        """Stop event should interrupt the retry countdown."""
        import asyncio
        from ralph_tui.config import RalphConfig
        from ralph_tui.orchestrator import Orchestrator
        from unittest.mock import AsyncMock, patch

        project = tmp_path / "project"
        project.mkdir()
        (project / "main.py").write_text("code")

        cfg = RalphConfig(
            project_path=str(project),
            initial_prompt="go",
            rerun_prompt="again",
            min_iterations=1,
            max_iterations=1,
        )

        orch = Orchestrator(cfg)

        async def mock_query(prompt, options):
            from claude_agent_sdk import AssistantMessage, ResultMessage, TextBlock

            assistant = MagicMock(spec=AssistantMessage)
            assistant.error = "server_error"
            text_block = MagicMock(spec=TextBlock)
            text_block.text = "Server error"
            assistant.content = [text_block]
            yield assistant

            result = MagicMock(spec=ResultMessage)
            result.is_error = True
            result.session_id = "sess"
            result.result = ""
            result.total_cost_usd = 0.0
            result.duration_ms = 0
            result.num_turns = 0
            yield result

        sleep_count = 0
        original_sleep = asyncio.sleep

        async def counting_sleep(t):
            nonlocal sleep_count
            sleep_count += 1
            if sleep_count >= 3:
                orch.stop()  # Stop during countdown

        with patch("ralph_tui.orchestrator.query", side_effect=mock_query), \
             patch("ralph_tui.orchestrator.asyncio.sleep", side_effect=counting_sleep), \
             patch("ralph_tui.orchestrator.analyze_output", new_callable=AsyncMock) as mock_analyze:
            mock_analyze.return_value = MagicMock(should_stop=True, reason="done", summary="done")
            state = await orch.run()

        assert "Stopped" in state.status or sleep_count < 30


class TestRichLogHorizontalFill:
    """Task 2: streamed output must fill terminal width, not wrap at column 80."""

    @staticmethod
    def _make_screen(tmp_path):
        """Build a RunnerScreen whose orchestrator worker is a no-op."""
        from unittest.mock import patch
        from ralph_tui.config import RalphConfig
        from ralph_tui.screens.runner_screen import RunnerScreen

        proj = tmp_path / "proj"
        proj.mkdir()
        (proj / "x.txt").write_text("x")
        cfg = RalphConfig(
            project_path=str(proj),
            initial_prompt="x",
            rerun_prompt="y",
            min_iterations=1,
            max_iterations=1,
        )
        screen = RunnerScreen(cfg)
        # Replace the orchestrator worker with a no-op so the test doesn't
        # actually copy the project / call the SDK.
        screen._run_orchestrator = lambda: None
        return cfg, screen

    @pytest.mark.asyncio
    async def test_richlog_fills_terminal_width_on_mount(self, tmp_path):
        """#output-log should be ~terminal_width - margins when mounted."""
        from ralph_tui.app import RalphApp
        cfg, screen = self._make_screen(tmp_path)
        app = RalphApp()
        async with app.run_test(size=(160, 40)) as pilot:
            await app.push_screen(screen)
            await pilot.pause()
            from textual.widgets import RichLog
            log = screen.query_one("#output-log", RichLog)
            assert log.size.width >= 150, f"widget width {log.size.width} too narrow"

    @pytest.mark.asyncio
    async def test_richlog_width_updates_on_resize(self, tmp_path):
        """Resizing the terminal must widen the log area."""
        from ralph_tui.app import RalphApp
        cfg, screen = self._make_screen(tmp_path)
        app = RalphApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await app.push_screen(screen)
            await pilot.pause()
            from textual.widgets import RichLog
            log = screen.query_one("#output-log", RichLog)
            initial = log.size.width
            await pilot.resize_terminal(220, 50)
            await pilot.pause()
            assert log.size.width > initial, (
                f"widget did not widen after resize: {initial} -> {log.size.width}"
            )

    @pytest.mark.asyncio
    async def test_long_line_wraps_at_terminal_width_not_80(self, tmp_path):
        """A 300-char line written via the TextChunk handler must wrap beyond col 80."""
        from ralph_tui.app import RalphApp
        from ralph_tui.screens.runner_screen import TextChunk
        cfg, screen = self._make_screen(tmp_path)
        app = RalphApp()
        async with app.run_test(size=(160, 40)) as pilot:
            await app.push_screen(screen)
            await pilot.pause()
            screen._on_text(TextChunk("A" * 300))
            await pilot.pause()
            await pilot.pause()
            from textual.widgets import RichLog
            log = screen.query_one("#output-log", RichLog)
            assert log.lines, "no lines rendered"
            first_len = log.lines[0].cell_length
            assert first_len > 100, (
                f"first line wrapped at {first_len} cells "
                f"(expected > 100 — near terminal width)"
            )

    @pytest.mark.asyncio
    async def test_lines_written_after_resize_use_new_width(self, tmp_path):
        """A line written before a widen+resize should be narrower than a line
        written after. Guards against expand=True caching the mount-time width.
        """
        from ralph_tui.app import RalphApp
        from ralph_tui.screens.runner_screen import TextChunk
        cfg, screen = self._make_screen(tmp_path)
        app = RalphApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await app.push_screen(screen)
            await pilot.pause()

            screen._on_text(TextChunk("A" * 400))
            await pilot.pause()
            await pilot.pause()
            from textual.widgets import RichLog
            log = screen.query_one("#output-log", RichLog)
            # Snapshot a line rendered at the narrow width.
            pre_len = log.lines[0].cell_length
            pre_widget = log.size.width

            await pilot.resize_terminal(240, 40)
            await pilot.pause()
            await pilot.pause()
            screen._on_text(TextChunk("B" * 400))
            await pilot.pause()
            await pilot.pause()

            # Find a line rendered from the post-resize 'B' stream.
            b_lines = [
                s for s in log.lines
                if "BBBB" in (s.text if hasattr(s, "text") else "")
            ]
            assert b_lines, "no post-resize line rendered"
            post_len = b_lines[0].cell_length
            post_widget = log.size.width

            assert post_widget > pre_widget, (
                f"widget did not widen on resize: {pre_widget} -> {post_widget}"
            )
            assert post_len > pre_len + 50, (
                f"post-resize line did not grow: pre={pre_len}, post={post_len} "
                f"(widget {pre_widget} -> {post_widget})"
            )

    @pytest.mark.asyncio
    async def test_tool_end_timestamp_line_uses_expand_true(self, tmp_path):
        """The `[HH:MM:SS Done: X]` line is the second log.write() site in _on_activity.

        expand=True changes the *wrap width* from RichLog's default 78 to the widget
        inner width. On a typical short 'Done:' line nothing visible changes, so we
        assert directly on the call kwargs by intercepting log.write.
        """
        from ralph_tui.app import RalphApp
        from ralph_tui.screens.runner_screen import ActivityUpdate
        from ralph_tui.orchestrator import ActivityEvent
        cfg, screen = self._make_screen(tmp_path)
        app = RalphApp()
        async with app.run_test(size=(160, 40)) as pilot:
            await app.push_screen(screen)
            await pilot.pause()

            from textual.widgets import RichLog
            log = screen.query_one("#output-log", RichLog)
            captured: list[tuple[tuple, dict]] = []
            real_write = log.write
            def spy_write(*args, **kwargs):
                captured.append((args, kwargs))
                return real_write(*args, **kwargs)
            log.write = spy_write  # type: ignore[method-assign]

            screen._on_activity(ActivityUpdate(ActivityEvent(
                timestamp=0.0,
                event_type="tool_end",
                tool_name="Bash",
            )))
            await pilot.pause()

            done_calls = [
                (a, k) for (a, k) in captured
                if a and isinstance(a[0], str) and "Done: Bash" in a[0]
            ]
            assert done_calls, f"tool_end did not write; saw {captured}"
            _, kwargs = done_calls[0]
            assert kwargs.get("expand") is True, (
                f"tool_end 'Done:' line must be written with expand=True; got kwargs={kwargs}"
            )

    @pytest.mark.parametrize("term_width", [80, 120, 160, 220, 300])
    @pytest.mark.asyncio
    async def test_long_line_fills_widget_at_every_width(self, tmp_path, term_width):
        """At each terminal width, a long streamed line must fill the log widget's width.

        Parametrizes the sensible range the user might resize to. The check is
        two-step: the widget itself sits close to the full terminal width (so
        we know layout didn't gate it to 80 cols), and the rendered first line
        fills that widget (so we know expand=True is actually on).
        """
        from ralph_tui.app import RalphApp
        from ralph_tui.screens.runner_screen import TextChunk
        cfg, screen = self._make_screen(tmp_path)
        app = RalphApp()
        async with app.run_test(size=(term_width, 40)) as pilot:
            await app.push_screen(screen)
            await pilot.pause()
            # Stream a line guaranteed longer than any width in the grid.
            screen._on_text(TextChunk("A" * (term_width * 3)))
            await pilot.pause()
            await pilot.pause()
            from textual.widgets import RichLog
            log = screen.query_one("#output-log", RichLog)
            assert log.lines, "no lines rendered"
            # Widget must take most of the terminal (chrome is small).
            assert log.size.width >= term_width - 10, (
                f"widget too narrow at term_width={term_width}: "
                f"log.size.width={log.size.width}"
            )
            # Rendered line must fill the widget (border/padding may cost 1 cell).
            first_len = log.lines[0].cell_length
            assert first_len >= log.size.width - 2, (
                f"line did not fill widget at term_width={term_width}: "
                f"first_len={first_len}, widget_width={log.size.width}"
            )


class TestStickyScroll:
    """Task 3: the output log should pause auto-scroll when the user scrolls up."""

    @staticmethod
    def _make_screen(tmp_path):
        from ralph_tui.config import RalphConfig
        from ralph_tui.screens.runner_screen import RunnerScreen
        proj = tmp_path / "proj"
        proj.mkdir()
        (proj / "x.txt").write_text("x")
        cfg = RalphConfig(
            project_path=str(proj),
            initial_prompt="x",
            rerun_prompt="y",
            min_iterations=1,
            max_iterations=1,
        )
        screen = RunnerScreen(cfg)
        screen._run_orchestrator = lambda: None
        return screen

    @pytest.mark.asyncio
    async def test_sticky_log_keeps_position_when_scrolled_up(self, tmp_path):
        """Writes arriving while the user is scrolled up must not yank the view."""
        from ralph_tui.app import RalphApp
        from ralph_tui.screens.runner_screen import StickyRichLog, TextChunk
        screen = self._make_screen(tmp_path)
        app = RalphApp()
        async with app.run_test(size=(80, 24)) as pilot:
            await app.push_screen(screen)
            await pilot.pause()
            log = screen.query_one("#output-log", StickyRichLog)
            for i in range(200):
                log.write(f"line {i}", expand=True)
            await pilot.pause()
            # Scroll to the top-ish.
            log.scroll_to(y=0, animate=False)
            await pilot.pause()
            saved_y = log.scroll_y
            # More writes arriving while the user sits in history.
            for i in range(200, 250):
                log.write(f"line {i}", expand=True)
            await pilot.pause()
            assert log.scroll_y == saved_y, (
                f"view jumped: was {saved_y}, now {log.scroll_y}"
            )
            assert log.auto_scroll is False

    @pytest.mark.asyncio
    async def test_sticky_log_resumes_autoscroll_at_bottom(self, tmp_path):
        """End binding jumps to tail and re-engages auto_scroll."""
        from ralph_tui.app import RalphApp
        from ralph_tui.screens.runner_screen import StickyRichLog
        screen = self._make_screen(tmp_path)
        app = RalphApp()
        async with app.run_test(size=(80, 24)) as pilot:
            await app.push_screen(screen)
            await pilot.pause()
            log = screen.query_one("#output-log", StickyRichLog)
            for i in range(200):
                log.write(f"line {i}", expand=True)
            await pilot.pause()
            log.scroll_to(y=0, animate=False)
            await pilot.pause()
            assert log.auto_scroll is False
            screen.action_follow_tail()
            await pilot.pause()
            assert log.auto_scroll is True
            assert log.scroll_y >= log.max_scroll_y - 1

    @pytest.mark.asyncio
    async def test_end_binding_jumps_to_bottom(self, tmp_path):
        """End binding alone must be enough to snap to the tail."""
        from ralph_tui.app import RalphApp
        from ralph_tui.screens.runner_screen import StickyRichLog
        screen = self._make_screen(tmp_path)
        app = RalphApp()
        async with app.run_test(size=(80, 24)) as pilot:
            await app.push_screen(screen)
            await pilot.pause()
            log = screen.query_one("#output-log", StickyRichLog)
            for i in range(100):
                log.write(f"row {i}", expand=True)
            await pilot.pause()
            log.scroll_to(y=0, animate=False)
            await pilot.pause()
            await pilot.press("end")
            await pilot.pause()
            assert log.scroll_y >= log.max_scroll_y - 1
            assert log.auto_scroll is True

    @pytest.mark.asyncio
    async def test_round_trip_false_true_false(self, tmp_path):
        """scroll up → End → scroll up again: auto_scroll toggles False→True→False."""
        from ralph_tui.app import RalphApp
        from ralph_tui.screens.runner_screen import StickyRichLog
        screen = self._make_screen(tmp_path)
        app = RalphApp()
        async with app.run_test(size=(80, 24)) as pilot:
            await app.push_screen(screen)
            await pilot.pause()
            log = screen.query_one("#output-log", StickyRichLog)
            for i in range(200):
                log.write(f"line {i}", expand=True)
            await pilot.pause()
            assert log.max_scroll_y > 10

            # 1) scroll up → False
            log.scroll_to(y=0, animate=False)
            await pilot.pause()
            assert log.auto_scroll is False

            # 2) End → True
            screen.action_follow_tail()
            await pilot.pause()
            assert log.auto_scroll is True

            # 3) scroll up again → False
            log.scroll_to(y=0, animate=False)
            await pilot.pause()
            assert log.auto_scroll is False

    @pytest.mark.asyncio
    async def test_empty_log_keeps_auto_scroll_true(self, tmp_path):
        """When content fits in the viewport (max_scroll_y==0), auto_scroll stays True."""
        from ralph_tui.app import RalphApp
        from ralph_tui.screens.runner_screen import StickyRichLog
        screen = self._make_screen(tmp_path)
        app = RalphApp()
        async with app.run_test(size=(80, 40)) as pilot:  # tall terminal, little content
            await app.push_screen(screen)
            await pilot.pause()
            log = screen.query_one("#output-log", StickyRichLog)
            # No writes — buffer fits trivially.
            assert log.max_scroll_y == 0
            assert log.auto_scroll is True

            # A few short writes, still within viewport.
            for i in range(3):
                log.write(f"row {i}", expand=True)
            await pilot.pause()
            assert log.max_scroll_y == 0
            assert log.auto_scroll is True

    @pytest.mark.asyncio
    async def test_mid_buffer_partial_scroll_preserves_position(self, tmp_path):
        """Scrolled to halfway: auto_scroll False, and new writes don't change scroll_y."""
        from ralph_tui.app import RalphApp
        from ralph_tui.screens.runner_screen import StickyRichLog
        screen = self._make_screen(tmp_path)
        app = RalphApp()
        async with app.run_test(size=(80, 24)) as pilot:
            await app.push_screen(screen)
            await pilot.pause()
            log = screen.query_one("#output-log", StickyRichLog)
            for i in range(300):
                log.write(f"line {i}", expand=True)
            await pilot.pause()
            half = log.max_scroll_y // 2
            assert half >= 5  # sanity: enough buffer for a mid-position
            log.scroll_to(y=half, animate=False)
            await pilot.pause()
            await pilot.pause()

            assert log.auto_scroll is False, (
                f"mid-buffer must pause auto_scroll; scroll_y={log.scroll_y}, "
                f"max={log.max_scroll_y}"
            )
            saved = log.scroll_y

            for i in range(300, 360):
                log.write(f"line {i}", expand=True)
            await pilot.pause()
            await pilot.pause()
            assert log.scroll_y == saved, (
                f"scroll_y drifted while user held mid-buffer: {saved} -> {log.scroll_y}"
            )

    @pytest.mark.asyncio
    async def test_bottom_threshold_boundary_is_exactly_one_line(self, tmp_path):
        """BOTTOM_THRESHOLD=1: scroll_y == max_scroll_y-1 is still 'at tail',
        scroll_y == max_scroll_y-2 flips auto_scroll off."""
        from ralph_tui.app import RalphApp
        from ralph_tui.screens.runner_screen import StickyRichLog
        screen = self._make_screen(tmp_path)
        app = RalphApp()
        async with app.run_test(size=(80, 24)) as pilot:
            await app.push_screen(screen)
            await pilot.pause()
            log = screen.query_one("#output-log", StickyRichLog)
            assert log.BOTTOM_THRESHOLD == 1  # assumption the test rests on
            for i in range(200):
                log.write(f"line {i}", expand=True)
            await pilot.pause()
            assert log.max_scroll_y >= 3, (
                f"need >= 3 scroll rows for the test; got {log.max_scroll_y}"
            )

            # exactly one line above the tail → still 'at tail'
            log.scroll_to(y=log.max_scroll_y - 1, animate=False)
            await pilot.pause()
            await pilot.pause()
            assert log.auto_scroll is True, (
                f"scroll_y=max-1 should keep auto_scroll True "
                f"(scroll_y={log.scroll_y}, max={log.max_scroll_y})"
            )

            # two lines above the tail → scrolled up
            log.scroll_to(y=log.max_scroll_y - 2, animate=False)
            await pilot.pause()
            await pilot.pause()
            assert log.auto_scroll is False, (
                f"scroll_y=max-2 should flip auto_scroll False "
                f"(scroll_y={log.scroll_y}, max={log.max_scroll_y})"
            )


class TestActivityTimerDuringTool:
    """Task 4: in-flight tool execution must count as activity."""

    def test_activity_age_stays_zero_during_tool_in_flight(self):
        """While a tool is in flight the 'Last activity' counter must stay at
        0-1s so the user doesn't think a legitimate long Bash/Read has
        stalled. The tick treats an in-flight tool as activity and refreshes
        _last_activity_time every second. STALL WARNING is gated on
        _current_tool (see test_no_stall_warning_during_long_tool) so the
        two behaviors are consistent.
        """
        screen = FakeRunnerScreen()
        now = time.monotonic()
        screen._last_activity_time = now
        screen._current_tool = "Bash"
        with patch("time.monotonic", return_value=now + 30):
            screen.tick_activity()  # tool is in flight -> bump
            age = int(time.monotonic() - screen._last_activity_time)
        assert age == 0, (
            f"ticker must keep age at 0 while a tool is in flight; got {age}s"
        )

    def test_activity_age_grows_after_tool_end(self):
        """With no tool in flight, the age should reflect real idle time."""
        screen = FakeRunnerScreen()
        now = time.monotonic()
        screen._last_activity_time = now
        screen._current_tool = None
        with patch("time.monotonic", return_value=now + 30):
            screen.tick_activity()
            age = int(time.monotonic() - screen._last_activity_time)
        assert age == 30

    def test_no_stall_warning_during_long_tool(self):
        """STALL WARNING must not fire while a tool is running, even 120s in."""
        screen = FakeRunnerScreen()
        now = time.monotonic()
        screen._last_activity_time = now
        screen._current_tool = "Bash"
        screen._last_status = "Running"
        # Simulate 120s of elapsed time with the tool still in flight.
        fake_now = now + 120
        with patch("time.monotonic", return_value=fake_now):
            # Every tick keeps activity fresh while the tool runs.
            screen.tick_activity()
            parts = screen.build_status_parts()
        assert not any("STALL WARNING" in p for p in parts), parts

    def test_stall_warning_when_no_tool_and_idle(self):
        """STALL WARNING must still fire when truly idle (no tool + no stream)."""
        screen = FakeRunnerScreen()
        now = time.monotonic()
        screen._last_activity_time = now
        screen._current_tool = None
        screen._last_status = "Running"
        fake_now = now + 70
        with patch("time.monotonic", return_value=fake_now):
            screen.tick_activity()  # no-op since no current_tool
            parts = screen.build_status_parts()
        assert any("STALL WARNING" in p for p in parts), parts

    @pytest.mark.asyncio
    async def test_real_screen_stall_warning_on_true_idle(self, tmp_path):
        """CONTROL: on the live RunnerScreen, 70s of no-tool idle time must
        render 'STALL WARNING' in the status bar.

        The FakeRunnerScreen sibling test above exercises the mirror; this
        one exercises the real _tick_activity + _update_status_bar path so a
        drift between the two is caught.
        """
        from ralph_tui.app import RalphApp
        from ralph_tui.config import RalphConfig
        from ralph_tui.screens.runner_screen import RunnerScreen

        proj = tmp_path / "proj"
        proj.mkdir()
        (proj / "x.txt").write_text("x")
        cfg = RalphConfig(
            project_path=str(proj),
            initial_prompt="x",
            rerun_prompt="y",
            min_iterations=1,
            max_iterations=1,
        )
        screen = RunnerScreen(cfg)
        screen._run_orchestrator = lambda: None  # no real work
        app = RalphApp()
        async with app.run_test(size=(180, 40)) as pilot:
            await app.push_screen(screen)
            await pilot.pause()
            now = time.monotonic()
            screen._last_activity_time = now
            screen._current_tool = None
            screen._last_status = "Running"

            with patch("time.monotonic", return_value=now + 70):
                screen._tick_activity()
                from textual.widgets import Static
                bar = screen.query_one("#status-bar", Static)
                rendered = bar.render()
                status_text = (
                    rendered.plain if hasattr(rendered, "plain") else str(rendered)
                )

            assert "STALL WARNING" in status_text, (
                f"real RunnerScreen missed STALL WARNING at 70s idle: {status_text!r}"
            )

    def test_every_text_write_resets_timer_real_screen(self):
        """Production contract on the real widget: _on_text must reset
        _last_activity_time on every TextChunk, regardless of whether the
        write came from a text_delta, the plan-usage banner, a '[Resumed
        at ...]' line, or the AssistantMessage ToolUseBlock fallback.

        This is the root-cause fix for the user-reported bug that the
        counter kept climbing during plan-usage waits.
        """
        import asyncio

        async def _run():
            from ralph_tui.app import RalphApp
            from ralph_tui.config import RalphConfig
            from ralph_tui.screens.runner_screen import RunnerScreen, TextChunk
            import tempfile
            from pathlib import Path

            with tempfile.TemporaryDirectory() as td:
                proj = Path(td) / "proj"
                proj.mkdir()
                (proj / "x.txt").write_text("x")
                cfg = RalphConfig(
                    project_path=str(proj),
                    initial_prompt="x",
                    rerun_prompt="y",
                    min_iterations=1,
                    max_iterations=1,
                )
                screen = RunnerScreen(cfg)
                screen._run_orchestrator = lambda: None
                app = RalphApp()
                async with app.run_test(size=(160, 40)) as pilot:
                    await app.push_screen(screen)
                    await pilot.pause()
                    for i, chunk in enumerate([
                        "streamed text_delta fragment",
                        "\n[Plan usage limit hit — sleeping until 3:45]\n",
                        "\n[Resumed at 04:48 after waiting 1h 3m]\n",
                        "\n[Tool: Read]\n",
                    ]):
                        stale = time.monotonic() - 100
                        screen._last_activity_time = stale
                        screen._on_text(TextChunk(chunk))
                        await pilot.pause()
                        assert screen._last_activity_time > stale, (
                            f"TextChunk #{i} ({chunk!r}) did not reset the timer"
                        )

        asyncio.run(_run())

    def test_stall_warning_gate_uses_current_tool_not_tick_refresh(self):
        """Regression guard: prove the STALL WARNING gate is _current_tool,
        not a synthetic tick refresh.

        Force stream_age = 120s with a tool in flight, and do NOT call
        tick_activity. STALL WARNING must still be suppressed (via the gate)
        and 'Last activity: 120s ago' MUST appear (visible liveness signal).
        """
        screen = FakeRunnerScreen()
        now = time.monotonic()
        screen._last_activity_time = now
        screen._last_tool_time = now  # tool_start touched this
        screen._current_tool = "Bash"
        screen._last_status = "Running"
        with patch("time.monotonic", return_value=now + 120):
            # Intentionally NOT calling tick_activity() — the gate alone
            # must be enough to keep STALL WARNING away.
            parts = screen.build_status_parts()
        assert not any("STALL WARNING" in p for p in parts), parts
        assert any("Last activity: 120s ago" in p for p in parts), (
            f"counter hidden; user loses the liveness signal. parts={parts}"
        )

    def test_stall_warning_fires_once_tool_clears_even_with_old_tool_time(self):
        """Sanity: after tool_end, the gate opens back up and a 70s idle gap
        produces STALL WARNING. Catches an accidental permanent-suppress
        if someone gates on `_last_tool_time > 0` instead of `_current_tool`.
        """
        screen = FakeRunnerScreen()
        now = time.monotonic()
        screen._last_activity_time = now
        screen._last_tool_time = now  # a tool ran earlier
        screen._current_tool = None   # but nothing in flight now
        screen._last_status = "Running"
        with patch("time.monotonic", return_value=now + 70):
            parts = screen.build_status_parts()
        assert any("STALL WARNING" in p for p in parts), parts

    def test_back_to_back_tools_track_current_tool(self):
        """Bash finishes, Read starts: _current_tool transitions Bash → None → Read.

        The intermediate None is expected — tool_end fires before tool_start,
        so there is a one-event gap where no tool is in flight.
        """
        screen = FakeRunnerScreen()
        t0 = time.monotonic()

        screen.handle_activity(ActivityEvent(
            timestamp=t0, event_type="tool_start", tool_name="Bash",
        ))
        assert screen._current_tool == "Bash"

        screen.handle_activity(ActivityEvent(
            timestamp=t0 + 1, event_type="tool_end", tool_name="Bash",
        ))
        assert screen._current_tool is None, (
            f"tool_end must clear current_tool; got {screen._current_tool!r}"
        )

        screen.handle_activity(ActivityEvent(
            timestamp=t0 + 2, event_type="tool_start", tool_name="Read",
        ))
        assert screen._current_tool == "Read", (
            f"new tool_start must replace cleared slot; got {screen._current_tool!r}"
        )
