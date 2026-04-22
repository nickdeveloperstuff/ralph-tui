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

    def handle_activity(self, event: ActivityEvent) -> None:
        """Mirrors RunnerScreen._on_activity logic."""
        # Do NOT reset activity time on stall_warning — that's a meta-event
        if event.event_type != "stall_warning":
            self._last_activity_time = event.timestamp

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

        # Log tool_end with timestamp
        if event.event_type == "tool_end":
            ts = time.strftime("%H:%M:%S")
            tool = event.tool_name or "unknown"
            self._log_entries.append(f"[{ts} Done: {tool}]")

    def handle_status(self, status: str) -> None:
        self._last_status = status

    def build_status_parts(self) -> list[str]:
        """Mirrors RunnerScreen._update_status_bar logic, returns parts list."""
        status = self._last_status
        parts = [f"[{time.strftime('%H:%M:%S')}]"]

        parts.append(f"Status: {status}")

        # Activity age
        if self._last_activity_time > 0:
            age = int(time.monotonic() - self._last_activity_time)
            if age > 0:
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
            if stream_age > 60:
                parts.append("[bold red]STALL WARNING[/]")
            elif tool_age > 60 and self._last_tool_time > 0:
                parts.append(f"[yellow]Writing... {int(tool_age)}s[/]")

        return parts


class TestStallWarningDoesNotResetActivityTime:
    """1A: stall_warning events should NOT reset _last_activity_time."""

    def test_stall_warning_does_not_reset_activity_time(self):
        screen = FakeRunnerScreen()
        # Set activity time to a known past value
        old_time = time.monotonic() - 200
        screen._last_activity_time = old_time

        # Fire a stall_warning event
        event = ActivityEvent(
            timestamp=time.monotonic(),
            event_type="stall_warning",
        )
        screen.handle_activity(event)

        # Activity time should NOT have changed
        assert screen._last_activity_time == old_time

    def test_real_activity_resets_activity_time(self):
        screen = FakeRunnerScreen()
        old_time = time.monotonic() - 200
        screen._last_activity_time = old_time

        now = time.monotonic()
        event = ActivityEvent(
            timestamp=now,
            event_type="text_delta",
            text_fragment="hello",
        )
        screen.handle_activity(event)

        assert screen._last_activity_time == now


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
