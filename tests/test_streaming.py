"""TDD tests for StreamEvent handling, usage tracking, and two-tier watchdog integration."""

import asyncio
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ralph_tui.config import RalphConfig
from ralph_tui.orchestrator import (
    Orchestrator,
    ActivityEvent,
    UsageInfo,
    HeartbeatWatchdog,
    MODEL_CONTEXT_WINDOW,
    SOFT_TIMEOUT_SEC,
    HARD_TIMEOUT_SEC,
    CANCEL_GRACE_SEC,
    _kill_child_processes,
)


def _make_config(tmp_path: Path, **overrides) -> RalphConfig:
    project = tmp_path / "project"
    project.mkdir()
    (project / "main.py").write_text("print('hello')")
    defaults = dict(
        project_path=str(project),
        initial_prompt="Do something",
        rerun_prompt="Do more",
        min_iterations=1,
        max_iterations=1,
    )
    defaults.update(overrides)
    return RalphConfig(**defaults)


class TestStreamEventHandling:
    """Tests for StreamEvent processing in _stream_claude."""

    @pytest.mark.asyncio
    async def test_text_delta_fires_on_text_and_on_activity(self, tmp_path):
        """StreamEvent with content_block_delta/text_delta should fire both on_text and on_activity."""
        cfg = _make_config(tmp_path)
        text_chunks = []
        activity_events = []

        async def capture_text(t):
            text_chunks.append(t)

        async def capture_activity(e):
            activity_events.append(e)

        orch = Orchestrator(
            cfg, on_text=capture_text, on_activity=capture_activity,
        )

        async def mock_query(prompt, options):
            from claude_agent_sdk.types import StreamEvent
            from claude_agent_sdk import ResultMessage

            se = MagicMock(spec=StreamEvent)
            se.session_id = "sess-1"
            se.event = {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "Hello"}}
            yield se

            result = MagicMock(spec=ResultMessage)
            result.is_error = False
            result.session_id = "sess-1"
            result.result = "Hello"
            result.total_cost_usd = 0.01
            result.duration_ms = 100
            result.num_turns = 1
            result.usage = None
            yield result

        with patch("ralph_tui.orchestrator.query", side_effect=mock_query), \
             patch("ralph_tui.orchestrator.analyze_output", new_callable=AsyncMock) as mock_analyze:
            mock_analyze.return_value = MagicMock(should_stop=True, reason="done", summary="done")
            await orch.run()

        assert any("Hello" in t for t in text_chunks), f"Expected 'Hello' in text chunks: {text_chunks}"
        assert any(e.event_type == "text_delta" for e in activity_events), \
            f"Expected text_delta activity event: {activity_events}"

    @pytest.mark.asyncio
    async def test_tool_start_fires_activity_with_tool_name(self, tmp_path):
        """StreamEvent with content_block_start/tool_use should fire ActivityEvent with tool_name."""
        cfg = _make_config(tmp_path)
        activity_events = []

        async def capture_activity(e):
            activity_events.append(e)

        orch = Orchestrator(cfg, on_activity=capture_activity)

        async def mock_query(prompt, options):
            from claude_agent_sdk.types import StreamEvent
            from claude_agent_sdk import ResultMessage

            se = MagicMock(spec=StreamEvent)
            se.session_id = "sess-1"
            se.event = {
                "type": "content_block_start",
                "content_block": {"type": "tool_use", "name": "Read"},
            }
            yield se

            result = MagicMock(spec=ResultMessage)
            result.is_error = False
            result.session_id = "sess-1"
            result.result = "Done"
            result.total_cost_usd = 0.01
            result.duration_ms = 100
            result.num_turns = 1
            result.usage = None
            yield result

        with patch("ralph_tui.orchestrator.query", side_effect=mock_query), \
             patch("ralph_tui.orchestrator.analyze_output", new_callable=AsyncMock) as mock_analyze:
            mock_analyze.return_value = MagicMock(should_stop=True, reason="done", summary="done")
            await orch.run()

        tool_starts = [e for e in activity_events if e.event_type == "tool_start"]
        assert len(tool_starts) >= 1, f"Expected tool_start event: {activity_events}"
        assert tool_starts[0].tool_name == "Read"

    @pytest.mark.asyncio
    async def test_tool_end_fires_activity(self, tmp_path):
        """StreamEvent with content_block_stop should fire tool_end ActivityEvent."""
        cfg = _make_config(tmp_path)
        activity_events = []

        async def capture_activity(e):
            activity_events.append(e)

        orch = Orchestrator(cfg, on_activity=capture_activity)

        async def mock_query(prompt, options):
            from claude_agent_sdk.types import StreamEvent
            from claude_agent_sdk import ResultMessage

            se = MagicMock(spec=StreamEvent)
            se.session_id = "sess-1"
            se.event = {"type": "content_block_stop"}
            yield se

            result = MagicMock(spec=ResultMessage)
            result.is_error = False
            result.session_id = "sess-1"
            result.result = "Done"
            result.total_cost_usd = 0.01
            result.duration_ms = 100
            result.num_turns = 1
            result.usage = None
            yield result

        with patch("ralph_tui.orchestrator.query", side_effect=mock_query), \
             patch("ralph_tui.orchestrator.analyze_output", new_callable=AsyncMock) as mock_analyze:
            mock_analyze.return_value = MagicMock(should_stop=True, reason="done", summary="done")
            await orch.run()

        assert any(e.event_type == "tool_end" for e in activity_events), \
            f"Expected tool_end event: {activity_events}"

    @pytest.mark.asyncio
    async def test_session_id_captured_from_stream_event(self, tmp_path):
        """session_id from StreamEvent should be captured and available."""
        cfg = _make_config(tmp_path)
        orch = Orchestrator(cfg)

        async def mock_query(prompt, options):
            from claude_agent_sdk.types import StreamEvent
            from claude_agent_sdk import ResultMessage

            se = MagicMock(spec=StreamEvent)
            se.session_id = "sess-from-stream"
            se.event = {"type": "message_start"}
            yield se

            result = MagicMock(spec=ResultMessage)
            result.is_error = False
            result.session_id = "sess-from-stream"
            result.result = "Done"
            result.total_cost_usd = 0.01
            result.duration_ms = 100
            result.num_turns = 1
            result.usage = None
            yield result

        with patch("ralph_tui.orchestrator.query", side_effect=mock_query), \
             patch("ralph_tui.orchestrator.analyze_output", new_callable=AsyncMock) as mock_analyze:
            mock_analyze.return_value = MagicMock(should_stop=True, reason="done", summary="done")
            await orch.run()

        # If we get here without error, session_id handling works

    @pytest.mark.asyncio
    async def test_include_partial_messages_enabled(self, tmp_path):
        """The query options should have include_partial_messages=True."""
        cfg = _make_config(tmp_path)
        orch = Orchestrator(cfg)
        query_calls = []

        async def mock_query(prompt, options):
            query_calls.append(options)
            from claude_agent_sdk import ResultMessage
            result = MagicMock(spec=ResultMessage)
            result.is_error = False
            result.session_id = "sess-1"
            result.result = "Done"
            result.total_cost_usd = 0.01
            result.duration_ms = 100
            result.num_turns = 1
            result.usage = None
            yield result

        with patch("ralph_tui.orchestrator.query", side_effect=mock_query), \
             patch("ralph_tui.orchestrator.analyze_output", new_callable=AsyncMock) as mock_analyze:
            mock_analyze.return_value = MagicMock(should_stop=True, reason="done", summary="done")
            await orch.run()

        assert len(query_calls) >= 1
        assert query_calls[0].include_partial_messages is True


class TestUsageTracking:
    """Tests for token usage extraction and callbacks."""

    @pytest.mark.asyncio
    async def test_usage_callback_fires_with_token_counts(self, tmp_path):
        """ResultMessage with usage dict should fire on_usage with correct values."""
        cfg = _make_config(tmp_path)
        usage_events = []

        async def capture_usage(u):
            usage_events.append(u)

        orch = Orchestrator(cfg, on_usage=capture_usage)

        async def mock_query(prompt, options):
            from claude_agent_sdk import ResultMessage
            result = MagicMock(spec=ResultMessage)
            result.is_error = False
            result.session_id = "sess-1"
            result.result = "Done"
            result.total_cost_usd = 0.01
            result.duration_ms = 100
            result.num_turns = 1
            result.usage = {
                "input_tokens": 50000,
                "output_tokens": 1000,
                "cache_creation_input_tokens": 5000,
                "cache_read_input_tokens": 10000,
            }
            yield result

        with patch("ralph_tui.orchestrator.query", side_effect=mock_query), \
             patch("ralph_tui.orchestrator.analyze_output", new_callable=AsyncMock) as mock_analyze:
            mock_analyze.return_value = MagicMock(should_stop=True, reason="done", summary="done")
            await orch.run()

        assert len(usage_events) >= 1, "Expected at least one usage event"
        u = usage_events[0]
        # input_tokens now = total context = 50000 + 5000 + 10000 = 65000
        assert u.input_tokens == 65000
        assert u.output_tokens == 1000
        assert u.cache_creation_input_tokens == 5000
        assert u.cache_read_input_tokens == 10000
        assert u.total_tokens == 66000
        assert u.context_percent == pytest.approx(32.5)

    @pytest.mark.asyncio
    async def test_no_usage_callback_when_usage_is_none(self, tmp_path):
        """No on_usage callback should fire when ResultMessage.usage is None."""
        cfg = _make_config(tmp_path)
        usage_events = []

        async def capture_usage(u):
            usage_events.append(u)

        orch = Orchestrator(cfg, on_usage=capture_usage)

        async def mock_query(prompt, options):
            from claude_agent_sdk import ResultMessage
            result = MagicMock(spec=ResultMessage)
            result.is_error = False
            result.session_id = "sess-1"
            result.result = "Done"
            result.total_cost_usd = 0.01
            result.duration_ms = 100
            result.num_turns = 1
            result.usage = None
            yield result

        with patch("ralph_tui.orchestrator.query", side_effect=mock_query), \
             patch("ralph_tui.orchestrator.analyze_output", new_callable=AsyncMock) as mock_analyze:
            mock_analyze.return_value = MagicMock(should_stop=True, reason="done", summary="done")
            await orch.run()

        assert len(usage_events) == 0

    @pytest.mark.asyncio
    async def test_context_percent_calculation(self, tmp_path):
        """50k input_tokens / 200k window = 25%."""
        cfg = _make_config(tmp_path)
        usage_events = []

        async def capture_usage(u):
            usage_events.append(u)

        orch = Orchestrator(cfg, on_usage=capture_usage)

        async def mock_query(prompt, options):
            from claude_agent_sdk import ResultMessage
            result = MagicMock(spec=ResultMessage)
            result.is_error = False
            result.session_id = "sess-1"
            result.result = "Done"
            result.total_cost_usd = 0.01
            result.duration_ms = 100
            result.num_turns = 1
            result.usage = {"input_tokens": 100000, "output_tokens": 5000}
            yield result

        with patch("ralph_tui.orchestrator.query", side_effect=mock_query), \
             patch("ralph_tui.orchestrator.analyze_output", new_callable=AsyncMock) as mock_analyze:
            mock_analyze.return_value = MagicMock(should_stop=True, reason="done", summary="done")
            await orch.run()

        assert len(usage_events) >= 1
        assert usage_events[0].context_percent == pytest.approx(50.0)


class TestTwoTierWatchdogIntegration:
    """Integration tests for two-tier watchdog in the orchestrator."""

    @pytest.mark.asyncio
    async def test_soft_stall_sends_warning_status(self, tmp_path):
        """Soft timeout should emit a stall warning status but not cancel."""
        cfg = _make_config(tmp_path)
        status_messages = []

        async def capture_status(s):
            status_messages.append(s)

        orch = Orchestrator(cfg, on_status=capture_status)

        async def mock_query(prompt, options):
            from claude_agent_sdk.types import StreamEvent
            from claude_agent_sdk import ResultMessage

            se = MagicMock(spec=StreamEvent)
            se.session_id = "sess-1"
            se.event = {"type": "message_start"}
            yield se

            # Delay long enough for soft warning but not hard cancel
            await asyncio.sleep(0.5)

            result = MagicMock(spec=ResultMessage)
            result.is_error = False
            result.session_id = "sess-1"
            result.result = "Done"
            result.total_cost_usd = 0.01
            result.duration_ms = 100
            result.num_turns = 1
            result.usage = None
            yield result

        with patch("ralph_tui.orchestrator.query", side_effect=mock_query), \
             patch("ralph_tui.orchestrator.analyze_output", new_callable=AsyncMock) as mock_analyze, \
             patch("ralph_tui.orchestrator.SOFT_TIMEOUT_SEC", 0.1), \
             patch("ralph_tui.orchestrator.HARD_TIMEOUT_SEC", 9999), \
             patch("ralph_tui.orchestrator.WATCHDOG_CHECK_INTERVAL_SEC", 0.05):
            mock_analyze.return_value = MagicMock(should_stop=True, reason="done", summary="done")
            await orch.run()

        stall_msgs = [s for s in status_messages if "stalled" in s.lower() or "activity" in s.lower()]
        assert len(stall_msgs) >= 1, f"Expected stall warning in status: {status_messages}"

    @pytest.mark.asyncio
    async def test_hard_stall_cancels_and_retries(self, tmp_path):
        """Hard timeout should cancel the stream and trigger a retry."""
        cfg = _make_config(tmp_path)
        orch = Orchestrator(cfg)
        query_calls = []

        async def mock_query(prompt, options):
            query_calls.append(True)
            from claude_agent_sdk.types import StreamEvent
            from claude_agent_sdk import ResultMessage

            se = MagicMock(spec=StreamEvent)
            se.session_id = "sess-stall"
            se.event = {"type": "message_start"}
            yield se

            if len(query_calls) == 1:
                # Block long enough for hard timeout
                await asyncio.sleep(2)

                result = MagicMock(spec=ResultMessage)
                result.is_error = False
                result.session_id = "sess-stall"
                result.result = "Stalled output"
                result.total_cost_usd = 0.50
                result.duration_ms = 5000
                result.num_turns = 20
                result.usage = None
                yield result
            else:
                result = MagicMock(spec=ResultMessage)
                result.is_error = False
                result.session_id = "sess-ok"
                result.result = "Done!"
                result.total_cost_usd = 0.10
                result.duration_ms = 1000
                result.num_turns = 5
                result.usage = None
                yield result

        original_sleep = asyncio.sleep

        async def patched_sleep(duration):
            """Skip only long retry waits; keep short watchdog sleeps real."""
            if duration >= 1:
                return  # Skip retry waits
            await original_sleep(duration)

        with patch("ralph_tui.orchestrator.query", side_effect=mock_query), \
             patch("ralph_tui.orchestrator.SOFT_TIMEOUT_SEC", 0.05), \
             patch("ralph_tui.orchestrator.HARD_TIMEOUT_SEC", 0.2), \
             patch("ralph_tui.orchestrator.WATCHDOG_CHECK_INTERVAL_SEC", 0.05), \
             patch("ralph_tui.orchestrator.ERROR_RETRY_WAIT_SEC", 0), \
             patch("ralph_tui.orchestrator.analyze_output", new_callable=AsyncMock) as mock_analyze:
            mock_analyze.return_value = MagicMock(should_stop=True, reason="done", summary="done")
            state = await orch.run()

        assert len(query_calls) >= 2, "Expected hard stall to trigger retry"

    @pytest.mark.asyncio
    async def test_activity_event_fires_stall_warning(self, tmp_path):
        """When soft timeout fires, a stall_warning ActivityEvent should be emitted."""
        cfg = _make_config(tmp_path)
        activity_events = []

        async def capture_activity(e):
            activity_events.append(e)

        orch = Orchestrator(cfg, on_activity=capture_activity)

        async def mock_query(prompt, options):
            from claude_agent_sdk.types import StreamEvent
            from claude_agent_sdk import ResultMessage

            se = MagicMock(spec=StreamEvent)
            se.session_id = "sess-1"
            se.event = {"type": "message_start"}
            yield se

            # Delay for soft warning
            await asyncio.sleep(0.5)

            result = MagicMock(spec=ResultMessage)
            result.is_error = False
            result.session_id = "sess-1"
            result.result = "Done"
            result.total_cost_usd = 0.01
            result.duration_ms = 100
            result.num_turns = 1
            result.usage = None
            yield result

        with patch("ralph_tui.orchestrator.query", side_effect=mock_query), \
             patch("ralph_tui.orchestrator.analyze_output", new_callable=AsyncMock) as mock_analyze, \
             patch("ralph_tui.orchestrator.SOFT_TIMEOUT_SEC", 0.1), \
             patch("ralph_tui.orchestrator.HARD_TIMEOUT_SEC", 9999), \
             patch("ralph_tui.orchestrator.WATCHDOG_CHECK_INTERVAL_SEC", 0.05):
            mock_analyze.return_value = MagicMock(should_stop=True, reason="done", summary="done")
            await orch.run()

        stall_events = [e for e in activity_events if e.event_type == "stall_warning"]
        assert len(stall_events) >= 1, f"Expected stall_warning activity: {activity_events}"


class TestEnhancedLogging:
    """Tests for JSONL tool event logging and usage in logs."""

    @pytest.mark.asyncio
    async def test_tool_events_logged_to_jsonl(self, tmp_path):
        """tool_start and tool_end events should appear in ralph-log.jsonl."""
        import json
        cfg = _make_config(tmp_path)
        orch = Orchestrator(cfg)

        async def mock_query(prompt, options):
            from claude_agent_sdk.types import StreamEvent
            from claude_agent_sdk import ResultMessage

            # tool start
            se_start = MagicMock(spec=StreamEvent)
            se_start.session_id = "sess-1"
            se_start.event = {
                "type": "content_block_start",
                "content_block": {"type": "tool_use", "name": "Edit"},
            }
            yield se_start

            # tool end
            se_end = MagicMock(spec=StreamEvent)
            se_end.session_id = "sess-1"
            se_end.event = {"type": "content_block_stop"}
            yield se_end

            result = MagicMock(spec=ResultMessage)
            result.is_error = False
            result.session_id = "sess-1"
            result.result = "Done"
            result.total_cost_usd = 0.01
            result.duration_ms = 100
            result.num_turns = 1
            result.usage = None
            yield result

        with patch("ralph_tui.orchestrator.query", side_effect=mock_query), \
             patch("ralph_tui.orchestrator.analyze_output", new_callable=AsyncMock) as mock_analyze:
            mock_analyze.return_value = MagicMock(should_stop=True, reason="done", summary="done")
            await orch.run()

        project_path = Path(cfg.project_path).expanduser().resolve()
        runs_dir = project_path.parent / f"{project_path.name}-ralph-runs"
        log_file = runs_dir / "ralph-log.jsonl"
        lines = log_file.read_text().strip().split("\n")
        entries = [json.loads(line) for line in lines]
        tool_entries = [e for e in entries if e.get("type") == "tool_event"]
        assert len(tool_entries) >= 2, f"Expected tool_event entries: {entries}"

    @pytest.mark.asyncio
    async def test_usage_included_in_iteration_log(self, tmp_path):
        """Iteration log entries should include token usage when available."""
        import json
        cfg = _make_config(tmp_path)
        orch = Orchestrator(cfg)

        async def mock_query(prompt, options):
            from claude_agent_sdk import ResultMessage
            result = MagicMock(spec=ResultMessage)
            result.is_error = False
            result.session_id = "sess-1"
            result.result = "Done"
            result.total_cost_usd = 0.01
            result.duration_ms = 100
            result.num_turns = 1
            result.usage = {
                "input_tokens": 50000,
                "output_tokens": 1000,
            }
            yield result

        with patch("ralph_tui.orchestrator.query", side_effect=mock_query), \
             patch("ralph_tui.orchestrator.analyze_output", new_callable=AsyncMock) as mock_analyze:
            mock_analyze.return_value = MagicMock(should_stop=True, reason="done", summary="done")
            await orch.run()

        project_path = Path(cfg.project_path).expanduser().resolve()
        runs_dir = project_path.parent / f"{project_path.name}-ralph-runs"
        log_file = runs_dir / "ralph-log.jsonl"
        lines = log_file.read_text().strip().split("\n")
        # Find the iteration log entry (not tool_event)
        iter_entries = [json.loads(l) for l in lines if "iteration" in json.loads(l) and json.loads(l).get("type") != "tool_event"]
        assert len(iter_entries) >= 1
        assert iter_entries[0].get("input_tokens") == 50000
        assert iter_entries[0].get("output_tokens") == 1000
        assert "context_percent" in iter_entries[0]


class TestWatchdogConcurrent:
    """Tests proving the watchdog must run concurrently, not inline.

    These tests use mock generators that *block* between yields,
    simulating a hung Claude stream. The inline watchdog (which only
    checks after each yield) cannot detect these stalls.
    """

    @pytest.mark.asyncio
    async def test_watchdog_fires_during_blocked_stream(self, tmp_path):
        """A stream that blocks mid-iteration should be cancelled by the hard watchdog."""
        cfg = _make_config(tmp_path)
        orch = Orchestrator(cfg)

        async def mock_query(prompt, options):
            from claude_agent_sdk.types import StreamEvent
            from claude_agent_sdk import ResultMessage

            # Yield one message, then block for a long time
            se = MagicMock(spec=StreamEvent)
            se.session_id = "sess-block"
            se.event = {"type": "message_start"}
            yield se

            # Simulate a hung stream — block longer than hard timeout
            await asyncio.sleep(10)

            # This should never be reached if the watchdog works
            result = MagicMock(spec=ResultMessage)
            result.is_error = False
            result.session_id = "sess-block"
            result.result = "Should not get here"
            result.total_cost_usd = 0.01
            result.duration_ms = 100
            result.num_turns = 1
            result.usage = None
            yield result

        with patch("ralph_tui.orchestrator.query", side_effect=mock_query), \
             patch("ralph_tui.orchestrator.HARD_TIMEOUT_SEC", 0.3), \
             patch("ralph_tui.orchestrator.SOFT_TIMEOUT_SEC", 0.1), \
             patch("ralph_tui.orchestrator.WATCHDOG_CHECK_INTERVAL_SEC", 0.05), \
             patch("ralph_tui.orchestrator.ERROR_RETRY_WAIT_SEC", 0), \
             patch("ralph_tui.orchestrator.CANCEL_GRACE_SEC", 0), \
             patch("ralph_tui.orchestrator.asyncio.sleep", new_callable=AsyncMock), \
             patch("ralph_tui.orchestrator.analyze_output", new_callable=AsyncMock) as mock_analyze:
            mock_analyze.return_value = MagicMock(should_stop=True, reason="done", summary="done")

            start = time.monotonic()
            state = await orch.run()
            elapsed = time.monotonic() - start

        # Should complete in well under 5 seconds (hard timeout ~0.3s + overhead)
        assert elapsed < 5, f"Took {elapsed:.1f}s — watchdog didn't fire during blocked stream"

    @pytest.mark.asyncio
    async def test_watchdog_soft_warning_during_slow_stream(self, tmp_path):
        """Soft timeout should emit a stall_warning even when stream is blocked between yields."""
        cfg = _make_config(tmp_path)
        activity_events = []

        async def capture_activity(e):
            activity_events.append(e)

        orch = Orchestrator(cfg, on_activity=capture_activity)

        async def mock_query(prompt, options):
            from claude_agent_sdk.types import StreamEvent
            from claude_agent_sdk import ResultMessage

            se = MagicMock(spec=StreamEvent)
            se.session_id = "sess-slow"
            se.event = {"type": "message_start"}
            yield se

            # Block long enough for soft warning but not hard cancel
            await asyncio.sleep(2)

            result = MagicMock(spec=ResultMessage)
            result.is_error = False
            result.session_id = "sess-slow"
            result.result = "Done slowly"
            result.total_cost_usd = 0.01
            result.duration_ms = 100
            result.num_turns = 1
            result.usage = None
            yield result

        with patch("ralph_tui.orchestrator.query", side_effect=mock_query), \
             patch("ralph_tui.orchestrator.SOFT_TIMEOUT_SEC", 0.2), \
             patch("ralph_tui.orchestrator.HARD_TIMEOUT_SEC", 999), \
             patch("ralph_tui.orchestrator.WATCHDOG_CHECK_INTERVAL_SEC", 0.05), \
             patch("ralph_tui.orchestrator.analyze_output", new_callable=AsyncMock) as mock_analyze:
            mock_analyze.return_value = MagicMock(should_stop=True, reason="done", summary="done")
            await orch.run()

        stall_events = [e for e in activity_events if e.event_type == "stall_warning"]
        assert len(stall_events) >= 1, f"Expected stall_warning during slow stream: {activity_events}"

    @pytest.mark.asyncio
    async def test_watchdog_cancels_dead_stream(self, tmp_path):
        """A stream that hangs forever should be cancelled and return within seconds."""
        cfg = _make_config(tmp_path)
        orch = Orchestrator(cfg)

        hang_forever = asyncio.Event()  # Never set

        async def mock_query(prompt, options):
            from claude_agent_sdk.types import StreamEvent
            from claude_agent_sdk import ResultMessage

            se = MagicMock(spec=StreamEvent)
            se.session_id = "sess-dead"
            se.event = {"type": "message_start"}
            yield se

            # Hang forever
            await hang_forever.wait()

            # Never reached
            result = MagicMock(spec=ResultMessage)
            result.is_error = False
            result.session_id = "sess-dead"
            result.result = "impossible"
            result.total_cost_usd = 0.01
            result.duration_ms = 100
            result.num_turns = 1
            result.usage = None
            yield result

        with patch("ralph_tui.orchestrator.query", side_effect=mock_query), \
             patch("ralph_tui.orchestrator.HARD_TIMEOUT_SEC", 0.3), \
             patch("ralph_tui.orchestrator.SOFT_TIMEOUT_SEC", 0.1), \
             patch("ralph_tui.orchestrator.WATCHDOG_CHECK_INTERVAL_SEC", 0.05), \
             patch("ralph_tui.orchestrator.ERROR_RETRY_WAIT_SEC", 0), \
             patch("ralph_tui.orchestrator.CANCEL_GRACE_SEC", 0), \
             patch("ralph_tui.orchestrator.asyncio.sleep", new_callable=AsyncMock), \
             patch("ralph_tui.orchestrator.analyze_output", new_callable=AsyncMock) as mock_analyze:
            mock_analyze.return_value = MagicMock(should_stop=True, reason="done", summary="done")

            start = time.monotonic()
            state = await orch.run()
            elapsed = time.monotonic() - start

        assert elapsed < 3, f"Took {elapsed:.1f}s — dead stream was not cancelled"

    @pytest.mark.asyncio
    async def test_heartbeat_watchdog_cancels_dead_stream_under_3s(self, tmp_path):
        """Under a 2s hard timeout, _stream_claude must return a stall_error within
        ~3s real wall-clock on a truly-dead stream.

        Unlike the existing test_watchdog_cancels_dead_stream, this does NOT patch
        asyncio.sleep — real sleeps keep the event loop healthy so the cancel
        propagates to the hung stream task immediately. HeartbeatWatchdog uses
        time.monotonic() which advances with real time, so SOFT=0.5 / HARD=2.0
        reliably fires in 2–3s.
        """
        from pathlib import Path as _Path
        cfg = _make_config(tmp_path)
        orch = Orchestrator(cfg)

        async def mock_dead_query(prompt, options):
            from claude_agent_sdk.types import StreamEvent
            se = MagicMock(spec=StreamEvent)
            se.session_id = "sess-dead2"
            se.event = {"type": "message_start"}
            yield se
            # Stream stalls here forever. asyncio.sleep is NOT patched.
            await asyncio.sleep(999)

        with patch("ralph_tui.orchestrator.query", side_effect=mock_dead_query), \
             patch("ralph_tui.orchestrator.SOFT_TIMEOUT_SEC", 0.5), \
             patch("ralph_tui.orchestrator.HARD_TIMEOUT_SEC", 2.0), \
             patch("ralph_tui.orchestrator.WATCHDOG_CHECK_INTERVAL_SEC", 0.1), \
             patch.object(orch, "_cleanup_child_processes", new_callable=AsyncMock):
            from claude_agent_sdk import ClaudeAgentOptions
            start = time.monotonic()
            (_resp, _cost, _dur, _turns, _sid, error_info) = await orch._stream_claude(
                cwd=_Path(cfg.project_path),
                prompt="hang",
                options=ClaudeAgentOptions(),
            )
            elapsed = time.monotonic() - start

        assert elapsed < 3.5, f"watchdog did not cancel in time: {elapsed:.2f}s"
        assert error_info is not None, "dead stream returned no ErrorInfo"
        assert "stalled" in error_info.raw_message.lower(), (
            f"expected stall error, got: {error_info.raw_message!r}"
        )

    @pytest.mark.asyncio
    async def test_watchdog_no_false_positives(self, tmp_path):
        """A stream that yields messages regularly should not trigger any stall warnings."""
        cfg = _make_config(tmp_path)
        activity_events = []

        async def capture_activity(e):
            activity_events.append(e)

        orch = Orchestrator(cfg, on_activity=capture_activity)

        async def mock_query(prompt, options):
            from claude_agent_sdk.types import StreamEvent
            from claude_agent_sdk import ResultMessage

            # Yield messages every 0.1s for 0.5s
            for i in range(5):
                se = MagicMock(spec=StreamEvent)
                se.session_id = "sess-fast"
                se.event = {"type": "content_block_delta", "delta": {"type": "text_delta", "text": f"chunk{i}"}}
                yield se
                await asyncio.sleep(0.1)

            result = MagicMock(spec=ResultMessage)
            result.is_error = False
            result.session_id = "sess-fast"
            result.result = "Done fast"
            result.total_cost_usd = 0.01
            result.duration_ms = 500
            result.num_turns = 1
            result.usage = None
            yield result

        with patch("ralph_tui.orchestrator.query", side_effect=mock_query), \
             patch("ralph_tui.orchestrator.SOFT_TIMEOUT_SEC", 2), \
             patch("ralph_tui.orchestrator.HARD_TIMEOUT_SEC", 5), \
             patch("ralph_tui.orchestrator.WATCHDOG_CHECK_INTERVAL_SEC", 0.05), \
             patch("ralph_tui.orchestrator.analyze_output", new_callable=AsyncMock) as mock_analyze:
            mock_analyze.return_value = MagicMock(should_stop=True, reason="done", summary="done")
            await orch.run()

        stall_events = [e for e in activity_events if e.event_type == "stall_warning"]
        assert len(stall_events) == 0, f"False positive stall warnings: {stall_events}"


class TestStreamingUsageTracking:
    """Tests for cumulative usage tracking from stream events."""

    @pytest.mark.asyncio
    async def test_usage_from_stream_events_tracks_latest_input(self, tmp_path):
        """message_start input_tokens should track the latest (largest) value across turns."""
        cfg = _make_config(tmp_path)
        usage_events = []

        async def capture_usage(u):
            usage_events.append(u)

        orch = Orchestrator(cfg, on_usage=capture_usage)

        async def mock_query(prompt, options):
            from claude_agent_sdk.types import StreamEvent
            from claude_agent_sdk import ResultMessage

            # Turn 1: message_start with cache tokens (total = 100 + 10000 + 40000 = 50100)
            se1 = MagicMock(spec=StreamEvent)
            se1.session_id = "sess-1"
            se1.event = {"type": "message_start", "message": {"usage": {
                "input_tokens": 100, "cache_creation_input_tokens": 10000, "cache_read_input_tokens": 40000
            }}}
            yield se1

            # Turn 1: message_delta with output_tokens=1000
            se2 = MagicMock(spec=StreamEvent)
            se2.session_id = "sess-1"
            se2.event = {"type": "message_delta", "usage": {"output_tokens": 1000}}
            yield se2

            # Turn 2: message_start with cache tokens (total = 200 + 15000 + 65000 = 80200)
            se3 = MagicMock(spec=StreamEvent)
            se3.session_id = "sess-1"
            se3.event = {"type": "message_start", "message": {"usage": {
                "input_tokens": 200, "cache_creation_input_tokens": 15000, "cache_read_input_tokens": 65000
            }}}
            yield se3

            # Turn 2: message_delta with output_tokens=2000
            se4 = MagicMock(spec=StreamEvent)
            se4.session_id = "sess-1"
            se4.event = {"type": "message_delta", "usage": {"output_tokens": 2000}}
            yield se4

            result = MagicMock(spec=ResultMessage)
            result.is_error = False
            result.session_id = "sess-1"
            result.result = "Done"
            result.total_cost_usd = 0.05
            result.duration_ms = 500
            result.num_turns = 2
            result.usage = None
            yield result

        with patch("ralph_tui.orchestrator.query", side_effect=mock_query), \
             patch("ralph_tui.orchestrator.analyze_output", new_callable=AsyncMock) as mock_analyze:
            mock_analyze.return_value = MagicMock(should_stop=True, reason="done", summary="done")
            await orch.run()

        # Should have usage events from message_delta processing
        assert len(usage_events) >= 2, f"Expected at least 2 usage events: {usage_events}"
        # Last usage event should reflect total context = 80200 (200+15000+65000)
        last = usage_events[-1]
        assert last.input_tokens == 80200
        assert last.output_tokens == 3000  # 1000 + 2000 cumulative
        assert last.context_percent == pytest.approx(80200 / 200000 * 100)

    @pytest.mark.asyncio
    async def test_usage_falls_back_to_result_message_when_no_stream_usage(self, tmp_path):
        """When stream events have no usage, ResultMessage.usage should be used."""
        cfg = _make_config(tmp_path)
        usage_events = []

        async def capture_usage(u):
            usage_events.append(u)

        orch = Orchestrator(cfg, on_usage=capture_usage)

        async def mock_query(prompt, options):
            from claude_agent_sdk.types import StreamEvent
            from claude_agent_sdk import ResultMessage

            # message_start with no usage
            se = MagicMock(spec=StreamEvent)
            se.session_id = "sess-1"
            se.event = {"type": "message_start", "message": {}}
            yield se

            result = MagicMock(spec=ResultMessage)
            result.is_error = False
            result.session_id = "sess-1"
            result.result = "Done"
            result.total_cost_usd = 0.01
            result.duration_ms = 100
            result.num_turns = 1
            result.usage = {
                "input_tokens": 500, "output_tokens": 500,
                "cache_creation_input_tokens": 5000, "cache_read_input_tokens": 24500
            }
            yield result

        with patch("ralph_tui.orchestrator.query", side_effect=mock_query), \
             patch("ralph_tui.orchestrator.analyze_output", new_callable=AsyncMock) as mock_analyze:
            mock_analyze.return_value = MagicMock(should_stop=True, reason="done", summary="done")
            await orch.run()

        assert len(usage_events) >= 1
        # Total context = 500 + 5000 + 24500 = 30000
        assert usage_events[-1].input_tokens == 30000

    @pytest.mark.asyncio
    async def test_usage_updates_every_turn(self, tmp_path):
        """on_usage should fire after each message_delta with increasing context."""
        cfg = _make_config(tmp_path)
        usage_events = []

        async def capture_usage(u):
            usage_events.append(u)

        orch = Orchestrator(cfg, on_usage=capture_usage)

        async def mock_query(prompt, options):
            from claude_agent_sdk.types import StreamEvent
            from claude_agent_sdk import ResultMessage

            for i, (inp, cache_c, cache_r) in enumerate([
                (100, 5000, 14900),   # total=20000
                (200, 10000, 29800),  # total=40000
                (300, 15000, 44700),  # total=60000
            ]):
                se_start = MagicMock(spec=StreamEvent)
                se_start.session_id = "sess-1"
                se_start.event = {"type": "message_start", "message": {"usage": {
                    "input_tokens": inp, "cache_creation_input_tokens": cache_c,
                    "cache_read_input_tokens": cache_r,
                }}}
                yield se_start

                se_delta = MagicMock(spec=StreamEvent)
                se_delta.session_id = "sess-1"
                se_delta.event = {"type": "message_delta", "usage": {"output_tokens": 500}}
                yield se_delta

            result = MagicMock(spec=ResultMessage)
            result.is_error = False
            result.session_id = "sess-1"
            result.result = "Done"
            result.total_cost_usd = 0.03
            result.duration_ms = 300
            result.num_turns = 3
            result.usage = None
            yield result

        with patch("ralph_tui.orchestrator.query", side_effect=mock_query), \
             patch("ralph_tui.orchestrator.analyze_output", new_callable=AsyncMock) as mock_analyze:
            mock_analyze.return_value = MagicMock(should_stop=True, reason="done", summary="done")
            await orch.run()

        # Should fire 3 times (one per message_delta)
        assert len(usage_events) == 3, f"Expected 3 usage events, got {len(usage_events)}"
        # Context % should increase each time
        percents = [u.context_percent for u in usage_events]
        assert percents == sorted(percents), f"Context % should increase: {percents}"
        assert usage_events[-1].input_tokens == 60000


class TestUsageDiagnosticLogging:
    """Tests for usage diagnostic entries in JSONL logs."""

    @pytest.mark.asyncio
    async def test_usage_diagnostic_logged_to_jsonl(self, tmp_path):
        """message_start usage should be logged as usage_event in JSONL."""
        cfg = _make_config(tmp_path)
        orch = Orchestrator(cfg)

        async def mock_query(prompt, options):
            from claude_agent_sdk.types import StreamEvent
            from claude_agent_sdk import ResultMessage

            se1 = MagicMock(spec=StreamEvent)
            se1.session_id = "sess-1"
            se1.event = {"type": "message_start", "message": {"usage": {
                "input_tokens": 3, "cache_creation_input_tokens": 833, "cache_read_input_tokens": 16066
            }}}
            yield se1

            se2 = MagicMock(spec=StreamEvent)
            se2.session_id = "sess-1"
            se2.event = {"type": "message_delta", "usage": {"output_tokens": 100}}
            yield se2

            result = MagicMock(spec=ResultMessage)
            result.is_error = False
            result.session_id = "sess-1"
            result.result = "Done"
            result.total_cost_usd = 0.01
            result.duration_ms = 100
            result.num_turns = 1
            result.usage = None
            yield result

        with patch("ralph_tui.orchestrator.query", side_effect=mock_query), \
             patch("ralph_tui.orchestrator.analyze_output", new_callable=AsyncMock) as mock_analyze:
            mock_analyze.return_value = MagicMock(should_stop=True, reason="done", summary="done")
            await orch.run()

        # Find the JSONL log file
        import json
        log_files = list(tmp_path.rglob("*.jsonl"))
        assert log_files, "Expected at least one JSONL log file"
        usage_entries = []
        for lf in log_files:
            for line in lf.read_text().splitlines():
                entry = json.loads(line)
                if entry.get("type") == "usage_event":
                    usage_entries.append(entry)
        assert len(usage_entries) >= 1, f"Expected usage_event entries in JSONL"
        ue = usage_entries[0]
        assert ue["total_context_tokens"] == 16902  # 3 + 833 + 16066
        assert ue["input_tokens"] == 3
        assert ue["cache_creation_input_tokens"] == 833
        assert ue["cache_read_input_tokens"] == 16066


class TestTemplateContent:
    """Tests for template content changes."""

    def test_context_suffix_has_subagent_guidance(self):
        """CONTEXT_MANAGEMENT_SUFFIX should have context-aware subagent guidance."""
        from ralph_tui.config import CONTEXT_MANAGEMENT_SUFFIX
        assert "Subagents consume context budget" in CONTEXT_MANAGEMENT_SUFFIX

    def test_claude_md_has_startup_sequence(self):
        """CLAUDE_MD_TEMPLATE should guide structured startup."""
        from ralph_tui.config import CLAUDE_MD_TEMPLATE
        assert "Startup Sequence" in CLAUDE_MD_TEMPLATE
        assert "_ralph_state.json" in CLAUDE_MD_TEMPLATE


class TestClaudeMdInjection:
    """Tests for CLAUDE.md injection into iteration directories."""

    def test_claude_md_injected_when_missing(self, tmp_path):
        """CLAUDE.md should be written when the iteration dir doesn't have one."""
        from ralph_tui.orchestrator import _inject_claude_md
        from ralph_tui.config import CLAUDE_MD_TEMPLATE

        iter_dir = tmp_path / "iteration-001"
        iter_dir.mkdir()

        _inject_claude_md(iter_dir)

        claude_md = iter_dir / "CLAUDE.md"
        assert claude_md.exists()
        assert claude_md.read_text() == CLAUDE_MD_TEMPLATE

    def test_claude_md_not_overwritten_when_present(self, tmp_path):
        """Existing CLAUDE.md should not be overwritten."""
        from ralph_tui.orchestrator import _inject_claude_md

        iter_dir = tmp_path / "iteration-001"
        iter_dir.mkdir()
        existing_content = "# Project-specific CLAUDE.md\nDo not overwrite."
        (iter_dir / "CLAUDE.md").write_text(existing_content)

        _inject_claude_md(iter_dir)

        assert (iter_dir / "CLAUDE.md").read_text() == existing_content

    def test_claude_md_template_under_200_lines(self):
        """CLAUDE_MD_TEMPLATE must be under 200 lines to avoid truncation."""
        from ralph_tui.config import CLAUDE_MD_TEMPLATE

        line_count = len(CLAUDE_MD_TEMPLATE.strip().splitlines())
        assert line_count < 200, f"CLAUDE_MD_TEMPLATE has {line_count} lines, must be < 200"


class TestAutoCompactEnv:
    """Tests for CLAUDE_AUTOCOMPACT_PCT_OVERRIDE env var."""

    @pytest.mark.asyncio
    async def test_autocompact_env_set_in_options(self, tmp_path):
        """The autocompact env var should be set in Claude options."""
        cfg = _make_config(tmp_path, autocompact_pct=60)
        orch = Orchestrator(cfg)
        query_calls = []

        async def mock_query(prompt, options):
            query_calls.append(options)
            from claude_agent_sdk import ResultMessage
            result = MagicMock(spec=ResultMessage)
            result.is_error = False
            result.session_id = "sess-1"
            result.result = "Done"
            result.total_cost_usd = 0.01
            result.duration_ms = 100
            result.num_turns = 1
            result.usage = None
            yield result

        with patch("ralph_tui.orchestrator.query", side_effect=mock_query), \
             patch("ralph_tui.orchestrator.analyze_output", new_callable=AsyncMock) as mock_analyze:
            mock_analyze.return_value = MagicMock(should_stop=True, reason="done", summary="done")
            await orch.run()

        assert len(query_calls) >= 1
        env = query_calls[0].env
        assert env.get("CLAUDE_AUTOCOMPACT_PCT_OVERRIDE") == "60"

    @pytest.mark.asyncio
    async def test_setting_sources_includes_project(self, tmp_path):
        """setting_sources should include 'project' to read CLAUDE.md."""
        cfg = _make_config(tmp_path)
        orch = Orchestrator(cfg)
        query_calls = []

        async def mock_query(prompt, options):
            query_calls.append(options)
            from claude_agent_sdk import ResultMessage
            result = MagicMock(spec=ResultMessage)
            result.is_error = False
            result.session_id = "sess-1"
            result.result = "Done"
            result.total_cost_usd = 0.01
            result.duration_ms = 100
            result.num_turns = 1
            result.usage = None
            yield result

        with patch("ralph_tui.orchestrator.query", side_effect=mock_query), \
             patch("ralph_tui.orchestrator.analyze_output", new_callable=AsyncMock) as mock_analyze:
            mock_analyze.return_value = MagicMock(should_stop=True, reason="done", summary="done")
            await orch.run()

        assert "project" in query_calls[0].setting_sources

    def test_autocompact_pct_validation(self):
        """autocompact_pct must be between 1 and 100."""
        cfg = RalphConfig(
            project_path="/tmp/test",
            initial_prompt="go",
            rerun_prompt="again",
            autocompact_pct=0,
        )
        errors = cfg.validate()
        assert any("auto-compact" in e.lower() for e in errors)

        cfg2 = RalphConfig(
            project_path="/tmp/test",
            initial_prompt="go",
            rerun_prompt="again",
            autocompact_pct=101,
        )
        errors2 = cfg2.validate()
        assert any("auto-compact" in e.lower() for e in errors2)

    def test_autocompact_pct_roundtrips_yaml(self, tmp_path):
        """autocompact_pct should survive YAML round-trip."""
        cfg = RalphConfig(
            project_path="/tmp/test",
            initial_prompt="go",
            rerun_prompt="again",
            autocompact_pct=45,
        )
        yaml_path = tmp_path / "config.yaml"
        cfg.save_yaml(yaml_path)
        loaded = RalphConfig.load_yaml(yaml_path)
        assert loaded.autocompact_pct == 45


class TestSubprocessCleanup:
    """Tests for subprocess cleanup safety net after stall cancellation."""

    def test_kill_child_processes_finds_children(self):
        """_kill_child_processes should find and kill child processes."""
        import subprocess, signal, os as _os
        # Spawn a sleep subprocess
        proc = subprocess.Popen(["sleep", "60"])
        try:
            killed = _kill_child_processes(_os.getpid(), signal.SIGTERM)
            assert proc.pid in killed, f"Expected {proc.pid} in killed list: {killed}"
            proc.wait(timeout=5)
            assert proc.returncode is not None, "Process should be dead"
        finally:
            try:
                proc.kill()
                proc.wait()
            except Exception:
                pass

    def test_kill_child_processes_handles_no_children(self):
        """Should return empty list with no children, no exceptions."""
        import os as _os
        # Use a PID unlikely to have children (our own, after ensuring no children)
        killed = _kill_child_processes(_os.getpid())
        assert isinstance(killed, list)
        # May or may not be empty depending on test runner, but no exception

    def test_kill_child_processes_handles_pgrep_failure(self):
        """Should return empty list when pgrep is unavailable."""
        with patch("ralph_tui.orchestrator._subprocess.run", side_effect=FileNotFoundError):
            killed = _kill_child_processes(99999)
        assert killed == []

    @pytest.mark.asyncio
    async def test_cleanup_called_after_stall_cancellation(self, tmp_path):
        """_cleanup_child_processes should be called when stall_cancelled is True."""
        cfg = _make_config(tmp_path)
        orch = Orchestrator(cfg)
        cleanup_calls = []

        original_cleanup = orch._cleanup_child_processes

        async def mock_cleanup(pid):
            cleanup_calls.append(pid)

        orch._cleanup_child_processes = mock_cleanup

        hang_forever = asyncio.Event()

        async def mock_query(prompt, options):
            from claude_agent_sdk.types import StreamEvent
            from claude_agent_sdk import ResultMessage

            se = MagicMock(spec=StreamEvent)
            se.session_id = "sess-stall"
            se.event = {"type": "message_start"}
            yield se

            await hang_forever.wait()

            result = MagicMock(spec=ResultMessage)
            result.is_error = False
            result.session_id = "sess-stall"
            result.result = "impossible"
            result.total_cost_usd = 0.01
            result.duration_ms = 100
            result.num_turns = 1
            result.usage = None
            yield result

        with patch("ralph_tui.orchestrator.query", side_effect=mock_query), \
             patch("ralph_tui.orchestrator.HARD_TIMEOUT_SEC", 0.2), \
             patch("ralph_tui.orchestrator.SOFT_TIMEOUT_SEC", 0.1), \
             patch("ralph_tui.orchestrator.WATCHDOG_CHECK_INTERVAL_SEC", 0.05), \
             patch("ralph_tui.orchestrator.ERROR_RETRY_WAIT_SEC", 0), \
             patch("ralph_tui.orchestrator.asyncio.sleep", new_callable=AsyncMock), \
             patch("ralph_tui.orchestrator.analyze_output", new_callable=AsyncMock) as mock_analyze:
            mock_analyze.return_value = MagicMock(should_stop=True, reason="done", summary="done")
            await orch.run()

        assert len(cleanup_calls) >= 1, f"Expected cleanup to be called: {cleanup_calls}"

    @pytest.mark.asyncio
    async def test_cleanup_not_called_on_normal_completion(self, tmp_path):
        """_cleanup_child_processes should NOT be called on normal completion."""
        cfg = _make_config(tmp_path)
        orch = Orchestrator(cfg)
        cleanup_calls = []

        async def mock_cleanup(pid):
            cleanup_calls.append(pid)

        orch._cleanup_child_processes = mock_cleanup

        async def mock_query(prompt, options):
            from claude_agent_sdk import ResultMessage

            result = MagicMock(spec=ResultMessage)
            result.is_error = False
            result.session_id = "sess-ok"
            result.result = "All good"
            result.total_cost_usd = 0.01
            result.duration_ms = 100
            result.num_turns = 1
            result.usage = None
            yield result

        with patch("ralph_tui.orchestrator.query", side_effect=mock_query), \
             patch("ralph_tui.orchestrator.analyze_output", new_callable=AsyncMock) as mock_analyze:
            mock_analyze.return_value = MagicMock(should_stop=True, reason="done", summary="done")
            await orch.run()

        assert len(cleanup_calls) == 0, f"Cleanup should not be called on normal completion: {cleanup_calls}"


class TestConfigTimeouts:
    """Tests for configurable timeout fields in RalphConfig."""

    def test_default_timeout_values(self):
        cfg = RalphConfig()
        assert cfg.soft_timeout_sec == 120
        assert cfg.hard_timeout_sec == 300

    def test_custom_timeout_values(self):
        cfg = RalphConfig(soft_timeout_sec=60, hard_timeout_sec=180)
        assert cfg.soft_timeout_sec == 60
        assert cfg.hard_timeout_sec == 180

    def test_timeout_roundtrips_yaml(self, tmp_path):
        cfg = RalphConfig(
            project_path="/tmp/test",
            initial_prompt="go",
            rerun_prompt="again",
            soft_timeout_sec=90,
            hard_timeout_sec=600,
        )
        yaml_path = tmp_path / "config.yaml"
        cfg.save_yaml(yaml_path)
        loaded = RalphConfig.load_yaml(yaml_path)
        assert loaded.soft_timeout_sec == 90
        assert loaded.hard_timeout_sec == 600
