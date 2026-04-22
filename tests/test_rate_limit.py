"""Behavioral tests for rate-limit detection and retry-time parsing."""

from datetime import datetime, timedelta
from unittest.mock import MagicMock

import pytest
from freezegun import freeze_time

from ralph_tui.rate_limit import (
    RateLimitInfo,
    detect_rate_limit,
    parse_retry_time,
    _parse_reset_clause,
)


def _make_assistant_message(error=None, text=""):
    """Create a mock AssistantMessage with optional error field and text content."""
    msg = MagicMock()
    msg.error = error
    block = MagicMock()
    block.text = text
    msg.content = [block]
    return msg


def _make_result_message(is_error=False, session_id="sess-123", result=""):
    """Create a mock ResultMessage."""
    msg = MagicMock()
    msg.is_error = is_error
    msg.session_id = session_id
    msg.result = result
    return msg


class TestDetectRateLimit:
    def test_detects_rate_limit_from_error_field(self):
        messages = [_make_assistant_message(error="rate_limit")]
        result = _make_result_message(is_error=True, session_id="sess-abc")
        info = detect_rate_limit(messages, result)
        assert info is not None
        assert info.session_id == "sess-abc"

    def test_no_rate_limit_for_normal_messages(self):
        messages = [_make_assistant_message(error=None, text="All done!")]
        result = _make_result_message(is_error=False, session_id="sess-xyz")
        info = detect_rate_limit(messages, result)
        assert info is None


class TestParseRetryTime:
    def test_parses_retry_time_absolute(self):
        # "available at 3:45 PM" should parse to today at 15:45
        text = "Rate limit exceeded. Your quota will be available at 3:45 PM."
        dt = parse_retry_time(text)
        assert dt is not None
        assert dt.hour == 15
        assert dt.minute == 45

    def test_parses_retry_time_relative(self):
        text = "Too many requests. Please try again in 15 minutes."
        before = datetime.now()
        dt = parse_retry_time(text)
        after = datetime.now()
        assert dt is not None
        # Should be ~15 minutes from now (within 2 second tolerance)
        expected_min = before + timedelta(minutes=15) - timedelta(seconds=2)
        expected_max = after + timedelta(minutes=15) + timedelta(seconds=2)
        assert expected_min <= dt <= expected_max

    def test_parses_retry_time_iso_format(self):
        text = "Rate limit resets at 2026-02-28T15:45:00"
        dt = parse_retry_time(text)
        assert dt is not None
        assert dt.year == 2026
        assert dt.month == 2
        assert dt.day == 28
        assert dt.hour == 15
        assert dt.minute == 45

    def test_default_backoff_when_no_time_found(self):
        messages = [_make_assistant_message(error="rate_limit", text="rate limited")]
        result = _make_result_message(is_error=True, session_id="sess-def")
        before = datetime.now()
        info = detect_rate_limit(messages, result)
        after = datetime.now()
        assert info is not None
        # Default: 15 min + 3 min buffer = 18 min from now
        expected_min = before + timedelta(minutes=18) - timedelta(seconds=2)
        expected_max = after + timedelta(minutes=18) + timedelta(seconds=2)
        assert expected_min <= info.retry_at <= expected_max

    def test_buffer_added_to_retry_time(self):
        text = "try again in 10 minutes"
        before = datetime.now()
        dt = parse_retry_time(text)
        after = datetime.now()
        assert dt is not None
        # parse_retry_time returns the raw parsed time WITHOUT buffer.
        # The buffer is applied by detect_rate_limit.
        # So dt should be ~10 min from now.
        expected_min = before + timedelta(minutes=10) - timedelta(seconds=2)
        expected_max = after + timedelta(minutes=10) + timedelta(seconds=2)
        assert expected_min <= dt <= expected_max

        # Now verify detect_rate_limit adds the 3-min buffer
        messages = [_make_assistant_message(error="rate_limit", text=text)]
        result = _make_result_message(is_error=True, session_id="sess-buf")
        before2 = datetime.now()
        info = detect_rate_limit(messages, result)
        after2 = datetime.now()
        assert info is not None
        expected_min2 = before2 + timedelta(minutes=13) - timedelta(seconds=2)
        expected_max2 = after2 + timedelta(minutes=13) + timedelta(seconds=2)
        assert expected_min2 <= info.retry_at <= expected_max2


class TestPlanUsageDetection:
    @freeze_time("2026-04-21 14:00:00")
    def test_detect_plan_usage_session_limit(self):
        """5-hour session limit with absolute-clock reset."""
        text = "You've hit your session limit · resets 3:45pm"
        messages = [_make_assistant_message(error=None, text=text)]
        result = _make_result_message(is_error=False, session_id="sess-s")
        info = detect_rate_limit(messages, result)
        assert info is not None
        assert info.kind == "plan_usage"
        # 15:45 today + 3min buffer = 15:48 today
        assert info.retry_at == datetime(2026, 4, 21, 15, 48, 0)

    @freeze_time("2026-04-22 12:00:00")  # Wednesday
    def test_detect_plan_usage_weekly_limit(self):
        """Weekly limit rolls to the named weekday."""
        text = "You've hit your weekly limit · resets Mon 12:00am"
        messages = [_make_assistant_message(error=None, text=text)]
        result = _make_result_message(is_error=False, session_id="sess-w")
        info = detect_rate_limit(messages, result)
        assert info is not None
        assert info.kind == "plan_usage"
        # 2026-04-22 is a Wednesday. Next Monday 00:00 + 3min buffer.
        assert info.retry_at == datetime(2026, 4, 27, 0, 3, 0)

    @freeze_time("2026-04-21 14:00:00")
    def test_detect_plan_usage_opus_limit(self):
        text = "You've hit your Opus limit · resets 3:45pm"
        messages = [_make_assistant_message(error=None, text=text)]
        result = _make_result_message(is_error=False, session_id="sess-o")
        info = detect_rate_limit(messages, result)
        assert info is not None
        assert info.kind == "plan_usage"
        assert info.retry_at == datetime(2026, 4, 21, 15, 48, 0)

    @freeze_time("2026-04-21 16:00:00")
    def test_absolute_time_rolls_to_tomorrow(self):
        """When now > reset clock, parse_retry_time rolls 'at 3:45pm' to tomorrow."""
        text = "Retry at 3:45 PM"
        dt = parse_retry_time(text)
        assert dt is not None
        assert dt == datetime(2026, 4, 22, 15, 45, 0)

    @freeze_time("2026-04-21 15:45:00")
    def test_reset_time_exactly_now_rolls_forward(self):
        """Boundary: candidate == now must roll forward (uses <=, not <)."""
        parsed = _parse_reset_clause("3:45pm")
        assert parsed == datetime(2026, 4, 22, 15, 45, 0)

    @freeze_time("2026-04-21 14:00:00")
    def test_plan_usage_lowercase_pm_no_space(self):
        """Message format ships tight: '3:45pm' with no space before pm."""
        text = "You've hit your session limit · resets 3:45pm"
        messages = [_make_assistant_message(error=None, text=text)]
        result = _make_result_message(is_error=False, session_id="s")
        info = detect_rate_limit(messages, result)
        assert info is not None and info.kind == "plan_usage"

    def test_api_rate_limit_keeps_kind_api(self):
        """Plain SDK rate_limit with no plan-usage phrase stays kind='api_rate_limit'."""
        messages = [_make_assistant_message(error="rate_limit", text="try again in 5 minutes")]
        result = _make_result_message(is_error=True, session_id="sess-api")
        info = detect_rate_limit(messages, result)
        assert info is not None
        assert info.kind == "api_rate_limit"

    @freeze_time("2026-04-21 09:00:00")
    def test_plan_usage_12am_midnight(self):
        """12:00am must parse to hour=0, minute=0 (midnight)."""
        parsed = _parse_reset_clause("12:00am")
        # Now is 09:00, so next midnight is tomorrow.
        assert parsed == datetime(2026, 4, 22, 0, 0, 0)

    @freeze_time("2026-04-21 09:00:00")
    def test_plan_usage_12pm_noon(self):
        """12:00pm must parse to hour=12 (noon)."""
        parsed = _parse_reset_clause("12:00pm")
        assert parsed == datetime(2026, 4, 21, 12, 0, 0)

    @freeze_time("2026-04-21 09:00:00")
    def test_plan_usage_11pm_end_of_day(self):
        parsed = _parse_reset_clause("11:30pm")
        assert parsed == datetime(2026, 4, 21, 23, 30, 0)

    @freeze_time("2026-04-21 09:00:00")
    def test_plan_usage_bare_hour_no_minutes(self):
        """'7am' without minutes defaults to minute=0."""
        parsed = _parse_reset_clause("7am")
        assert parsed == datetime(2026, 4, 21, 7, 0, 0) + timedelta(days=1)  # past noon, rolls

    @freeze_time("2026-04-21 14:00:00")
    @pytest.mark.parametrize("msg", [
        "You've hit your session limit · resets 3:45pm",
        "You've hit your session limit • resets 3:45pm",
        "You've hit your session limit - resets 3:45pm",
        "You've hit your session limit . resets 3:45pm",
        "You've hit your session limit resets 3:45pm",
        "You've hit your weekly limit · resets Mon 12:00am",
        "You've hit your Opus limit · resets 3:45pm",
        "You've hit your opus limit · resets 3:45PM",   # casing
        "Error occurred. You've hit your session limit · resets 3:45pm. Please wait.",
    ])
    def test_plan_usage_format_variants(self, msg):
        """PLAN_USAGE_RE should handle bullet/dash/period/no-separator variants."""
        messages = [_make_assistant_message(error=None, text=msg)]
        result = _make_result_message(is_error=False, session_id="s")
        info = detect_rate_limit(messages, result)
        assert info is not None, f"failed to detect: {msg!r}"
        assert info.kind == "plan_usage"

    def test_plan_usage_beats_api_rate_limit_when_both_conditions(self):
        """If message has BOTH rate_limit error AND plan_usage phrase, kind is plan_usage."""
        messages = [_make_assistant_message(
            error="rate_limit",
            text="You've hit your session limit · resets 11:59pm",
        )]
        result = _make_result_message(is_error=True, session_id="sess-dual")
        info = detect_rate_limit(messages, result)
        assert info is not None
        assert info.kind == "plan_usage", (
            "plan_usage phrase must take precedence over the api_rate_limit error"
        )
