"""Behavioral tests for rate-limit detection and retry-time parsing."""

from datetime import datetime, timedelta
from unittest.mock import MagicMock

import pytest

from ralph_tui.rate_limit import (
    RateLimitInfo,
    detect_rate_limit,
    parse_retry_time,
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
