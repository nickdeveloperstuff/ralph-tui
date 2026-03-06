"""Tests for comprehensive error detection and classification."""

from unittest.mock import MagicMock

import pytest

from ralph_tui.error_handling import ErrorType, ErrorInfo, detect_error


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


class TestDetectErrorFromAssistantMessage:
    """Test detection of errors from AssistantMessage.error field."""

    def test_detects_authentication_failed_error(self):
        messages = [_make_assistant_message(error="authentication_failed")]
        result = _make_result_message(is_error=True)
        info = detect_error(messages, result)
        assert info is not None
        assert info.type == ErrorType.AUTH
        assert info.retryable is False

    def test_detects_billing_error(self):
        messages = [_make_assistant_message(error="billing_error")]
        result = _make_result_message(is_error=True)
        info = detect_error(messages, result)
        assert info is not None
        assert info.type == ErrorType.BILLING
        assert info.retryable is False

    def test_detects_rate_limit_error(self):
        messages = [_make_assistant_message(error="rate_limit")]
        result = _make_result_message(is_error=True, session_id="sess-rl")
        info = detect_error(messages, result)
        assert info is not None
        assert info.type == ErrorType.RATE_LIMIT
        assert info.retryable is True

    def test_detects_server_error(self):
        messages = [_make_assistant_message(error="server_error")]
        result = _make_result_message(is_error=True)
        info = detect_error(messages, result)
        assert info is not None
        assert info.type == ErrorType.SERVER
        assert info.retryable is True

    def test_detects_invalid_request_error(self):
        messages = [_make_assistant_message(error="invalid_request")]
        result = _make_result_message(is_error=True)
        info = detect_error(messages, result)
        assert info is not None
        assert info.type == ErrorType.INVALID_REQUEST
        assert info.retryable is False

    def test_detects_unknown_error(self):
        messages = [_make_assistant_message(error="unknown")]
        result = _make_result_message(is_error=True)
        info = detect_error(messages, result)
        assert info is not None
        assert info.type == ErrorType.UNKNOWN
        assert info.retryable is True


class TestDetectErrorFromException:
    """Test detection of errors from SDK exceptions."""

    def test_detects_process_error_exception(self):
        from claude_agent_sdk import ProcessError
        exc = ProcessError("CLI crashed", exit_code=1)
        info = detect_error([], None, exception=exc)
        assert info is not None
        assert info.type == ErrorType.PROCESS
        assert info.retryable is True

    def test_detects_cli_connection_error(self):
        from claude_agent_sdk import CLIConnectionError
        exc = CLIConnectionError("Connection refused")
        info = detect_error([], None, exception=exc)
        assert info is not None
        assert info.type == ErrorType.CONNECTION
        assert info.retryable is True

    def test_detects_cli_not_found(self):
        from claude_agent_sdk import CLINotFoundError
        exc = CLINotFoundError("claude not in PATH")
        info = detect_error([], None, exception=exc)
        assert info is not None
        assert info.type == ErrorType.CLI_NOT_FOUND
        assert info.retryable is False

    def test_detects_json_decode_error(self):
        from claude_agent_sdk import CLIJSONDecodeError
        exc = CLIJSONDecodeError("Invalid JSON", original_error=ValueError("bad"))
        info = detect_error([], None, exception=exc)
        assert info is not None
        assert info.type == ErrorType.PARSE
        assert info.retryable is True

    def test_detects_message_parse_error(self):
        from claude_agent_sdk._errors import MessageParseError
        exc = MessageParseError("Bad message format")
        info = detect_error([], None, exception=exc)
        assert info is not None
        assert info.type == ErrorType.PARSE
        assert info.retryable is True


class TestDetectErrorEdgeCases:
    """Test edge cases and special detection paths."""

    def test_no_error_returns_none(self):
        messages = [_make_assistant_message(error=None, text="All good!")]
        result = _make_result_message(is_error=False)
        info = detect_error(messages, result)
        assert info is None

    def test_result_message_context_exhausted(self):
        """ResultMessage.is_error=True with context/token keywords → CONTEXT_EXHAUSTED."""
        messages = []
        result = _make_result_message(
            is_error=True,
            result="Conversation too long - context window token limit exceeded",
        )
        info = detect_error(messages, result)
        assert info is not None
        assert info.type == ErrorType.CONTEXT_EXHAUSTED
        assert info.retryable is True

    def test_result_message_context_exhausted_case_insensitive(self):
        messages = []
        result = _make_result_message(
            is_error=True,
            result="Error: Context limit reached, please start a new conversation",
        )
        info = detect_error(messages, result)
        assert info is not None
        assert info.type == ErrorType.CONTEXT_EXHAUSTED

    def test_no_messages_no_result_no_exception_returns_none(self):
        info = detect_error([], None)
        assert info is None

    def test_error_info_includes_raw_message(self):
        messages = [_make_assistant_message(error="server_error", text="Internal failure")]
        result = _make_result_message(is_error=True)
        info = detect_error(messages, result)
        assert info is not None
        assert "Internal failure" in info.raw_message

    def test_error_info_includes_session_id(self):
        messages = [_make_assistant_message(error="server_error")]
        result = _make_result_message(is_error=True, session_id="sess-xyz")
        info = detect_error(messages, result)
        assert info is not None
        assert info.session_id == "sess-xyz"

    def test_rate_limit_includes_retry_at(self):
        """Rate limit errors should have a retry_at datetime."""
        from datetime import datetime, timedelta

        messages = [_make_assistant_message(error="rate_limit", text="Try again in 5 minutes")]
        result = _make_result_message(is_error=True, session_id="sess-rl")
        info = detect_error(messages, result)
        assert info is not None
        assert info.retry_at is not None
        # Should be ~5 min + 3 min buffer from now
        assert info.retry_at > datetime.now()


class TestExceptionPatternMatching:
    """Test pattern matching on generic exception messages."""

    def test_rate_limit_in_generic_exception_classified(self):
        exc = Exception("API rate limit exceeded, please retry later")
        info = detect_error([], None, exception=exc)
        assert info is not None
        assert info.type == ErrorType.RATE_LIMIT
        assert info.retryable is True

    def test_overloaded_503_classified_as_server(self):
        exc = Exception("HTTP 503 Service Unavailable - server overloaded")
        info = detect_error([], None, exception=exc)
        assert info is not None
        assert info.type == ErrorType.SERVER
        assert info.retryable is True

    def test_529_classified_as_server(self):
        exc = Exception("Error 529: API overloaded")
        info = detect_error([], None, exception=exc)
        assert info is not None
        assert info.type == ErrorType.SERVER

    def test_context_exhaustion_in_exception_classified(self):
        exc = Exception("Conversation too long - context window token limit exceeded")
        info = detect_error([], None, exception=exc)
        assert info is not None
        assert info.type == ErrorType.CONTEXT_EXHAUSTED
        assert info.retryable is True

    def test_truly_unknown_exception_still_works(self):
        exc = Exception("Something completely unexpected")
        info = detect_error([], None, exception=exc)
        assert info is not None
        assert info.type == ErrorType.UNKNOWN
        assert info.retryable is True
