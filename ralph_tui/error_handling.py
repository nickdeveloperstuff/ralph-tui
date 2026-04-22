"""Comprehensive error detection and classification for Claude SDK responses.

Subsumes rate_limit.py's detection logic while adding handling for all SDK
error types and exceptions.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from datetime import datetime

from claude_agent_sdk import (
    ProcessError,
    CLIConnectionError,
    CLINotFoundError,
    CLIJSONDecodeError,
)
from claude_agent_sdk._errors import MessageParseError

from ralph_tui.rate_limit import detect_rate_limit, RateLimitInfo, RateLimitKind


class ErrorType(enum.Enum):
    """Classification of SDK errors."""
    AUTH = "authentication_failed"
    BILLING = "billing_error"
    RATE_LIMIT = "rate_limit"
    SERVER = "server_error"
    INVALID_REQUEST = "invalid_request"
    UNKNOWN = "unknown"
    PROCESS = "process_error"
    CONNECTION = "connection_error"
    CLI_NOT_FOUND = "cli_not_found"
    PARSE = "parse_error"
    CONTEXT_EXHAUSTED = "context_exhausted"


# Maps AssistantMessage.error string values to ErrorType
_ERROR_FIELD_MAP: dict[str, ErrorType] = {
    "authentication_failed": ErrorType.AUTH,
    "billing_error": ErrorType.BILLING,
    "rate_limit": ErrorType.RATE_LIMIT,
    "server_error": ErrorType.SERVER,
    "invalid_request": ErrorType.INVALID_REQUEST,
    "unknown": ErrorType.UNKNOWN,
}

_NON_RETRYABLE = {ErrorType.AUTH, ErrorType.BILLING, ErrorType.INVALID_REQUEST, ErrorType.CLI_NOT_FOUND}


@dataclass
class ErrorInfo:
    """Classified error with retry guidance."""
    type: ErrorType
    retryable: bool
    session_id: str = ""
    raw_message: str = ""
    retry_at: datetime | None = None  # Only set for rate limits
    rate_limit_kind: RateLimitKind | None = None  # Only set for rate limits


def _is_context_exhausted(text: str) -> bool:
    """Check if error text indicates context window exhaustion."""
    lower = text.lower()
    context_keywords = ("context", "token", "conversation too long")
    limit_keywords = ("limit", "exceeded", "too long", "exhausted")
    return any(k in lower for k in context_keywords) and any(k in lower for k in limit_keywords)


def detect_error(
    messages: list,
    result_message: object | None,
    exception: Exception | None = None,
) -> ErrorInfo | None:
    """Detect and classify errors from SDK messages and/or exceptions.

    Args:
        messages: List of AssistantMessage objects from the stream.
        result_message: The ResultMessage, if any.
        exception: An exception caught during streaming, if any.

    Returns:
        ErrorInfo if an error was detected, None otherwise.
    """
    # 1. Exception-based detection takes priority
    if exception is not None:
        return _classify_exception(exception)

    # 2. Check AssistantMessage.error fields
    all_text = ""
    for msg in messages:
        error_field = getattr(msg, "error", None)
        if error_field and error_field in _ERROR_FIELD_MAP:
            error_type = _ERROR_FIELD_MAP[error_field]

            # Collect text for raw_message
            for block in getattr(msg, "content", []):
                text = getattr(block, "text", "")
                if text:
                    all_text += text + "\n"

            session_id = ""
            if result_message is not None:
                session_id = getattr(result_message, "session_id", "") or ""

            info = ErrorInfo(
                type=error_type,
                retryable=error_type not in _NON_RETRYABLE,
                session_id=session_id,
                raw_message=all_text.strip(),
            )

            # For rate limits, parse retry time using existing logic
            if error_type == ErrorType.RATE_LIMIT and result_message is not None:
                rl_info = detect_rate_limit(messages, result_message)
                if rl_info:
                    info.retry_at = rl_info.retry_at
                    info.rate_limit_kind = rl_info.kind

            return info

        # Accumulate text regardless
        for block in getattr(msg, "content", []):
            text = getattr(block, "text", "")
            if text:
                all_text += text + "\n"

    # 3. Check ResultMessage for context exhaustion or other errors
    if result_message is not None and getattr(result_message, "is_error", False):
        result_text = getattr(result_message, "result", "") or ""
        session_id = getattr(result_message, "session_id", "") or ""

        if _is_context_exhausted(result_text):
            return ErrorInfo(
                type=ErrorType.CONTEXT_EXHAUSTED,
                retryable=True,
                session_id=session_id,
                raw_message=result_text,
            )

        # Check for rate limit via result message (delegate to existing logic)
        rl_info = detect_rate_limit(messages, result_message)
        if rl_info:
            return ErrorInfo(
                type=ErrorType.RATE_LIMIT,
                retryable=True,
                session_id=rl_info.session_id,
                raw_message=rl_info.raw_message,
                retry_at=rl_info.retry_at,
                rate_limit_kind=rl_info.kind,
            )

    return None


def _classify_exception(exc: Exception) -> ErrorInfo:
    """Classify an SDK exception into an ErrorInfo."""
    # Order matters: CLINotFoundError is a subclass of CLIConnectionError
    if isinstance(exc, CLINotFoundError):
        return ErrorInfo(
            type=ErrorType.CLI_NOT_FOUND,
            retryable=False,
            raw_message=str(exc),
        )
    if isinstance(exc, CLIConnectionError):
        return ErrorInfo(
            type=ErrorType.CONNECTION,
            retryable=True,
            raw_message=str(exc),
        )
    if isinstance(exc, ProcessError):
        return ErrorInfo(
            type=ErrorType.PROCESS,
            retryable=True,
            raw_message=str(exc),
        )
    if isinstance(exc, (CLIJSONDecodeError, MessageParseError)):
        return ErrorInfo(
            type=ErrorType.PARSE,
            retryable=True,
            raw_message=str(exc),
        )

    # Pattern matching on exception message before falling through to UNKNOWN
    msg = str(exc).lower()
    if "rate" in msg and "limit" in msg:
        return ErrorInfo(
            type=ErrorType.RATE_LIMIT,
            retryable=True,
            raw_message=str(exc),
        )
    if "overloaded" in msg or "529" in msg or "503" in msg:
        return ErrorInfo(
            type=ErrorType.SERVER,
            retryable=True,
            raw_message=str(exc),
        )
    if _is_context_exhausted(str(exc)):
        return ErrorInfo(
            type=ErrorType.CONTEXT_EXHAUSTED,
            retryable=True,
            raw_message=str(exc),
        )

    # Truly unknown exception
    return ErrorInfo(
        type=ErrorType.UNKNOWN,
        retryable=True,
        raw_message=f"{type(exc).__name__}: {exc}",
    )
