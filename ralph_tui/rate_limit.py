"""Rate-limit detection and retry-time parsing for Claude Code SDK responses."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta

BUFFER_MINUTES = 3
DEFAULT_BACKOFF_MINUTES = 15


@dataclass
class RateLimitInfo:
    retry_at: datetime  # When to retry (already includes 3-min buffer)
    session_id: str  # Session to resume
    raw_message: str  # Original text for logging


def detect_rate_limit(messages: list, result) -> RateLimitInfo | None:
    """Check streaming messages + result for rate limit signals.

    Returns RateLimitInfo if a rate limit was detected, None otherwise.
    """
    # Check for rate_limit error on any AssistantMessage
    is_rate_limited = False
    all_text = ""

    for msg in messages:
        if getattr(msg, "error", None) == "rate_limit":
            is_rate_limited = True
        for block in getattr(msg, "content", []):
            text = getattr(block, "text", "")
            if text:
                all_text += text + "\n"

    # Also check ResultMessage
    if getattr(result, "is_error", False):
        result_text = getattr(result, "result", "") or ""
        if "rate" in result_text.lower() and "limit" in result_text.lower():
            is_rate_limited = True
        all_text += result_text

    if not is_rate_limited:
        return None

    session_id = getattr(result, "session_id", "") or ""

    # Try to parse a specific retry time from the text
    parsed_time = parse_retry_time(all_text)
    if parsed_time:
        retry_at = parsed_time + timedelta(minutes=BUFFER_MINUTES)
    else:
        retry_at = datetime.now() + timedelta(minutes=DEFAULT_BACKOFF_MINUTES + BUFFER_MINUTES)

    return RateLimitInfo(
        retry_at=retry_at,
        session_id=session_id,
        raw_message=all_text.strip(),
    )


def parse_retry_time(text: str) -> datetime | None:
    """Extract retry time from Claude's natural language output.

    Handles:
    - ISO format: "resets at 2026-02-28T15:45:00"
    - Absolute time: "available at 3:45 PM"
    - Relative time: "try again in 15 minutes"

    Returns the parsed datetime (WITHOUT buffer — caller adds buffer).
    Returns None if no time pattern is found.
    """
    if not text:
        return None

    # 1. ISO format: 2026-02-28T15:45:00
    iso_match = re.search(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}", text)
    if iso_match:
        return datetime.fromisoformat(iso_match.group())

    # 2. Relative time: "in N minutes", "in N hours"
    relative_match = re.search(r"in\s+(\d+)\s+(minute|min|hour|hr)s?", text, re.IGNORECASE)
    if relative_match:
        amount = int(relative_match.group(1))
        unit = relative_match.group(2).lower()
        if unit in ("hour", "hr"):
            return datetime.now() + timedelta(hours=amount)
        else:
            return datetime.now() + timedelta(minutes=amount)

    # 3. Absolute time: "at 3:45 PM" or "at 15:45"
    absolute_match = re.search(
        r"at\s+(\d{1,2}):(\d{2})\s*(AM|PM)?", text, re.IGNORECASE
    )
    if absolute_match:
        hour = int(absolute_match.group(1))
        minute = int(absolute_match.group(2))
        ampm = absolute_match.group(3)
        if ampm:
            ampm = ampm.upper()
            if ampm == "PM" and hour != 12:
                hour += 12
            elif ampm == "AM" and hour == 12:
                hour = 0
        now = datetime.now()
        return now.replace(hour=hour, minute=minute, second=0, microsecond=0)

    return None
