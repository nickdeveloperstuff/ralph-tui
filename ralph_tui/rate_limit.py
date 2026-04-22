"""Rate-limit detection and retry-time parsing for Claude Code SDK responses."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Literal

BUFFER_MINUTES = 3
DEFAULT_BACKOFF_MINUTES = 15

RateLimitKind = Literal["plan_usage", "api_rate_limit"]

WEEKDAYS = {"mon": 0, "tue": 1, "wed": 2, "thu": 3, "fri": 4, "sat": 5, "sun": 6}

# Detects Claude Code plan-usage exhaustion messages:
#   "You've hit your session limit · resets 3:45pm"
#   "You've hit your weekly limit · resets Mon 12:00am"
#   "You've hit your Opus limit · resets 3:45pm"
PLAN_USAGE_RE = re.compile(
    r"You'?ve hit your\s+(?P<scope>\w+)\s+limit\s*"
    r"[·•\.\-]?\s*"
    r"resets?\s+"
    r"(?P<when>"
        r"(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun)[a-z]*\s+\d{1,2}:\d{2}\s*[ap]m"
        r"|\d{1,2}:\d{2}\s*[ap]m"
        r"|\d{1,2}\s*[ap]m"
    r")",
    re.IGNORECASE,
)


@dataclass
class RateLimitInfo:
    retry_at: datetime  # When to retry (already includes 3-min buffer)
    session_id: str  # Session to resume
    raw_message: str  # Original text for logging
    kind: RateLimitKind = "api_rate_limit"


def _resolve_future_time(hour: int, minute: int, weekday: int | None = None) -> datetime:
    """Resolve a wall-clock HH:MM (optionally on a weekday) to the next future instant."""
    now = datetime.now()
    candidate = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if weekday is not None:
        days_ahead = (weekday - now.weekday()) % 7
        candidate = candidate + timedelta(days=days_ahead)
        if candidate <= now:
            candidate += timedelta(days=7)
    elif candidate <= now:
        candidate += timedelta(days=1)
    return candidate


def _parse_reset_clause(text: str) -> datetime | None:
    """Parse "[Weekday] H[:MM] [a|p]m" into a future datetime."""
    m = re.match(
        r"\s*(?:(?P<wday>mon|tue|wed|thu|fri|sat|sun)[a-z]*\s+)?"
        r"(?P<h>\d{1,2})(?::(?P<m>\d{2}))?\s*(?P<ap>[ap]m)",
        text,
        re.IGNORECASE,
    )
    if not m:
        return None
    h = int(m["h"])
    minute = int(m["m"] or 0)
    ap = m["ap"].lower()
    if ap == "pm" and h != 12:
        h += 12
    if ap == "am" and h == 12:
        h = 0
    wday = WEEKDAYS[m["wday"].lower()[:3]] if m["wday"] else None
    return _resolve_future_time(h, minute, wday)


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

    # Plan-usage detection runs before the generic rate-limit path so we can
    # set kind="plan_usage" and parse the natural-language reset clock.
    plan_match = PLAN_USAGE_RE.search(all_text)
    if plan_match:
        parsed = _parse_reset_clause(plan_match.group("when"))
        if parsed is not None:
            session_id = getattr(result, "session_id", "") or ""
            return RateLimitInfo(
                retry_at=parsed + timedelta(minutes=BUFFER_MINUTES),
                session_id=session_id,
                raw_message=all_text.strip(),
                kind="plan_usage",
            )

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
            return _resolve_future_time(hour, minute)
        # No AM/PM: treat as 24-hour clock, still roll forward if past
        return _resolve_future_time(hour, minute)

    return None
