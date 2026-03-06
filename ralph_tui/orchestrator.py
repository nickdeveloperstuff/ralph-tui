"""Core orchestration loop: copy folder → run Claude → analyze → repeat."""

from __future__ import annotations

import asyncio
import json
import os
import signal
import shutil
import subprocess as _subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Awaitable

from claude_agent_sdk import (
    query,
    ClaudeAgentOptions,
    AssistantMessage,
    ResultMessage,
    TextBlock,
    ToolUseBlock,
    ProcessError,
    CLIConnectionError,
    CLINotFoundError,
    CLIJSONDecodeError,
)
from claude_agent_sdk.types import StreamEvent
from claude_agent_sdk._errors import MessageParseError

from ralph_tui.analyzer import AnalysisResult, analyze_output
from ralph_tui.config import RalphConfig, CONTEXT_MANAGEMENT_SUFFIX, CLAUDE_MD_TEMPLATE
from ralph_tui.error_handling import ErrorType, ErrorInfo, detect_error
from ralph_tui.rate_limit import detect_rate_limit


EXCLUDE_DIRS = {"__pycache__", "node_modules", ".venv", "venv"}
SOFT_TIMEOUT_SEC = 2 * 60    # 2 min -> warning in TUI
HARD_TIMEOUT_SEC = 5 * 60    # 5 min -> cancel + retry
WATCHDOG_CHECK_INTERVAL_SEC = 1  # How often the watchdog task checks for stalls
CANCEL_GRACE_SEC = 5   # wait for SDK cleanup before force-killing children
MAX_RATE_LIMIT_RETRIES = 5
MAX_ERROR_RETRIES = 3
ERROR_RETRY_WAIT_SEC = 30
MODEL_CONTEXT_WINDOW = 200_000  # claude-opus-4-6


def _total_context_tokens(usage: dict) -> int:
    """Sum all input token types to get actual context window consumption."""
    return (usage.get("input_tokens", 0)
            + usage.get("cache_creation_input_tokens", 0)
            + usage.get("cache_read_input_tokens", 0))


@dataclass
class IterationResult:
    iteration: int
    claude_response: str
    cost_usd: float
    duration_ms: int
    num_turns: int
    analysis: AnalysisResult | None  # None if below min_iterations
    skipped_analysis: bool  # True when below min_iterations


@dataclass
class OrchestratorState:
    status: str = "idle"
    current_iteration: int = 0
    total_cost_usd: float = 0.0
    start_time: float = 0.0
    results: list[IterationResult] = field(default_factory=list)


@dataclass
class ActivityEvent:
    """Real-time activity signal from the Claude stream."""
    timestamp: float          # time.monotonic()
    event_type: str           # "text_delta", "tool_start", "tool_end", "message_start", "message_stop", "stall_warning"
    tool_name: str | None = None
    text_fragment: str | None = None


@dataclass
class UsageInfo:
    """Token usage from a completed turn."""
    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0
    total_tokens: int = 0
    context_percent: float = 0.0


class HeartbeatWatchdog:
    """Two-tier watchdog: soft warning then hard cancel."""

    def __init__(self, soft_timeout_sec: int = SOFT_TIMEOUT_SEC, hard_timeout_sec: int = HARD_TIMEOUT_SEC):
        self._soft_timeout = soft_timeout_sec
        self._hard_timeout = hard_timeout_sec
        self._last_ping = time.monotonic()
        self._soft_fired = False

    def ping(self) -> None:
        self._last_ping = time.monotonic()
        self._soft_fired = False

    def elapsed(self) -> float:
        return time.monotonic() - self._last_ping

    def is_soft_stale(self) -> bool:
        if self._soft_fired:
            return False
        if self.elapsed() > self._soft_timeout:
            self._soft_fired = True
            return True
        return False

    def is_hard_stale(self) -> bool:
        return self.elapsed() > self._hard_timeout


# Callback types for TUI integration
StatusCallback = Callable[[str], Awaitable[None]]
TextCallback = Callable[[str], Awaitable[None]]
IterationCallback = Callable[[IterationResult], Awaitable[None]]
ActivityCallback = Callable[[ActivityEvent], Awaitable[None]]
UsageCallback = Callable[[UsageInfo], Awaitable[None]]


def _copy_project(
    src: Path, dst: Path
) -> None:
    """Copy project directory, excluding __pycache__, node_modules, etc."""
    if dst.exists():
        shutil.rmtree(dst)

    def _ignore(directory: str, contents: list[str]) -> set[str]:
        return {c for c in contents if c in EXCLUDE_DIRS}

    shutil.copytree(src, dst, ignore=_ignore)


def _inject_claude_md(iter_dir: Path) -> None:
    """Write CLAUDE.md into the iteration directory if none exists."""
    claude_md = iter_dir / "CLAUDE.md"
    if not claude_md.exists():
        claude_md.write_text(CLAUDE_MD_TEMPLATE)


RALPH_STATE_TEMPLATE = """\
{
  "iteration": 0,
  "phase": "startup",
  "tasks": [],
  "citations_to_verify": [],
  "key_findings": []
}
"""

RALPH_INTERNAL_FILES = {
    "_ralph_state.json", "_document_index.md", "CLAUDE.md",
}


def _inject_state_file(iter_dir: Path) -> None:
    """Write _ralph_state.json template if none exists."""
    state_file = iter_dir / "_ralph_state.json"
    if not state_file.exists():
        state_file.write_text(RALPH_STATE_TEMPLATE)


def _generate_document_index(project_dir: Path, iter_dir: Path) -> None:
    """Walk project_dir and create _document_index.md in iter_dir."""
    index_path = iter_dir / "_document_index.md"
    exclude_prefixes = ("_scratch_", "_ralph_state", "_document_index", "CLAUDE.md")

    # Collect files grouped by directory
    dir_files: dict[str, list[tuple[str, int]]] = {}
    total_files = 0
    total_size = 0

    for path in sorted(project_dir.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(project_dir)
        name = rel.name
        # Skip ralph internal files and scratch files
        if any(name.startswith(p) for p in exclude_prefixes):
            continue
        # Skip hidden dirs like .git
        parts = rel.parts
        if any(p.startswith(".") for p in parts):
            continue
        # Skip excluded dirs
        if any(p in EXCLUDE_DIRS for p in parts):
            continue

        size = path.stat().st_size
        parent = str(rel.parent) if str(rel.parent) != "." else ""
        dir_files.setdefault(parent, []).append((name, size))
        total_files += 1
        total_size += size

    # Build markdown
    def _fmt_size(b: int) -> str:
        if b >= 1_000_000:
            return f"{b / 1_000_000:.1f}MB"
        elif b >= 1_000:
            return f"{b / 1_000:.0f}KB"
        return f"{b}B"

    lines = [
        "# Document Index (auto-generated by Ralph TUI)",
        f"## {total_files} files, {_fmt_size(total_size)} total",
        "",
    ]

    for dir_name in sorted(dir_files.keys()):
        files = dir_files[dir_name]
        dir_size = sum(s for _, s in files)
        header = f"{dir_name}/" if dir_name else "(root)"
        lines.append(f"### {header} ({len(files)} files, {_fmt_size(dir_size)})")
        for fname, fsize in sorted(files):
            lines.append(f"- {fname} ({_fmt_size(fsize)})")
        lines.append("")

    index_path.write_text("\n".join(lines))


def _kill_child_processes(parent_pid: int, sig: int = signal.SIGTERM) -> list[int]:
    """Find and signal all child processes of the given PID.

    Safety net for when the SDK's own cleanup doesn't complete.
    Uses pgrep -P on macOS/Linux. Falls back silently if unavailable.
    """
    killed = []
    try:
        result = _subprocess.run(
            ["pgrep", "-P", str(parent_pid)],
            capture_output=True, text=True, timeout=5,
        )
        for line in result.stdout.strip().splitlines():
            child_pid = int(line.strip())
            try:
                os.kill(child_pid, sig)
                killed.append(child_pid)
            except ProcessLookupError:
                pass
    except (FileNotFoundError, _subprocess.TimeoutExpired, ValueError):
        pass
    return killed


class Orchestrator:
    def __init__(
        self,
        config: RalphConfig,
        on_status: StatusCallback | None = None,
        on_text: TextCallback | None = None,
        on_iteration_done: IterationCallback | None = None,
        on_activity: ActivityCallback | None = None,
        on_usage: UsageCallback | None = None,
    ):
        self.config = config
        self.state = OrchestratorState()
        self._stop_event = asyncio.Event()
        self._on_status = on_status
        self._on_text = on_text
        self._on_iteration_done = on_iteration_done
        self._on_activity = on_activity
        self._on_usage = on_usage
        self._last_usage_info: UsageInfo | None = None
        self._current_log_file: Path | None = None

    async def _notify_status(self, status: str) -> None:
        self.state.status = status
        if self._on_status:
            await self._on_status(status)

    async def _notify_text(self, text: str) -> None:
        if self._on_text:
            await self._on_text(text)

    async def _notify_activity(self, event: ActivityEvent) -> None:
        if self._current_log_file:
            self._log_tool_event(self._current_log_file, event, self.state.current_iteration)
        if self._on_activity:
            await self._on_activity(event)

    async def _notify_usage(self, usage: UsageInfo) -> None:
        self._last_usage_info = usage
        if self._on_usage:
            await self._on_usage(usage)

    async def _cleanup_child_processes(self, parent_pid: int) -> None:
        """Kill child processes after stall cancellation.

        Wait CANCEL_GRACE_SEC for SDK cleanup, then force-kill survivors.
        """
        await asyncio.sleep(CANCEL_GRACE_SEC)
        killed = await asyncio.to_thread(_kill_child_processes, parent_pid, signal.SIGTERM)
        if killed:
            await self._notify_text(f"\n[Watchdog: SIGTERM sent to {len(killed)} orphaned process(es): {killed}]\n")
            if self._current_log_file:
                entry = {
                    "timestamp": datetime.now().isoformat(),
                    "type": "subprocess_cleanup",
                    "iteration": self.state.current_iteration,
                    "pids_killed": killed,
                }
                with open(self._current_log_file, "a") as f:
                    f.write(json.dumps(entry) + "\n")
            # Wait briefly, then SIGKILL any survivors
            await asyncio.sleep(2)
            for pid in killed:
                try:
                    os.kill(pid, 0)  # Check if still alive
                    os.kill(pid, signal.SIGKILL)
                    await self._notify_text(f"\n[Watchdog: force-killed PID {pid}]\n")
                except ProcessLookupError:
                    pass  # Already dead

    def _log_tool_event(self, log_file: Path, event: ActivityEvent, iteration: int) -> None:
        if event.event_type not in ("tool_start", "tool_end", "stall_warning"):
            return
        entry = {
            "timestamp": datetime.now().isoformat(),
            "type": "tool_event",
            "iteration": iteration,
            "event_type": event.event_type,
            "tool_name": event.tool_name,
        }
        with open(log_file, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def _select_prompt(self, iteration: int) -> str:
        """Select the prompt for a given iteration number.

        Verification iterations (if configured) replace normal drafting iterations.

        Three-phase system:
          - Iteration 1: initial_prompt
          - Iterations 2 to (transition_iteration - 1): rerun_prompt
          - Iterations transition_iteration to max: final_prompt

        When transition_iteration is 0 (disabled), falls back to two-prompt mode:
          - Iteration 1: initial_prompt
          - Iterations 2+: rerun_prompt

        All prompts get the context management suffix appended.
        """
        # Verification iterations (if configured)
        if (self.config.verification_interval > 0
                and self.config.verification_prompt
                and iteration > 1
                and iteration % self.config.verification_interval == 0):
            from ralph_tui.config import VERIFICATION_METHODOLOGY_TEMPLATE
            base = VERIFICATION_METHODOLOGY_TEMPLATE.format(
                verification_prompt=self.config.verification_prompt
            )
            return base + CONTEXT_MANAGEMENT_SUFFIX

        if iteration == 1:
            base = self.config.initial_prompt
        elif (
            self.config.transition_iteration > 0
            and iteration >= self.config.transition_iteration
        ):
            base = self.config.final_prompt
        else:
            base = self.config.rerun_prompt
        return base + CONTEXT_MANAGEMENT_SUFFIX

    def stop(self) -> None:
        """Signal the orchestrator to stop after current iteration."""
        self._stop_event.set()

    def _log_iteration(
        self, log_file: Path, result: IterationResult, error_type: str | None = None,
        usage: UsageInfo | None = None,
    ) -> None:
        """Append iteration result as a JSON line to the log file."""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "iteration": result.iteration,
            "status": "error" if error_type else "ok",
            "error_type": error_type,
            "cost_usd": result.cost_usd,
            "duration_ms": result.duration_ms,
            "num_turns": result.num_turns,
            "analysis_summary": (
                result.analysis.summary if result.analysis else None
            ),
            "claude_response_preview": result.claude_response[:500],
        }
        if usage:
            entry["input_tokens"] = usage.input_tokens
            entry["output_tokens"] = usage.output_tokens
            entry["context_percent"] = usage.context_percent
        with open(log_file, "a") as f:
            f.write(json.dumps(entry) + "\n")

    async def run(self) -> OrchestratorState:
        """Execute the full orchestration loop."""
        self.state = OrchestratorState(start_time=time.time())
        project_path = Path(self.config.project_path).expanduser().resolve()
        # Place runs dir as a sibling to avoid recursive copy issues
        runs_dir = project_path.parent / f"{project_path.name}-ralph-runs"
        runs_dir.mkdir(exist_ok=True)
        log_file = runs_dir / "ralph-log.jsonl"
        self._current_log_file = log_file

        for iteration in range(1, self.config.max_iterations + 1):
            if self._stop_event.is_set():
                await self._notify_status("Stopped by user")
                break

            self.state.current_iteration = iteration
            iter_dir = runs_dir / f"iteration-{iteration:03d}"

            # 1. Copy folder
            await self._notify_status(f"Copying project (iteration {iteration})")
            if iteration == 1:
                src = project_path
            else:
                src = runs_dir / f"iteration-{iteration - 1:03d}"

            await asyncio.to_thread(_copy_project, src, iter_dir)
            _inject_claude_md(iter_dir)
            _inject_state_file(iter_dir)

            # Generate document index on first iteration only
            if iteration == 1:
                await asyncio.to_thread(_generate_document_index, project_path, iter_dir)

            # Clean up prior iteration dirs (keep N-1 as fallback)
            if iteration > 2:
                for prev in range(1, iteration - 1):
                    prev_dir = runs_dir / f"iteration-{prev:03d}"
                    if prev_dir.exists():
                        try:
                            await asyncio.to_thread(shutil.rmtree, prev_dir)
                        except OSError:
                            pass  # Best-effort cleanup

            # 2. Select prompt (three-phase system)
            prompt = self._select_prompt(iteration)

            # 3. Run Claude Code via SDK (with rate-limit retry)
            await self._notify_status(f"Running Claude (iteration {iteration})")
            await self._notify_text(f"\n{'='*60}\n ITERATION {iteration}\n{'='*60}\n")

            claude_response, cost, duration_ms, num_turns, last_error_type = (
                await self._run_claude(iter_dir, prompt)
            )

            self.state.total_cost_usd += cost

            # 4. Analyze or skip
            analysis: AnalysisResult | None = None
            skipped = False

            if iteration < self.config.min_iterations:
                skipped = True
                await self._notify_status(
                    f"Iteration {iteration} < min ({self.config.min_iterations}), skipping analysis"
                )
            elif not self._stop_event.is_set():
                await self._notify_status(f"Analyzing output (iteration {iteration})")
                analysis = await analyze_output(
                    claude_response,
                    self.config.analysis_prompt,
                    self.config.exit_condition_prompt,
                )

            # 5. Record result
            result = IterationResult(
                iteration=iteration,
                claude_response=claude_response,
                cost_usd=cost,
                duration_ms=duration_ms,
                num_turns=num_turns,
                analysis=analysis,
                skipped_analysis=skipped,
            )
            self.state.results.append(result)
            self._log_iteration(log_file, result, error_type=last_error_type, usage=self._last_usage_info)
            if self._on_iteration_done:
                await self._on_iteration_done(result)

            # 6. Decide whether to continue
            if analysis and analysis.should_stop:
                await self._notify_status(
                    f"Completed after {iteration} iterations: {analysis.summary}"
                )
                break
        else:
            await self._notify_status(f"Reached max iterations ({self.config.max_iterations})")

        return self.state

    async def _run_claude(
        self, cwd: Path, prompt: str
    ) -> tuple[str, float, int, int, str | None]:
        """Run Claude Code with error retry and heartbeat monitoring.

        Handles rate limits (wait + resume), retryable errors (wait + fresh session),
        and non-retryable errors (return error text immediately).

        Returns (response_text, cost, duration_ms, num_turns, last_error_type).
        """
        # Build a clean env that strips CLAUDECODE to allow nested sessions
        clean_env = {
            "CLAUDE_AUTOCOMPACT_PCT_OVERRIDE": str(self.config.autocompact_pct),
        }
        if os.environ.get("CLAUDECODE"):
            clean_env["CLAUDECODE"] = ""

        def _handle_stderr(line: str) -> None:
            if self._current_log_file:
                entry = {
                    "timestamp": datetime.now().isoformat(),
                    "type": "stderr",
                    "iteration": self.state.current_iteration,
                    "message": line.strip(),
                }
                with open(self._current_log_file, "a") as f:
                    f.write(json.dumps(entry) + "\n")

        def _fresh_options(resume_session: str | None = None) -> ClaudeAgentOptions:
            return ClaudeAgentOptions(
                model="claude-opus-4-6",
                cwd=str(cwd),
                setting_sources=["project"],
                permission_mode="bypassPermissions",
                env=clean_env,
                resume=resume_session,
                include_partial_messages=True,
                stderr=_handle_stderr,
            )

        options = _fresh_options()

        total_response_text = ""
        total_cost = 0.0
        total_duration_ms = 0
        total_num_turns = 0
        error_retries = 0
        rate_limit_retries = 0
        last_error_type: str | None = None

        while True:
            response_text, cost, duration_ms, num_turns, captured_session_id, error_info = (
                await self._stream_claude(cwd, prompt, options)
            )

            total_response_text += response_text
            total_cost += cost
            total_duration_ms += duration_ms
            total_num_turns += num_turns

            # No error — done
            if error_info is None:
                last_error_type = None
                break

            last_error_type = error_info.type.value

            # Non-retryable error — return immediately
            if not error_info.retryable:
                await self._notify_text(
                    f"\n[Non-retryable error ({error_info.type.value}): {error_info.raw_message[:200]}]\n"
                )
                break

            # Rate limit — special handling: wait + resume same session
            if error_info.type == ErrorType.RATE_LIMIT:
                rate_limit_retries += 1
                if rate_limit_retries >= self.config.max_rate_limit_retries:
                    await self._notify_text(
                        f"\nRate limited {self.config.max_rate_limit_retries} times — giving up.\n"
                    )
                    break

                if error_info.retry_at:
                    wait_seconds = int(max(0, (error_info.retry_at - datetime.now()).total_seconds()))
                else:
                    wait_seconds = ERROR_RETRY_WAIT_SEC
                await self._notify_text(
                    f"\n[Rate limited — waiting {wait_seconds}s before retry]\n"
                )
                for remaining in range(wait_seconds, 0, -1):
                    await self._notify_status(
                        f"Rate limited — retry in {remaining}s "
                        f"(attempt {rate_limit_retries}/{self.config.max_rate_limit_retries})"
                    )
                    await asyncio.sleep(1)
                    if self._stop_event.is_set():
                        break

                # Resume the same session for rate limits
                resume_id = error_info.session_id or captured_session_id
                options = _fresh_options(resume_session=resume_id)
                continue

            # Retryable error (server, connection, parse, context, unknown, process)
            error_retries += 1
            if error_retries >= self.config.max_error_retries:
                await self._notify_text(
                    f"\n[Error retry limit reached ({self.config.max_error_retries}) — moving on]\n"
                )
                break

            await self._notify_text(
                f"\n[{error_info.type.value}: {error_info.raw_message[:200]}]\n"
            )
            for remaining in range(ERROR_RETRY_WAIT_SEC, 0, -1):
                await self._notify_status(
                    f"Retry in {remaining}s — {error_info.type.value}: "
                    f"{error_info.raw_message[:80]} "
                    f"(attempt {error_retries}/{self.config.max_error_retries})"
                )
                await asyncio.sleep(1)
                if self._stop_event.is_set():
                    break

            # Resume-first retry for unknown/process/connection errors
            if error_info.type in (ErrorType.UNKNOWN, ErrorType.PROCESS, ErrorType.CONNECTION):
                if error_retries == 1:  # First retry: try resume
                    resume_id = error_info.session_id or captured_session_id
                    if resume_id:
                        options = _fresh_options(resume_session=resume_id)
                        continue
            # Fresh session for subsequent retries or other error types
            options = _fresh_options()

        return total_response_text, total_cost, total_duration_ms, total_num_turns, last_error_type

    async def _stream_claude(
        self, cwd: Path, prompt: str, options: ClaudeAgentOptions
    ) -> tuple[str, float, int, int, str | None, ErrorInfo | None]:
        """Stream a single Claude query, returning results and error info.

        Returns (response_text, cost, duration_ms, num_turns, session_id, error_info).
        error_info is None if no error was detected.

        The watchdog runs as a separate asyncio.Task so it can detect stalls
        even when the stream is blocked and yields no messages. On hard timeout,
        the stream task is cancelled directly.
        """
        import contextlib

        ralph_pid = os.getpid()
        watchdog = HeartbeatWatchdog(soft_timeout_sec=SOFT_TIMEOUT_SEC, hard_timeout_sec=HARD_TIMEOUT_SEC)

        response_text = ""
        cost = 0.0
        duration_ms = 0
        num_turns = 0
        session_id: str | None = None
        collected_messages: list = []
        result_message = None
        current_tool_name: str | None = None
        stall_cancelled = False
        stream_exception: Exception | None = None
        latest_input_tokens = 0        # Updated on each message_start (last = current context)
        cumulative_output_tokens = 0   # Accumulated across all turns

        async def _consume_stream():
            nonlocal response_text, cost, duration_ms, num_turns, session_id
            nonlocal result_message, current_tool_name
            nonlocal latest_input_tokens, cumulative_output_tokens

            async for message in query(prompt=prompt, options=options):
                if self._stop_event.is_set():
                    break

                watchdog.ping()

                if isinstance(message, StreamEvent):
                    session_id = session_id or message.session_id
                    event = message.event
                    event_type = event.get("type", "")

                    if event_type == "content_block_delta":
                        delta = event.get("delta", {})
                        if delta.get("type") == "text_delta":
                            text = delta.get("text", "")
                            await self._notify_text(text)
                            await self._notify_activity(ActivityEvent(
                                timestamp=time.monotonic(),
                                event_type="text_delta",
                                text_fragment=text,
                            ))

                    elif event_type == "content_block_start":
                        cb = event.get("content_block", {})
                        if cb.get("type") == "tool_use":
                            tool_name = cb.get("name", "unknown")
                            current_tool_name = tool_name
                            ts = datetime.now().strftime("%H:%M:%S")
                            await self._notify_text(f"\n[{ts} Tool: {tool_name}]\n")
                            await self._notify_activity(ActivityEvent(
                                timestamp=time.monotonic(),
                                event_type="tool_start",
                                tool_name=tool_name,
                            ))

                    elif event_type == "content_block_stop":
                        await self._notify_activity(ActivityEvent(
                            timestamp=time.monotonic(),
                            event_type="tool_end",
                            tool_name=current_tool_name,
                        ))
                        current_tool_name = None

                    elif event_type == "message_start":
                        msg_usage = event.get("message", {}).get("usage", {})
                        total_input = _total_context_tokens(msg_usage)
                        if total_input > 0:
                            latest_input_tokens = total_input
                            pct = (latest_input_tokens / MODEL_CONTEXT_WINDOW) * 100
                            # Diagnostic usage logging
                            if self._current_log_file and msg_usage:
                                entry = {
                                    "timestamp": datetime.now().isoformat(),
                                    "type": "usage_event",
                                    "iteration": self.state.current_iteration,
                                    "event_type": "message_start",
                                    "input_tokens": msg_usage.get("input_tokens", 0),
                                    "cache_creation_input_tokens": msg_usage.get("cache_creation_input_tokens", 0),
                                    "cache_read_input_tokens": msg_usage.get("cache_read_input_tokens", 0),
                                    "total_context_tokens": total_input,
                                    "context_percent": pct,
                                }
                                with open(self._current_log_file, "a") as f:
                                    f.write(json.dumps(entry) + "\n")
                        await self._notify_activity(ActivityEvent(
                            timestamp=time.monotonic(),
                            event_type="message_start",
                        ))

                    elif event_type == "message_delta":
                        delta_usage = event.get("usage", {})
                        turn_output = delta_usage.get("output_tokens", 0)
                        if turn_output > 0:
                            cumulative_output_tokens += turn_output
                        delta_input = _total_context_tokens(delta_usage)
                        if delta_input > latest_input_tokens:
                            latest_input_tokens = delta_input
                        # Fire real-time usage update
                        if latest_input_tokens > 0:
                            pct = (latest_input_tokens / MODEL_CONTEXT_WINDOW) * 100
                            await self._notify_usage(UsageInfo(
                                input_tokens=latest_input_tokens,
                                output_tokens=cumulative_output_tokens,
                                total_tokens=latest_input_tokens + cumulative_output_tokens,
                                context_percent=pct,
                            ))
                        await self._notify_activity(ActivityEvent(
                            timestamp=time.monotonic(),
                            event_type="message_delta",
                        ))

                    elif event_type == "message_stop":
                        await self._notify_activity(ActivityEvent(
                            timestamp=time.monotonic(),
                            event_type="message_stop",
                        ))

                elif isinstance(message, AssistantMessage):
                    collected_messages.append(message)
                    for block in message.content:
                        if isinstance(block, ToolUseBlock):
                            if not current_tool_name:
                                await self._notify_text(f"\n[Tool: {block.name}]\n")

                elif isinstance(message, ResultMessage):
                    result_message = message
                    response_text = message.result or ""
                    cost = message.total_cost_usd or 0.0
                    duration_ms = message.duration_ms
                    num_turns = message.num_turns
                    session_id = session_id or getattr(message, "session_id", None)

                    # Only use ResultMessage.usage as fallback when stream events
                    # didn't provide usage data (e.g. old CLI version)
                    if latest_input_tokens == 0 and isinstance(getattr(message, 'usage', None), dict):
                        input_tok = _total_context_tokens(message.usage)
                        output_tok = message.usage.get("output_tokens", 0)
                        cache_create = message.usage.get("cache_creation_input_tokens", 0)
                        cache_read = message.usage.get("cache_read_input_tokens", 0)
                        total = input_tok + output_tok
                        pct = (input_tok / MODEL_CONTEXT_WINDOW) * 100
                        await self._notify_usage(UsageInfo(
                            input_tokens=input_tok,
                            output_tokens=output_tok,
                            cache_creation_input_tokens=cache_create,
                            cache_read_input_tokens=cache_read,
                            total_tokens=total,
                            context_percent=pct,
                        ))

        stream_task = asyncio.create_task(_consume_stream())

        async def _watchdog_loop():
            """Runs concurrently, cancels stream_task on hard timeout."""
            nonlocal stall_cancelled
            while not stream_task.done():
                await asyncio.sleep(WATCHDOG_CHECK_INTERVAL_SEC)
                if stream_task.done():
                    break
                if watchdog.is_hard_stale():
                    await self._notify_status(f"No output for {int(watchdog.elapsed())}s - cancelling stalled stream")
                    stall_cancelled = True
                    stream_task.cancel()
                    return
                if watchdog.is_soft_stale():
                    await self._notify_status(f"No activity for {int(watchdog.elapsed())}s - Claude may be stalled")
                    await self._notify_activity(ActivityEvent(
                        timestamp=time.monotonic(),
                        event_type="stall_warning",
                    ))

        watchdog_task = asyncio.create_task(_watchdog_loop())

        try:
            await stream_task
        except asyncio.CancelledError:
            pass  # Expected when watchdog cancels the stream
        except (ProcessError, CLIConnectionError, CLIJSONDecodeError, MessageParseError) as e:
            stream_exception = e
        except Exception as e:
            stream_exception = e
        finally:
            watchdog_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await watchdog_task

        if stream_exception is not None:
            error_msg = f"Claude SDK error: {type(stream_exception).__name__}: {stream_exception}"
            await self._notify_text(f"\n{error_msg}\n")
            response_text = error_msg
            error_info = detect_error(collected_messages, result_message, exception=stream_exception)
            return response_text, cost, duration_ms, num_turns, session_id, error_info

        if stall_cancelled:
            await self._cleanup_child_processes(ralph_pid)
            stall_error = ErrorInfo(
                type=ErrorType.UNKNOWN,
                retryable=True,
                session_id=session_id or "",
                raw_message=f"Stream stalled for {int(watchdog.elapsed())}s",
            )
            return response_text, cost, duration_ms, num_turns, session_id, stall_error

        # Check for errors in messages/result (rate limits, context exhaustion, etc.)
        error_info = detect_error(collected_messages, result_message)
        return response_text, cost, duration_ms, num_turns, session_id, error_info
