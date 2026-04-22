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
from ralph_tui.config import (
    RalphConfig, CONTEXT_MANAGEMENT_SUFFIX, CLAUDE_MD_TEMPLATE,
    CLAUDE_MD_VERIFICATION_TEMPLATE, CONTEXT_RECOVERY_SUFFIX,
    VERIFICATION_METHODOLOGY_TEMPLATE,
)
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
    effective_iteration: int = 0   # 0 if ineffective, else the Nth effective iteration
    is_effective: bool = True      # True if real work was done


@dataclass
class OrchestratorState:
    status: str = "idle"
    current_iteration: int = 0
    total_cost_usd: float = 0.0
    start_time: float = 0.0
    results: list[IterationResult] = field(default_factory=list)
    effective_iterations: int = 0       # count of productive iterations
    consecutive_errors: int = 0         # for escalating backoff
    consecutive_stops: int = 0          # for "2 consecutive stops" end criteria


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
    "_verification_manifest.json", "_verification_report.md",
    "._ralph_hidden",
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
        self._pending_verification_feedback: str | None = None

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

    def _is_effective(self, cost_usd: float, last_error_type: str | None) -> bool:
        """An iteration is effective if Claude did real work (cost > 0, no error)."""
        return cost_usd > 0.0 and last_error_type is None

    async def _kill_zombie_claude_processes(self) -> None:
        """Kill any lingering claude CLI processes that may block new sessions."""
        try:
            result = await asyncio.to_thread(
                _subprocess.run,
                ["pkill", "-f", "claude"],
                capture_output=True, text=True, timeout=10,
            )
            if result.returncode == 0:
                await self._notify_text("\n[Killed zombie claude processes]\n")
        except (FileNotFoundError, _subprocess.TimeoutExpired):
            pass

    def _get_backoff_wait(self, consecutive_errors: int) -> int:
        """Return wait time in seconds based on consecutive error count."""
        if consecutive_errors <= 2:
            return 60
        elif consecutive_errors == 3:
            return 300  # 5 minutes
        elif consecutive_errors == 4:
            return 600  # 10 minutes
        else:
            return 900  # 15 minutes

    def _prepare_verification_dir(self, iter_dir: Path) -> None:
        """Move main document and scratch files into ._ralph_hidden/ for blind verification."""
        hidden = iter_dir / "._ralph_hidden"
        hidden.mkdir(exist_ok=True)

        # Move all root-level non-Ralph, non-directory files (the main document)
        for item in iter_dir.iterdir():
            if item.is_dir():
                continue
            name = item.name
            # Skip Ralph internal files and scratch files
            if name in RALPH_INTERNAL_FILES:
                continue
            if name.startswith("_scratch_"):
                # Move scratch files to hidden
                shutil.move(str(item), str(hidden / name))
                continue
            if name.startswith(".") or name.startswith("_"):
                continue
            # This is a user file (main document) — move it
            shutil.move(str(item), str(hidden / name))

        # Extract claims from _ralph_state.json → write _claims.json
        state_file = iter_dir / "_ralph_state.json"
        if state_file.exists():
            try:
                state = json.loads(state_file.read_text())
                citations = state.get("citations_to_verify", [])

                # Write claims file (full citations with claims)
                claims = [
                    {"citation": c.get("citation", ""), "claim": c.get("claim", ""), "status": c.get("status", "")}
                    for c in citations if isinstance(c, dict)
                ]
                (hidden / "_claims.json").write_text(json.dumps(claims, indent=2))

                # Write verification manifest (citation refs ONLY, no claims)
                manifest = [
                    {"citation": c.get("citation", ""), "status": c.get("status", "")}
                    for c in citations if isinstance(c, dict)
                ]
                (iter_dir / "_verification_manifest.json").write_text(json.dumps(manifest, indent=2))

                # Redact _ralph_state.json: remove claim field from citations
                for c in citations:
                    if isinstance(c, dict):
                        c.pop("claim", None)
                state_file.write_text(json.dumps(state, indent=2))

            except (json.JSONDecodeError, OSError):
                pass

    def _restore_from_hidden(self, iter_dir: Path) -> None:
        """Restore files from ._ralph_hidden/ back to iteration dir root."""
        hidden = iter_dir / "._ralph_hidden"
        if not hidden.exists():
            return
        for item in hidden.iterdir():
            dest = iter_dir / item.name
            if dest.exists():
                if dest.is_dir():
                    shutil.rmtree(dest)
                else:
                    dest.unlink()
            shutil.move(str(item), str(dest))
        # Remove the now-empty hidden dir
        try:
            hidden.rmdir()
        except OSError:
            pass
        # Clean up verification manifest
        manifest = iter_dir / "_verification_manifest.json"
        if manifest.exists():
            manifest.unlink()

    def _build_verification_feedback(self, iter_dir: Path) -> str | None:
        """Read verification results and format feedback for next drafting iteration."""
        state_file = iter_dir / "_ralph_state.json"
        if not state_file.exists():
            return None
        try:
            state = json.loads(state_file.read_text())
            citations = state.get("citations_to_verify", [])
            issues = []
            for c in citations:
                if not isinstance(c, dict):
                    continue
                status = c.get("status", "")
                citation_ref = c.get("citation", "unknown")
                if status == "disputed":
                    disc = c.get("discrepancy", "no details")
                    issues.append(f"- [{citation_ref}] — DISPUTED: {disc}")
                elif status == "unable_to_verify":
                    reason = c.get("reason", "no details")
                    issues.append(f"- [{citation_ref}] — UNABLE TO VERIFY: {reason}")

            if not issues:
                return None

            return (
                "\n## VERIFICATION FINDINGS — ADDRESS THESE FIRST\n"
                + "\n".join(issues) + "\n"
            )
        except (json.JSONDecodeError, OSError):
            return None

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

    def _read_task_summary(self, iter_dir: Path) -> str | None:
        """Read _ralph_state.json and return a one-line task summary."""
        state_file = iter_dir / "_ralph_state.json"
        if not state_file.exists():
            return None
        try:
            state = json.loads(state_file.read_text())
            tasks = state.get("tasks", [])
            if not tasks:
                return None
            done = sum(1 for t in tasks if isinstance(t, dict) and t.get("status") == "completed")
            return f"{done}/{len(tasks)} tasks completed"
        except (json.JSONDecodeError, OSError):
            return None

    def _log_iteration(
        self, log_file: Path, result: IterationResult, error_type: str | None = None,
        usage: UsageInfo | None = None,
    ) -> None:
        """Append iteration result as a JSON line to the log file."""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "iteration": result.iteration,
            "effective_iteration": result.effective_iteration,
            "is_effective": result.is_effective,
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
        """Execute the full orchestration loop.

        Uses effective iteration counting: only iterations where Claude did real work
        (cost > 0, no errors) count toward min/max/transition/verification thresholds.
        Escalating backoff on consecutive errors. Two consecutive stops required to end.
        """
        self.state = OrchestratorState(start_time=time.time())
        project_path = Path(self.config.project_path).expanduser().resolve()
        # Place runs dir as a sibling to avoid recursive copy issues
        runs_dir = project_path.parent / f"{project_path.name}-ralph-runs"
        runs_dir.mkdir(exist_ok=True)
        log_file = runs_dir / "ralph-log.jsonl"
        self._current_log_file = log_file

        raw_iteration = 0

        while self.state.effective_iterations < self.config.max_iterations:
            if self._stop_event.is_set():
                await self._notify_status("Stopped by user")
                break

            raw_iteration += 1
            self.state.current_iteration = raw_iteration
            iter_dir = runs_dir / f"iteration-{raw_iteration:03d}"

            # Prospective effective iteration (if this one succeeds)
            prospective_effective = self.state.effective_iterations + 1

            # 1. Copy folder
            await self._notify_status(
                f"Copying project (raw {raw_iteration}, effective {self.state.effective_iterations})"
            )
            if raw_iteration == 1:
                src = project_path
            else:
                src = runs_dir / f"iteration-{raw_iteration - 1:03d}"

            await asyncio.to_thread(_copy_project, src, iter_dir)

            # Restore hidden files from verification iteration (if prev was verification)
            self._restore_from_hidden(iter_dir)

            # Determine if this will be a verification iteration (using effective count)
            is_verification = (
                self.config.verification_interval > 0
                and bool(self.config.verification_prompt)
                and prospective_effective > 1
                and prospective_effective % self.config.verification_interval == 0
            )

            if is_verification:
                # Write verification-specific CLAUDE.md
                (iter_dir / "CLAUDE.md").write_text(CLAUDE_MD_VERIFICATION_TEMPLATE)
                # Prepare blind verification directory structure
                self._prepare_verification_dir(iter_dir)
            else:
                # Always write standard template (overwrite any verification template
                # carried forward from previous iteration copy)
                (iter_dir / "CLAUDE.md").write_text(CLAUDE_MD_TEMPLATE)

            _inject_state_file(iter_dir)

            # Generate document index on first raw iteration only
            if raw_iteration == 1:
                await asyncio.to_thread(_generate_document_index, project_path, iter_dir)

            # Clean up prior iteration dirs (keep N-1 as fallback)
            if raw_iteration > 2:
                for prev in range(1, raw_iteration - 1):
                    prev_dir = runs_dir / f"iteration-{prev:03d}"
                    if prev_dir.exists():
                        try:
                            await asyncio.to_thread(shutil.rmtree, prev_dir)
                        except OSError:
                            pass  # Best-effort cleanup

            # 2. Select prompt (using effective iteration count)
            prompt = self._select_prompt(prospective_effective)

            # Prepend verification feedback if available
            if self._pending_verification_feedback and not is_verification:
                prompt = self._pending_verification_feedback + "\n" + prompt
                self._pending_verification_feedback = None

            # Append context recovery suffix if in recovery mode
            if self.state.consecutive_errors >= 3:
                prompt += CONTEXT_RECOVERY_SUFFIX

            # 3. Run Claude Code via SDK (with rate-limit retry)
            await self._notify_status(
                f"Running Claude (raw {raw_iteration}, effective {prospective_effective})"
            )
            await self._notify_text(
                f"\n{'='*60}\n ITERATION {raw_iteration} "
                f"(effective {prospective_effective})"
                f"{' [VERIFICATION]' if is_verification else ''}\n{'='*60}\n"
            )

            claude_response, cost, duration_ms, num_turns, last_error_type = (
                await self._run_claude(iter_dir, prompt)
            )

            self.state.total_cost_usd += cost

            # Determine effectiveness
            effective = self._is_effective(cost, last_error_type)

            if effective:
                self.state.effective_iterations += 1
                effective_num = self.state.effective_iterations
                self.state.consecutive_errors = 0

                # Collect verification feedback after a verification iteration
                if is_verification:
                    self._pending_verification_feedback = self._build_verification_feedback(iter_dir)
            else:
                effective_num = 0
                self.state.consecutive_errors += 1

                # Circuit breaker
                if self.state.consecutive_errors >= self.config.max_consecutive_errors:
                    await self._notify_status(
                        f"Circuit breaker: {self.state.consecutive_errors} consecutive errors"
                    )
                    await self._notify_text(
                        f"\n[CIRCUIT BREAKER: {self.state.consecutive_errors} consecutive "
                        f"errors — stopping]\n"
                    )
                    break

                # Escalating backoff
                wait_seconds = self._get_backoff_wait(self.state.consecutive_errors)
                await self._notify_text(
                    f"\n[Error #{self.state.consecutive_errors}: waiting {wait_seconds}s "
                    f"before retry]\n"
                )

                # Kill zombies on 3+ consecutive errors
                if self.state.consecutive_errors >= 3:
                    await self._kill_zombie_claude_processes()

                for remaining in range(wait_seconds, 0, -1):
                    await self._notify_status(
                        f"Backoff: retry in {remaining}s "
                        f"(error {self.state.consecutive_errors}/"
                        f"{self.config.max_consecutive_errors})"
                    )
                    await asyncio.sleep(1)
                    if self._stop_event.is_set():
                        break

            # 4. Analyze or skip (only for effective iterations)
            analysis: AnalysisResult | None = None
            skipped = False

            if not effective:
                skipped = True
            elif effective_num < self.config.min_iterations:
                skipped = True
                await self._notify_status(
                    f"Effective iteration {effective_num} < min ({self.config.min_iterations}), "
                    f"skipping analysis"
                )
            elif not self._stop_event.is_set():
                await self._notify_status(f"Analyzing output (effective {effective_num})")

                # Determine phase name
                if effective_num == 1:
                    phase = "initial"
                elif is_verification:
                    phase = "verification"
                elif (self.config.transition_iteration > 0
                      and effective_num >= self.config.transition_iteration):
                    phase = "final"
                else:
                    phase = "rerun"

                # Read task summary from state file if available
                task_summary = self._read_task_summary(iter_dir)

                iteration_context = {
                    "iteration": effective_num,
                    "raw_iteration": raw_iteration,
                    "max_iterations": self.config.max_iterations,
                    "is_verification": is_verification,
                    "phase": phase,
                    "remaining": self.config.max_iterations - effective_num,
                    "task_summary": task_summary,
                }

                analysis = await analyze_output(
                    claude_response,
                    self.config.analysis_prompt,
                    self.config.exit_condition_prompt,
                    iteration_context=iteration_context,
                )

            # 5. Record result
            result = IterationResult(
                iteration=raw_iteration,
                claude_response=claude_response,
                cost_usd=cost,
                duration_ms=duration_ms,
                num_turns=num_turns,
                analysis=analysis,
                skipped_analysis=skipped,
                effective_iteration=effective_num,
                is_effective=effective,
            )
            self.state.results.append(result)
            self._log_iteration(log_file, result, error_type=last_error_type, usage=self._last_usage_info)
            if self._on_iteration_done:
                await self._on_iteration_done(result)

            # 6. Decide whether to continue (2 consecutive stops required)
            if effective and not is_verification:
                if analysis and analysis.should_stop:
                    self.state.consecutive_stops += 1
                    if self.state.consecutive_stops >= 2:
                        await self._notify_status(
                            f"Completed after {effective_num} effective iterations "
                            f"(2 consecutive stops): {analysis.summary}"
                        )
                        break
                    else:
                        await self._notify_status(
                            f"Analyzer says stop (1/2 needed). Running one more to confirm."
                        )
                else:
                    self.state.consecutive_stops = 0
            elif effective and is_verification:
                # Verification iterations don't affect consecutive_stops
                if analysis and analysis.should_stop:
                    await self._notify_status(
                        f"Verification complete (effective {effective_num}), continuing workflow"
                    )
        else:
            await self._notify_status(
                f"Reached max effective iterations ({self.config.max_iterations})"
            )

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
