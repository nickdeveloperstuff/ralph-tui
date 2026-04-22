"""Live execution monitoring screen for Ralph TUI."""

from __future__ import annotations

import time
from dataclasses import dataclass

from textual import work, on
from textual.app import ComposeResult
from textual.containers import Vertical
from textual.message import Message
from textual.screen import Screen
from textual.widgets import (
    Header,
    Footer,
    Static,
    RichLog,
    ListView,
    ListItem,
    Label,
    Button,
)

from ralph_tui.config import RalphConfig
from ralph_tui.orchestrator import Orchestrator, IterationResult, OrchestratorState, ActivityEvent, UsageInfo


class StatusUpdate(Message):
    def __init__(self, status: str) -> None:
        super().__init__()
        self.status = status


class TextChunk(Message):
    def __init__(self, text: str) -> None:
        super().__init__()
        self.text = text


class IterationDone(Message):
    def __init__(self, result: IterationResult) -> None:
        super().__init__()
        self.result = result


class RunComplete(Message):
    def __init__(self, state: OrchestratorState) -> None:
        super().__init__()
        self.state = state


class ActivityUpdate(Message):
    def __init__(self, event: ActivityEvent) -> None:
        super().__init__()
        self.event = event


class UsageUpdate(Message):
    def __init__(self, usage: UsageInfo) -> None:
        super().__init__()
        self.usage = usage


class RunnerScreen(Screen):
    CSS = """
    RunnerScreen {
        background: $surface;
    }
    #status-bar {
        height: 3;
        margin: 0 1;
        padding: 0 1;
        background: $primary-background;
        color: $text;
    }
    #output-log {
        height: 1fr;
        margin: 0 1;
        border: solid $primary;
    }
    #analysis-list {
        height: auto;
        max-height: 12;
        margin: 0 1;
        border: solid $secondary;
    }
    #analysis-header {
        margin: 1 1 0 1;
        text-style: bold;
    }
    #footer-bar {
        height: 3;
        margin: 0 1;
        align: center middle;
    }
    """

    BINDINGS = [
        ("escape", "go_back", "Back"),
    ]

    def __init__(self, config: RalphConfig) -> None:
        super().__init__()
        self.config = config
        self.orchestrator: Orchestrator | None = None
        self._start_time = 0.0
        self._total_cost = 0.0
        self._last_activity_time: float = 0.0
        self._last_tool_time: float = 0.0
        self._current_tool: str | None = None
        self._last_usage: UsageInfo | None = None
        self._stall_warning_shown: bool = False
        self._last_status: str = "Initializing"

    def compose(self) -> ComposeResult:
        yield Header()
        yield Static(
            "Iteration: 0/0  Status: Initializing  Cost: $0.0000  Elapsed: 0s",
            id="status-bar",
        )
        yield RichLog(highlight=True, wrap=True, id="output-log")
        yield Label("Analysis History:", id="analysis-header")
        yield ListView(id="analysis-list")
        with Vertical(id="footer-bar"):
            yield Button("Stop", variant="error", id="btn-stop")
        yield Footer()

    def on_mount(self) -> None:
        self._start_time = time.time()
        self._last_activity_time = time.monotonic()
        self.set_interval(1.0, self._tick_activity)
        self._run_orchestrator()

    @work(thread=False)
    async def _run_orchestrator(self) -> None:
        """Run the orchestrator in a Textual worker."""
        self.orchestrator = Orchestrator(
            config=self.config,
            on_status=self._handle_status,
            on_text=self._handle_text,
            on_iteration_done=self._handle_iteration_done,
            on_activity=self._handle_activity,
            on_usage=self._handle_usage,
        )
        state = await self.orchestrator.run()
        self.post_message(RunComplete(state))

    async def _handle_status(self, status: str) -> None:
        self.post_message(StatusUpdate(status))

    async def _handle_text(self, text: str) -> None:
        self.post_message(TextChunk(text))

    async def _handle_iteration_done(self, result: IterationResult) -> None:
        self.post_message(IterationDone(result))

    async def _handle_activity(self, event: ActivityEvent) -> None:
        self.post_message(ActivityUpdate(event))

    async def _handle_usage(self, usage: UsageInfo) -> None:
        self.post_message(UsageUpdate(usage))

    def _tick_activity(self) -> None:
        """Called every second to update activity indicators in the status bar."""
        self._update_status_bar(self._last_status)

    def _update_status_bar(self, status: str) -> None:
        elapsed = int(time.time() - self._start_time)
        minutes, seconds = divmod(elapsed, 60)
        iteration = self.orchestrator.state.current_iteration if self.orchestrator else 0
        self._total_cost = self.orchestrator.state.total_cost_usd if self.orchestrator else 0

        # Build activity info parts — wall clock first
        parts = [
            f"[{time.strftime('%H:%M:%S')}]",
            f"Iteration: {iteration}/{self.config.max_iterations}",
            f"(min: {self.config.min_iterations})",
            f"Status: {status}",
            f"Cost: ${self._total_cost:.4f}",
            f"Elapsed: {minutes}m {seconds:02d}s",
        ]

        # Activity age
        if self._last_activity_time > 0:
            age = int(time.monotonic() - self._last_activity_time)
            if age > 0:
                parts.append(f"Last activity: {age}s ago")

        # Current tool
        if self._current_tool:
            parts.append(f"Tool: {self._current_tool}")

        # Context usage with color thresholds
        if self._last_usage:
            pct = self._last_usage.context_percent
            if pct >= 60:
                parts.append(f"[bold red]Context: ~{pct:.0f}%[/]")
            elif pct >= 40:
                parts.append(f"[yellow]Context: ~{pct:.0f}%[/]")
            else:
                parts.append(f"Context: ~{pct:.0f}%")

        # Stall vs writing indicator — only if status doesn't already say stalled/errored
        status_lower = status.lower()
        is_error_or_stall = any(kw in status_lower for kw in ("error", "stall", "retry", "no activity"))

        if not is_error_or_stall and self._last_activity_time > 0:
            stream_age = time.monotonic() - self._last_activity_time
            tool_age = (time.monotonic() - self._last_tool_time) if self._last_tool_time > 0 else 0
            if stream_age > 60:
                # No events at all — true stall
                parts.append("[bold red]STALL WARNING[/]")
            elif tool_age > 60 and self._last_tool_time > 0:
                # Text is flowing but no tools for a while
                parts.append(f"[yellow]Writing... {int(tool_age)}s[/]")

        self.query_one("#status-bar", Static).update("  ".join(parts))

    @on(StatusUpdate)
    def _on_status(self, event: StatusUpdate) -> None:
        self._last_status = event.status
        self._update_status_bar(event.status)

    @on(TextChunk)
    def _on_text(self, event: TextChunk) -> None:
        log = self.query_one("#output-log", RichLog)
        # expand=True overrides RichLog.min_width (default 78) so writes fill
        # the widget's inner width instead of wrapping at column ~80.
        log.write(event.text, expand=True)

    @on(ActivityUpdate)
    def _on_activity(self, event: ActivityUpdate) -> None:
        # Do NOT reset activity time on stall_warning — that's a meta-event, not real activity
        if event.event.event_type != "stall_warning":
            self._last_activity_time = event.event.timestamp

        if event.event.event_type in ("tool_start", "tool_end"):
            self._last_tool_time = event.event.timestamp
        if event.event.event_type == "tool_start":
            self._current_tool = event.event.tool_name
        elif event.event.event_type in ("tool_end", "message_stop"):
            self._current_tool = None

        # Clear stale error/stall status when real activity resumes
        if event.event.event_type in ("text_delta", "tool_start", "message_start"):
            if any(kw in self._last_status.lower() for kw in ("error", "stall", "retry", "no activity")):
                self._last_status = "Running"

        # Log tool_end with timestamp
        if event.event.event_type == "tool_end":
            ts = time.strftime("%H:%M:%S")
            tool = event.event.tool_name or "unknown"
            log = self.query_one("#output-log", RichLog)
            log.write(f"[dim][{ts} Done: {tool}][/dim]", expand=True)

    @on(UsageUpdate)
    def _on_usage(self, event: UsageUpdate) -> None:
        self._last_usage = event.usage

    @on(IterationDone)
    def _on_iteration_done(self, event: IterationDone) -> None:
        r = event.result
        lv = self.query_one("#analysis-list", ListView)
        if r.skipped_analysis:
            text = f"  #{r.iteration}: (below min, skipped analysis) — ${r.cost_usd:.4f}"
        elif r.analysis:
            icon = "[green]✓[/]" if r.analysis.should_stop else "[red]✗[/]"
            text = f"  #{r.iteration}: {icon} \"{r.analysis.summary}\" — ${r.cost_usd:.4f}"
        else:
            text = f"  #{r.iteration}: — (no analysis) — ${r.cost_usd:.4f}"
        lv.append(ListItem(Label(text, markup=True)))

    @on(RunComplete)
    def _on_run_complete(self, event: RunComplete) -> None:
        log = self.query_one("#output-log", RichLog)
        log.write(f"\n{'='*60}")
        log.write(f"Run complete. Total cost: ${event.state.total_cost_usd:.4f}")
        log.write(f"Status: {event.state.status}")
        log.write(f"Iterations completed: {len(event.state.results)}")
        self.query_one("#btn-stop", Button).label = "Back"

    @on(Button.Pressed, "#btn-stop")
    def _on_stop(self) -> None:
        if self.orchestrator and self.orchestrator.state.status not in (
            "idle",
        ) and "Completed" not in self.orchestrator.state.status and "Reached max" not in self.orchestrator.state.status and "Stopped" not in self.orchestrator.state.status:
            self.orchestrator.stop()
        else:
            self.app.pop_screen()

    def action_go_back(self) -> None:
        if self.orchestrator:
            self.orchestrator.stop()
        self.app.pop_screen()
