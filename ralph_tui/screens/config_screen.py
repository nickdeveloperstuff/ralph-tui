"""Configuration form screen for Ralph TUI."""

from __future__ import annotations

from pathlib import Path

from textual import on
from textual.app import ComposeResult
from textual.containers import VerticalScroll, Horizontal
from textual.screen import Screen
from textual.widgets import (
    Header,
    Footer,
    Input,
    Label,
    Button,
    TextArea,
    Static,
)

from ralph_tui.config import (
    RalphConfig,
    DEFAULT_ANALYSIS_PROMPT,
    DEFAULT_EXIT_CONDITION_PROMPT,
)


class ConfigScreen(Screen):
    CSS = """
    ConfigScreen {
        background: $surface;
    }
    #config-scroll {
        height: 1fr;
        margin: 1 2;
    }
    .field-label {
        margin-top: 1;
        color: $text;
        text-style: bold;
    }
    .field-desc {
        color: $text-muted;
        margin-bottom: 0;
    }
    TextArea {
        height: 6;
        margin-bottom: 1;
    }
    #ta-initial-prompt, #ta-rerun-prompt, #ta-final-prompt {
        height: 8;
    }
    #ta-analysis-prompt, #ta-exit-condition {
        height: 6;
    }
    Input {
        margin-bottom: 1;
    }
    Select {
        margin-bottom: 1;
    }
    #button-bar {
        height: 3;
        margin: 1 2;
        align: center middle;
    }
    #button-bar Button {
        margin: 0 1;
    }
    #error-display {
        color: $error;
        margin: 0 2;
        height: auto;
    }
    """

    BINDINGS = [
        ("ctrl+s", "save_config", "Save Config"),
        ("ctrl+l", "load_config", "Load Config"),
        ("ctrl+r", "start", "Start"),
    ]

    def compose(self) -> ComposeResult:
        yield Header()
        yield Static("", id="error-display")
        with VerticalScroll(id="config-scroll"):
            yield Label("Project Path", classes="field-label")
            yield Label("Defaults to current directory", classes="field-desc")
            yield Input(placeholder="/path/to/project", id="in-project-path")

            yield Label("Initial Prompt (iteration 1)", classes="field-label")
            yield Label("What to tell Claude on the first run", classes="field-desc")
            yield TextArea(id="ta-initial-prompt")

            yield Label("Re-run Prompt (iterations 2+)", classes="field-label")
            yield Label("Static prompt sent on every subsequent iteration", classes="field-desc")
            yield TextArea(id="ta-rerun-prompt")

            yield Label("Final Prompt (optional, iterations N+)", classes="field-label")
            yield Label(
                "Third-phase prompt used from transition iteration onward (leave empty to disable)",
                classes="field-desc",
            )
            yield TextArea(id="ta-final-prompt")

            yield Label("Transition Iteration (0 = disabled)", classes="field-label")
            yield Label(
                "Iteration number where final prompt takes over from re-run prompt",
                classes="field-desc",
            )
            yield Input("0", id="in-transition-iter", type="integer")

            yield Label("Analysis Prompt (for Gemini)", classes="field-label")
            yield Label(
                "System prompt telling Gemini how to evaluate Claude's output",
                classes="field-desc",
            )
            yield TextArea(DEFAULT_ANALYSIS_PROMPT, id="ta-analysis-prompt")

            yield Label("Exit Condition Prompt (for Gemini)", classes="field-label")
            yield Label(
                "Asks Gemini for a JSON verdict on whether to stop",
                classes="field-desc",
            )
            yield TextArea(DEFAULT_EXIT_CONDITION_PROMPT, id="ta-exit-condition")

            yield Label("Min Iterations", classes="field-label")
            yield Label("Force at least N runs regardless of analysis", classes="field-desc")
            yield Input("2", id="in-min-iter", type="integer")

            yield Label("Max Iterations", classes="field-label")
            yield Label("Safety cap on total iterations", classes="field-desc")
            yield Input("10", id="in-max-iter", type="integer")

            yield Label("Max Error Retries", classes="field-label")
            yield Label("How many times to retry on errors per iteration", classes="field-desc")
            yield Input("5", id="in-max-error-retries", type="integer")

            yield Label("Max Rate Limit Retries", classes="field-label")
            yield Label("How many times to retry on rate limits per iteration", classes="field-desc")
            yield Input("10", id="in-max-rl-retries", type="integer")

            yield Label("Verification Prompt", classes="field-label")
            yield Label(
                "What to verify (e.g., 'legal citations and factual claims'). Leave empty to disable.",
                classes="field-desc",
            )
            yield TextArea(id="ta-verification-prompt")

            yield Label("Verification Interval", classes="field-label")
            yield Label(
                "Every Nth iteration runs verification instead of drafting (0 = disabled)",
                classes="field-desc",
            )
            yield Input("0", id="in-verification-interval", type="integer")


        with Horizontal(id="button-bar"):
            yield Button("Save Config", variant="default", id="btn-save")
            yield Button("Load Config", variant="default", id="btn-load")
            yield Button("Start", variant="success", id="btn-start")

        yield Footer()

    def on_mount(self) -> None:
        cwd = getattr(self.app, "launch_cwd", "")
        if cwd:
            self.query_one("#in-project-path", Input).value = cwd

    def _build_config(self) -> RalphConfig:
        """Build a RalphConfig from current form values."""
        return RalphConfig(
            project_path=self.query_one("#in-project-path", Input).value,
            initial_prompt=self.query_one("#ta-initial-prompt", TextArea).text,
            rerun_prompt=self.query_one("#ta-rerun-prompt", TextArea).text,
            final_prompt=self.query_one("#ta-final-prompt", TextArea).text,
            analysis_prompt=self.query_one("#ta-analysis-prompt", TextArea).text,
            exit_condition_prompt=self.query_one("#ta-exit-condition", TextArea).text,
            min_iterations=int(self.query_one("#in-min-iter", Input).value or "2"),
            max_iterations=int(self.query_one("#in-max-iter", Input).value or "10"),
            transition_iteration=int(self.query_one("#in-transition-iter", Input).value or "0"),
            max_error_retries=int(self.query_one("#in-max-error-retries", Input).value or "5"),
            max_rate_limit_retries=int(self.query_one("#in-max-rl-retries", Input).value or "10"),
            verification_prompt=self.query_one("#ta-verification-prompt", TextArea).text,
            verification_interval=int(self.query_one("#in-verification-interval", Input).value or "0"),
        )

    def _load_config_to_form(self, config: RalphConfig) -> None:
        """Populate form fields from a RalphConfig."""
        self.query_one("#in-project-path", Input).value = config.project_path
        self.query_one("#ta-initial-prompt", TextArea).text = config.initial_prompt
        self.query_one("#ta-rerun-prompt", TextArea).text = config.rerun_prompt
        self.query_one("#ta-final-prompt", TextArea).text = config.final_prompt
        self.query_one("#ta-analysis-prompt", TextArea).text = config.analysis_prompt
        self.query_one("#ta-exit-condition", TextArea).text = config.exit_condition_prompt
        self.query_one("#in-min-iter", Input).value = str(config.min_iterations)
        self.query_one("#in-max-iter", Input).value = str(config.max_iterations)
        self.query_one("#in-transition-iter", Input).value = str(config.transition_iteration)
        self.query_one("#in-max-error-retries", Input).value = str(config.max_error_retries)
        self.query_one("#in-max-rl-retries", Input).value = str(config.max_rate_limit_retries)
        self.query_one("#ta-verification-prompt", TextArea).text = config.verification_prompt
        self.query_one("#in-verification-interval", Input).value = str(config.verification_interval)

    def _show_error(self, msg: str) -> None:
        self.query_one("#error-display", Static).update(msg)

    def _clear_error(self) -> None:
        self.query_one("#error-display", Static).update("")

    @on(Button.Pressed, "#btn-start")
    def action_start(self) -> None:
        from ralph_tui.screens.runner_screen import RunnerScreen

        self._clear_error()
        config = self._build_config()
        errors = config.validate()
        if errors:
            self._show_error("\n".join(errors))
            return
        self.app.push_screen(RunnerScreen(config))

    @on(Button.Pressed, "#btn-save")
    def action_save_config(self) -> None:
        self._clear_error()
        config = self._build_config()
        save_path = Path.cwd() / "configs" / "last.yaml"
        try:
            config.save_yaml(save_path)
            self._show_error(f"Saved to {save_path}")
        except Exception as e:
            self._show_error(f"Save failed: {e}")

    @on(Button.Pressed, "#btn-load")
    def action_load_config(self) -> None:
        self._clear_error()
        load_path = Path.cwd() / "configs" / "last.yaml"
        if not load_path.exists():
            self._show_error(f"No config found at {load_path}")
            return
        try:
            config = RalphConfig.load_yaml(load_path)
            self._load_config_to_form(config)
            self._show_error(f"Loaded from {load_path}")
        except Exception as e:
            self._show_error(f"Load failed: {e}")
