"""Main Textual app entry point for Ralph TUI."""

from __future__ import annotations

import os

from textual.app import App

from ralph_tui.screens.config_screen import ConfigScreen


class RalphApp(App):
    TITLE = "Ralph TUI"
    SUB_TITLE = "Meta-Orchestrator for Claude Code Sessions"

    CSS = """
    Screen {
        background: $surface;
    }
    """

    SCREENS = {"config": ConfigScreen}

    def __init__(self, launch_cwd: str = "", **kwargs) -> None:
        super().__init__(**kwargs)
        self.launch_cwd = launch_cwd

    def on_mount(self) -> None:
        self.push_screen("config")


def main() -> None:
    cwd = os.getcwd()
    app = RalphApp(launch_cwd=cwd)
    app.run()


if __name__ == "__main__":
    main()
