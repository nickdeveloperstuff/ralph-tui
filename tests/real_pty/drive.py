"""Tmux wrapper — my hands and eyes when I am being the end user.

Not a test framework. A small driver I run by hand (or by script) so I can
launch the real ralph TUI in a real PTY at a real terminal size, send real
keystrokes, and capture what a human would see. Captures are the evidence
that a feature works.

Stdlib only. tmux 3.6a at /opt/homebrew/bin/tmux.
"""

from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path
from typing import Callable


TMUX = "/opt/homebrew/bin/tmux"


class Tmux:
    def __init__(self, name: str):
        self.name = name

    def start(
        self,
        width: int,
        height: int,
        argv: list[str],
        env: dict[str, str] | None = None,
        cwd: str | None = None,
    ) -> None:
        """Spawn a detached tmux session at the exact given size.

        Uses -x/-y to fix size; -d to detach so tmux doesn't steal the
        terminal. argv is run as the session's first (and only) command.
        """
        self.kill()  # idempotent
        full_env = os.environ.copy()
        if env:
            full_env.update(env)
        # Wrap argv so we can see the exit status if it crashes.
        cmd = [
            TMUX, "new-session", "-d",
            "-s", self.name,
            "-x", str(width),
            "-y", str(height),
        ]
        if cwd:
            cmd += ["-c", cwd]
        cmd += argv
        r = subprocess.run(cmd, env=full_env, capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"tmux new-session failed: {r.stderr}")

    def send(self, *keys: str) -> None:
        """Send tmux send-keys tokens.

        For literal text (no special meaning), pass it as a single string;
        tmux treats anything it doesn't recognize as a key name as literal.
        For named keys, pass the tmux name: "PageUp", "End", "Enter", etc.
        """
        r = subprocess.run(
            [TMUX, "send-keys", "-t", self.name, *keys],
            capture_output=True, text=True,
        )
        if r.returncode != 0:
            raise RuntimeError(f"tmux send-keys failed: {r.stderr}")

    def pane(self) -> str:
        """Return the rendered pane as plain text (no ANSI)."""
        r = subprocess.run(
            [TMUX, "capture-pane", "-t", self.name, "-p"],
            capture_output=True, text=True,
        )
        if r.returncode != 0:
            raise RuntimeError(f"tmux capture-pane failed: {r.stderr}")
        return r.stdout

    def pane_raw(self) -> str:
        """Pane with ANSI escapes — useful when colour/markup matters."""
        r = subprocess.run(
            [TMUX, "capture-pane", "-t", self.name, "-p", "-e"],
            capture_output=True, text=True,
        )
        if r.returncode != 0:
            raise RuntimeError(f"tmux capture-pane failed: {r.stderr}")
        return r.stdout

    def wait_for(
        self,
        predicate: Callable[[str], bool],
        timeout: float,
        poll: float = 0.2,
    ) -> bool:
        """Poll pane() until predicate(text) is truthy or timeout elapses."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if predicate(self.pane()):
                return True
            time.sleep(poll)
        return False

    def wait_for_substring(self, needle: str, timeout: float) -> bool:
        return self.wait_for(lambda p: needle in p, timeout)

    def save(self, path: str | Path) -> str:
        """Write current pane to a file. Creates parent dirs."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        text = self.pane()
        path.write_text(text)
        return text

    def session_alive(self) -> bool:
        r = subprocess.run(
            [TMUX, "has-session", "-t", self.name],
            capture_output=True, text=True,
        )
        return r.returncode == 0

    def kill(self) -> None:
        subprocess.run(
            [TMUX, "kill-session", "-t", self.name],
            capture_output=True, text=True,
        )


if __name__ == "__main__":
    # Self-test: spawn bash, send echo, confirm we can read the output back.
    t = Tmux("drive-selftest")
    try:
        t.start(width=80, height=24, argv=["bash"])
        time.sleep(0.4)
        t.send("echo hello-from-drive", "Enter")
        assert t.wait_for_substring("hello-from-drive", timeout=3), \
            "tmux driver smoke failed — 'hello-from-drive' never appeared"
        print("[drive.py] self-test OK")
        print(t.pane())
    finally:
        t.kill()
