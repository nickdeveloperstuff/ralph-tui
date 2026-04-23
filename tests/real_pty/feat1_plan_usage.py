"""Feature 1 driver: auto-resume on plan-usage limit.

Flow:
  1. Spawn plan_usage_limit_tui.py in tmux at w=120, h=30.
  2. Wait for "Plan usage limit hit" banner.
  3. Capture every 5s for 35s -> plan_usage_t05.txt ... plan_usage_t35.txt.
     Parse the "(Ns left)" countdown from the status bar; assert it strictly
     decreases across captures.
  4. Mid-wait (~t=15s) send a harmless keystroke and confirm the next
     capture differs from the previous one (TUI is alive, not frozen).
  5. Wait for "Smoke run complete" to appear; save plan_usage_resumed.txt.
"""

from __future__ import annotations

import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from drive import Tmux


REPO = Path(__file__).resolve().parents[2]
CAPTURES = REPO / "tests" / "real_pty" / "captures"

COUNTDOWN_RE = re.compile(r"\((\d+)s left\)")


def parse_countdown(pane: str) -> int | None:
    m = COUNTDOWN_RE.search(pane)
    if not m:
        return None
    return int(m.group(1))


def main() -> int:
    CAPTURES.mkdir(parents=True, exist_ok=True)
    t = Tmux("ralph-plan-usage")
    try:
        argv = [
            ".venv/bin/python",
            "tests/smoke_fixtures/plan_usage_limit_tui.py",
        ]
        t.start(width=120, height=30, argv=argv, cwd=str(REPO))
        if not t.wait_for_substring("Plan usage limit hit", timeout=15):
            print("[plan] 'Plan usage limit hit' never appeared")
            print(t.pane())
            return 1
        # A beat for the countdown text to settle.
        time.sleep(0.5)

        samples = []  # list[(t_sec, countdown, pane_text)]
        t0 = time.monotonic()
        for i in range(1, 8):
            target = t0 + i * 5
            while time.monotonic() < target:
                time.sleep(0.1)
            # Around t=15s, send a harmless key before capturing.
            if i == 3:
                # Arrow keys are a no-op on RunnerScreen but tmux/Textual
                # still process them — a live app redraws, a frozen one
                # leaves the pane unchanged aside from the wall-clock tick.
                t.send("Down")
                time.sleep(0.2)
            pane = t.pane()
            cap = CAPTURES / f"plan_usage_t{i*5:02d}.txt"
            cap.write_text(pane)
            cd = parse_countdown(pane)
            samples.append((i * 5, cd, pane))
            print(f"[plan] t={i*5:02d}s  countdown={cd}  -> {cap.name}")

        # Wait for resume: the runner writes "[Resumed at ...]" when the
        # countdown ends AND the next query succeeds, and "Run complete."
        # when the orchestrator finishes. Either one is sufficient evidence
        # of auto-resume; we want both for a stronger signal.
        print("[plan] waiting for '[Resumed at' (up to 180s)")
        if not t.wait_for_substring("[Resumed at", timeout=180):
            print("[plan] '[Resumed at' never appeared")
            (CAPTURES / "plan_usage_timeout.txt").write_text(t.pane())
            return 1
        # Give the orchestrator a moment to emit "Run complete."
        if not t.wait_for_substring("Run complete.", timeout=30):
            print("[plan] 'Run complete.' never appeared (countdown ended but run didn't finish)")
            (CAPTURES / "plan_usage_timeout.txt").write_text(t.pane())
            return 1
        time.sleep(0.3)
        pane = t.pane()
        (CAPTURES / "plan_usage_resumed.txt").write_text(pane)
        print("[plan] captured plan_usage_resumed.txt")

        # Accept criteria
        failures = []
        # Countdown strictly decreases (allow the first None if status bar
        # briefly shows the '[Plan usage limit hit]' text write before the
        # first countdown tick).
        cds = [(ts, cd) for ts, cd, _ in samples if cd is not None]
        if len(cds) < 3:
            failures.append(
                f"saw only {len(cds)} captures with a countdown — the status bar "
                f"is not showing '(Ns left)'"
            )
        else:
            for (t1, cd1), (t2, cd2) in zip(cds, cds[1:]):
                if cd2 >= cd1:
                    failures.append(
                        f"countdown did not decrease between t={t1}s ({cd1}) and t={t2}s ({cd2})"
                    )

        # TUI is alive: every consecutive capture differs somewhere.
        for (t1, _, p1), (t2, _, p2) in zip(samples, samples[1:]):
            if p1 == p2:
                failures.append(f"captures at t={t1} and t={t2} were byte-identical — TUI frozen?")

        # Resume evidence landed
        if "[Resumed at" not in pane:
            failures.append("plan_usage_resumed.txt does not contain '[Resumed at'")
        if "Run complete." not in pane:
            failures.append("plan_usage_resumed.txt does not contain 'Run complete.'")

        if failures:
            print("\n=== FAILURES ===")
            for f in failures:
                print(f" - {f}")
            return 1
        print("\n=== FEATURE 1 PASSES ===")
        print(f"  countdown samples: {cds}")
        return 0
    finally:
        t.kill()


if __name__ == "__main__":
    sys.exit(main())
