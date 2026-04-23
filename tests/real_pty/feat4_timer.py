"""Feature 4 driver: activity timer during in-flight tool.

Run long_tool.py (50s silent Bash, then 70s idle, then completion) in a
real tmux at w=120, h=30. Capture every 5s for 30s of the in-tool window,
then one more ~75s after tool_end during the post-tool idle, and confirm:

  - during the tool: Tool: Bash visible, Last activity age <= 2, no STALL WARNING
  - after tool_end + 70s idle: STALL WARNING appears (control: legitimate stall
    still trips)
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

AGE_RE = re.compile(r"Last activity:\s+(\d+)s ago")


def parse_age(pane: str) -> int | None:
    m = AGE_RE.search(pane)
    if not m:
        return None
    return int(m.group(1))


def main() -> int:
    CAPTURES.mkdir(parents=True, exist_ok=True)
    t = Tmux("ralph-timer")
    try:
        argv = [".venv/bin/python", "tests/smoke_fixtures/long_tool.py"]
        t.start(width=120, height=30, argv=argv, cwd=str(REPO))
        # Wait for Tool: Bash to appear in status bar (tool_start event landed).
        if not t.wait_for_substring("Tool: Bash", timeout=15):
            print("[timer] Tool: Bash never appeared")
            print(t.pane())
            return 1

        samples: list[tuple[int, int | None, bool]] = []  # (t_sec, age, stall)
        # t=0 reference: after we see Tool: Bash, mark start.
        t0 = time.monotonic()
        for i in range(1, 8):  # 7 samples at 5s, 10s, ..., 35s
            target = t0 + i * 5
            while time.monotonic() < target:
                time.sleep(0.1)
            pane = t.pane()
            cap_path = CAPTURES / f"timer_t{i*5:02d}_w120.txt"
            cap_path.write_text(pane)
            age = parse_age(pane)
            stall = "STALL WARNING" in pane
            samples.append((i * 5, age, stall))
            print(f"[timer] t={i*5:02d}s  age={age}  stall={stall}  -> {cap_path.name}")

        # Fixture timeline relative to tool_start:
        #   +0s      tool_start (Bash)
        #   +50s     tool_end + text "Bash complete." (resets timer)
        #   +110.5s  STALL WARNING threshold crossed (60s past last text write)
        #   +120.5s  ResultMessage arrives -> RunComplete -> timer reset
        # Last sample was at my-t=35. Sleep 78s -> my-t=113, inside the
        # [110.5, 120.5] window, capturing a live STALL WARNING.
        print("[timer] sleeping 78s to observe post-tool idle STALL WARNING")
        time.sleep(78)
        pane = t.pane()
        cap_path = CAPTURES / "timer_post_tool_idle_w120.txt"
        cap_path.write_text(pane)
        post_age = parse_age(pane)
        post_stall = "STALL WARNING" in pane
        print(f"[timer] post-tool idle  age={post_age}  stall={post_stall}  -> {cap_path.name}")

        # Accept criteria
        failures = []
        for ts, age, stall in samples:
            if age is None:
                failures.append(f"t={ts}: no 'Last activity' parsed from capture")
                continue
            if age > 2:
                failures.append(f"t={ts}: activity age = {age}s > 2")
            if stall:
                failures.append(f"t={ts}: STALL WARNING appeared during in-flight tool")
        if not post_stall:
            failures.append("post-tool idle: STALL WARNING did NOT appear (control regressed)")

        if failures:
            print("\n=== FAILURES ===")
            for f in failures:
                print(f" - {f}")
            return 1
        print("\n=== FEATURE 4 PASSES ===")
        return 0
    finally:
        t.kill()


if __name__ == "__main__":
    sys.exit(main())
