"""Feature 3 driver: sticky scroll.

Flow (real tmux at w=120, h=30):
  1. Spawn tui_smoke.py; wait for lorem content + let several viewports
     worth of text stream in.
  2. Capture initial pane. Save scroll_initial_w120.txt.
  3. Send PageUp x3. Capture top visible content line -> frozen_top.
     Save scroll_paged_up_w120.txt.
  4. Sleep 4s while more text streams. Capture again.
     Save scroll_after_more_text_w120.txt.
  5. Extract the top content line of #3 and #4. They MUST match.
     (i.e., view did not yank to tail despite concurrent writes)
  6. Send End. Sleep 0.5s. Capture.
     Save scroll_returned_w120.txt.
  7. Extract the last content line of #6 — it must contain streamed text
     that was NOT visible at step #3 (we moved back to the tail).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from drive import Tmux


REPO = Path(__file__).resolve().parents[2]
CAPTURES = REPO / "tests" / "real_pty" / "captures"

WIDTH = 120
HEIGHT = 30


def content_lines(pane: str) -> list[str]:
    """Return the lines of pane that are inside the output-log RichLog box.

    The log is framed by top '┌' and bottom '└' border rows. Content rows
    start with ' │' and end with '│'. Extract the inner content (no border).
    """
    lines = pane.splitlines()
    # Find the first '┌' row and the first '└' row after it (top-level log box)
    top_idx = next((i for i, l in enumerate(lines) if l.lstrip().startswith("┌")), None)
    if top_idx is None:
        return []
    bot_idx = next(
        (i for i, l in enumerate(lines[top_idx + 1:], top_idx + 1) if l.lstrip().startswith("└")),
        None,
    )
    if bot_idx is None:
        return []
    out = []
    for l in lines[top_idx + 1:bot_idx]:
        stripped = l.strip()
        # inner content is between leading '│' and trailing '│'
        if stripped.startswith("│") and stripped.endswith("│"):
            inner = stripped[1:-1]
            # normalize trailing RichLog scrollbar chars (▅ etc) — they can
            # appear attached to the right border, polluting content compare.
            # Drop trailing block glyphs.
            inner = inner.rstrip(" ▅▃▁▂▄▆▇█▔")
            out.append(inner)
        else:
            out.append(stripped)
    return out


def first_nonblank(lines: list[str]) -> str:
    for l in lines:
        if l.strip():
            return l
    return ""


def last_nonblank(lines: list[str]) -> str:
    for l in reversed(lines):
        if l.strip():
            return l
    return ""


def main() -> int:
    CAPTURES.mkdir(parents=True, exist_ok=True)
    t = Tmux("ralph-scroll")
    try:
        argv = [".venv/bin/python", "tests/smoke_fixtures/tui_smoke.py"]
        t.start(width=WIDTH, height=HEIGHT, argv=argv, cwd=str(REPO))
        if not t.wait_for_substring("Lorem ipsum", timeout=15):
            print("[scroll] never saw lorem text")
            print(t.pane())
            return 1
        # Let enough content stream so that we have more than one viewport.
        # tui_smoke emits a chunk every 0.5s; 4 seconds = ~8 chunks, plenty
        # of lines to fill a 30-row terminal's log area multiple times over.
        time.sleep(4.0)

        t.save(CAPTURES / "scroll_initial_w120.txt")
        print("[scroll] step 1 captured scroll_initial_w120.txt")

        # Scroll up
        t.send("PageUp")
        time.sleep(0.2)
        t.send("PageUp")
        time.sleep(0.2)
        t.send("PageUp")
        time.sleep(0.4)  # let the scroll settle

        pane_paged = t.pane()
        (CAPTURES / "scroll_paged_up_w120.txt").write_text(pane_paged)
        paged_top = first_nonblank(content_lines(pane_paged))
        print(f"[scroll] step 2 captured scroll_paged_up_w120.txt  top={paged_top!r}")

        # Wait for more writes — if sticky is broken, the view yanks.
        time.sleep(4.0)
        pane_after = t.pane()
        (CAPTURES / "scroll_after_more_text_w120.txt").write_text(pane_after)
        after_top = first_nonblank(content_lines(pane_after))
        print(f"[scroll] step 3 captured scroll_after_more_text_w120.txt  top={after_top!r}")

        # Return to tail
        t.send("End")
        time.sleep(0.8)  # allow scroll_end + any pending writes to land
        pane_ret = t.pane()
        (CAPTURES / "scroll_returned_w120.txt").write_text(pane_ret)
        ret_top = first_nonblank(content_lines(pane_ret))
        ret_last = last_nonblank(content_lines(pane_ret))
        print(f"[scroll] step 4 captured scroll_returned_w120.txt  top={ret_top!r}  bottom={ret_last!r}")

        # Accept criteria
        failures = []
        if paged_top != after_top:
            failures.append(
                "top line after concurrent writes DID NOT match the paged-up top line\n"
                f"  paged_top: {paged_top!r}\n"
                f"  after_top: {after_top!r}"
            )
        if not ret_last.strip():
            failures.append("return-to-tail capture had no content at bottom")
        if ret_top == paged_top:
            failures.append(
                "return-to-tail top line was identical to paged-up top — End did not re-engage scroll"
            )

        if failures:
            print("\n=== FAILURES ===")
            for f in failures:
                print(f" - {f}")
            return 1
        print("\n=== STICKY SCROLL PASSES ===")
        print(f"  paged_top == after_top: {paged_top!r}")
        print(f"  returned top:  {ret_top!r}")
        print(f"  returned last: {ret_last!r}")
        return 0
    finally:
        t.kill()


if __name__ == "__main__":
    sys.exit(main())
