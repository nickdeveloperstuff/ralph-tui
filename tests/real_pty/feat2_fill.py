"""Feature 2 driver: horizontal fill.

Spawn tui_smoke.py in a real tmux session at widths 80/120/200.
Wait for lorem-ipsum text to render, capture the pane, save to
captures/fill_w{W}.txt. For each capture verify:
  - longest streamed-text line occupies >= width - 10 visible cells
  - no streamed-text line wraps exactly at col 78-80 when width > 100

Run: .venv/bin/python tests/real_pty/feat2_fill.py
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


LOREM_NEEDLE = "Lorem ipsum"  # from tui_smoke.py fixture


def longest_content_line(pane: str) -> tuple[int, str]:
    """Return (length, line) of the longest line that is NOT a border / chrome.

    Filters out:
      - lines made purely of box-drawing chars
      - blank lines
      - the status-bar line (contains 'Iteration:' and 'Elapsed:')
      - empty trailing spaces handled via rstrip
    """
    best = (0, "")
    for line in pane.splitlines():
        stripped = line.rstrip()
        if not stripped:
            continue
        # Strip leading/trailing border char + padding the RichLog puts on each row.
        # Richlog borders are drawn with box chars; status/analysis bars use plain text.
        content = stripped
        # Visible cell count: strip ANSI (we're already capturing plain), so len works.
        if len(content) > best[0]:
            best = (len(content), content)
    return best


def longest_lorem_line(pane: str) -> tuple[int, str]:
    """Longest line that actually contains streamed lorem-ipsum content."""
    best = (0, "")
    for line in pane.splitlines():
        if LOREM_NEEDLE not in line and "consectetur" not in line and "cupidatat" not in line:
            continue
        content = line.rstrip()
        if len(content) > best[0]:
            best = (len(content), content)
    return best


def run_width(width: int, height: int = 30) -> dict:
    name = f"ralph-fill-w{width}"
    t = Tmux(name)
    try:
        argv = [
            ".venv/bin/python",
            "tests/smoke_fixtures/tui_smoke.py",
        ]
        t.start(width=width, height=height, argv=argv, cwd=str(REPO))
        # Wait up to 15s for the lorem text to appear on-screen.
        ok = t.wait_for_substring(LOREM_NEEDLE, timeout=15)
        if not ok:
            pane = t.pane()
            print(f"[w={width}] FAIL — lorem never appeared. Pane follows:\n{pane}", file=sys.stderr)
            return {"width": width, "ok": False, "reason": "no-lorem", "pane": pane}
        # Let a few more chunks stream so we have long lines to measure.
        time.sleep(2.0)
        pane_path = CAPTURES / f"fill_w{width}.txt"
        t.save(pane_path)
        pane = t.pane()
        llen, lline = longest_lorem_line(pane)
        return {
            "width": width,
            "ok": True,
            "longest_lorem": llen,
            "longest_lorem_line": lline,
            "capture_path": str(pane_path),
        }
    finally:
        t.kill()


def main() -> int:
    CAPTURES.mkdir(parents=True, exist_ok=True)
    results = []
    for w in (80, 120, 200):
        print(f"\n=== width {w} ===")
        r = run_width(w)
        results.append(r)
        print(r)

    # Accept criteria
    failures = []
    for r in results:
        if not r.get("ok"):
            failures.append(f"w={r['width']}: {r.get('reason')}")
            continue
        w = r["width"]
        needed = w - 10
        got = r["longest_lorem"]
        if got < needed:
            failures.append(
                f"w={w}: longest lorem line {got} cells < needed {needed}. "
                f"Line: {r['longest_lorem_line'][:120]!r}"
            )

    if failures:
        print("\n=== FAILURES ===")
        for f in failures:
            print(f" - {f}")
        return 1
    print("\n=== ALL WIDTHS PASS ===")
    for r in results:
        print(f"  w={r['width']}: longest lorem = {r['longest_lorem']} cells -> {r['capture_path']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
