#!/usr/bin/env python3
"""Smoke test: exercises the real TUI (ConfigScreen → RunnerScreen → Orchestrator)
then runs independent blind verification of results.

Usage:
    python3 tests/smoke_test_e2e.py --mock          # Free, fast (~30s), synthetic Claude
    python3 tests/smoke_test_e2e.py --keep          # Real Claude, keep temp dir
    python3 tests/smoke_test_e2e.py --mock --keep   # Mock + keep dir for inspection
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
import time
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Pre-baked source files for mock mode
# ---------------------------------------------------------------------------

TRANSFORM_PY = textwrap.dedent('''\
    """String transformation utilities."""

    import re


    def slugify(text: str) -> str:
        """Convert text to a URL-friendly slug."""
        text = text.lower().strip()
        text = re.sub(r"[^a-z0-9\\s-]", "", text)
        text = re.sub(r"[\\s-]+", "-", text)
        text = text.strip("-")
        return text


    def truncate(text: str, max_len: int, suffix: str = "...") -> str:
        """Truncate text to max_len, appending suffix if truncated."""
        if len(text) <= max_len:
            return text
        if max_len < len(suffix):
            return suffix[:max_len]
        return text[: max_len - len(suffix)] + suffix
''')

FORMAT_PY = textwrap.dedent('''\
    """String formatting utilities."""

    import textwrap as _tw


    def wrap_lines(text: str, width: int) -> str:
        """Wrap text to the given line width, breaking on spaces."""
        if not text:
            return text
        paragraphs = text.split("\\n")
        wrapped = []
        for para in paragraphs:
            if not para.strip():
                wrapped.append("")
            else:
                wrapped.append(_tw.fill(para, width=width))
        return "\\n".join(wrapped)


    def indent(text: str, prefix: str) -> str:
        """Add prefix to the start of every line."""
        lines = text.split("\\n")
        return "\\n".join(prefix + line for line in lines)
''')

INIT_PY = textwrap.dedent('''\
    """String utilities package."""

    from strutils.transform import slugify, truncate
    from strutils.format import wrap_lines, indent

    __all__ = ["slugify", "truncate", "wrap_lines", "indent"]
''')

TEST_TRANSFORM_PY = textwrap.dedent('''\
    """Tests for strutils.transform."""

    from strutils.transform import slugify, truncate


    def test_slugify_basic():
        assert slugify("Hello World") == "hello-world"


    def test_slugify_special_chars():
        assert slugify("  My  Blog Post! ") == "my-blog-post"


    def test_slugify_collapse_hyphens():
        assert slugify("foo--bar") == "foo-bar"


    def test_slugify_empty():
        assert slugify("") == ""


    def test_slugify_already_slug():
        assert slugify("hello-world") == "hello-world"


    def test_truncate_no_change():
        assert truncate("hi", 10) == "hi"


    def test_truncate_exact():
        assert truncate("abcdef", 6) == "abcdef"


    def test_truncate_cut():
        assert truncate("hello world", 8) == "hello..."


    def test_truncate_short_max():
        assert truncate("abcdef", 5) == "ab..."


    def test_truncate_tiny_max():
        assert truncate("abcdef", 2) == ".."
''')

TEST_FORMAT_PY = textwrap.dedent('''\
    """Tests for strutils.format."""

    from strutils.format import wrap_lines, indent


    def test_indent_basic():
        assert indent("a\\nb", "  ") == "  a\\n  b"


    def test_indent_single_line():
        assert indent("hello", ">>> ") == ">>> hello"


    def test_indent_empty():
        assert indent("", "  ") == "  "


    def test_wrap_lines_short():
        assert wrap_lines("hi", 80) == "hi"


    def test_wrap_lines_empty():
        assert wrap_lines("", 10) == ""


    def test_wrap_lines_wraps():
        result = wrap_lines("hello world foo bar", 10)
        for line in result.split("\\n"):
            assert len(line) <= 10 or " " not in line


    def test_indent_multiline():
        result = indent("line1\\nline2\\nline3", "# ")
        lines = result.split("\\n")
        assert all(l.startswith("# ") for l in lines)


    def test_wrap_preserves_newlines():
        result = wrap_lines("para one\\n\\npara two", 80)
        assert "\\n\\n" in result
''')

# ---------------------------------------------------------------------------
# Section 1: Setup
# ---------------------------------------------------------------------------


def setup_project(base_dir: Path) -> Path:
    """Create the test project directory with git init and fixture plans."""
    project = base_dir / "strutils_project"
    project.mkdir()

    # Create strutils package placeholder
    pkg = project / "strutils"
    pkg.mkdir()
    (pkg / "__init__.py").write_text('"""String utilities."""\n')

    # Create tests dir placeholder
    tests = project / "tests"
    tests.mkdir()
    (tests / "__init__.py").write_text("")

    # Copy plan fixture files
    fixtures = Path(__file__).parent / "smoke_fixtures"
    plans = project / "plans"
    plans.mkdir()
    for name in ("plan1_implement.md", "plan1_test.md", "plan2_implement.md", "plan2_test.md"):
        src = fixtures / name
        if src.exists():
            shutil.copy(src, plans / name)
        else:
            # Fallback: create minimal plan
            (plans / name).write_text(f"# {name}\nSee instructions.\n")

    # Git init
    subprocess.run(["git", "init"], cwd=project, capture_output=True)
    subprocess.run(["git", "add", "."], cwd=project, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "Initial commit: project scaffold"],
        cwd=project, capture_output=True,
        env={**os.environ, "GIT_AUTHOR_NAME": "smoke", "GIT_AUTHOR_EMAIL": "smoke@test",
             "GIT_COMMITTER_NAME": "smoke", "GIT_COMMITTER_EMAIL": "smoke@test"},
    )

    return project


# ---------------------------------------------------------------------------
# Section 2: Run Through TUI
# ---------------------------------------------------------------------------


def _make_mock_result(session_id: str, text: str, cost: float = 0.01):
    """Create a mock ResultMessage for the SDK."""
    from claude_agent_sdk import ResultMessage
    result = MagicMock(spec=ResultMessage)
    result.is_error = False
    result.session_id = session_id
    result.result = text
    result.total_cost_usd = cost
    result.duration_ms = 2000
    result.num_turns = 3
    return result


def _write_prebaked_files(cwd: Path) -> None:
    """Write the pre-baked strutils library into the project dir."""
    pkg = cwd / "strutils"
    pkg.mkdir(exist_ok=True)
    (pkg / "__init__.py").write_text(INIT_PY)
    (pkg / "transform.py").write_text(TRANSFORM_PY)
    (pkg / "format.py").write_text(FORMAT_PY)

    tests = cwd / "tests"
    tests.mkdir(exist_ok=True)
    if not (tests / "__init__.py").exists():
        (tests / "__init__.py").write_text("")
    (tests / "test_transform.py").write_text(TEST_TRANSFORM_PY)
    (tests / "test_format.py").write_text(TEST_FORMAT_PY)


def _build_mock_query():
    """Build a mock async generator that simulates Claude writing files."""
    call_count = {"n": 0}

    async def mock_query(prompt, options):
        call_count["n"] += 1
        cwd = Path(options.cwd)

        if call_count["n"] == 1:
            # First iteration: write all the code
            _write_prebaked_files(cwd)
            result_text = (
                "I've created the strutils package with transform.py (slugify, truncate) "
                "and format.py (wrap_lines, indent). Tests are in tests/test_transform.py "
                "and tests/test_format.py. All tests pass."
            )
        else:
            # Subsequent iterations: just report success
            result_text = "All tests pass. The implementation is complete and correct."

        yield _make_mock_result(f"sess-{call_count['n']}", result_text)

    return mock_query


async def run_via_tui(project_dir: Path, mock: bool, model: str) -> "OrchestratorState":
    """Launch the TUI app headless, fill config, click Start, wait for completion."""
    # Import here to avoid import errors if ralph_tui isn't installed
    from ralph_tui.app import RalphApp
    from ralph_tui.config import RalphConfig
    from ralph_tui.orchestrator import OrchestratorState

    # Read plan files for prompt content
    plans_dir = project_dir / "plans"
    plan1_impl = (plans_dir / "plan1_implement.md").read_text()
    plan2_impl = (plans_dir / "plan2_implement.md").read_text()

    initial_prompt = (
        "You are building a Python string utilities library.\n\n"
        "## Phase 1\n" + plan1_impl + "\n\n"
        "## Phase 2\n" + plan2_impl + "\n\n"
        "Complete BOTH phases. Create all files, write all tests, "
        "and run `python3 -m pytest tests/ -v` to verify everything passes."
    )

    rerun_prompt = (
        "Continue working on the strutils library. "
        "Run `python3 -m pytest tests/ -v` and fix any failures. "
        "Ensure all functions work correctly."
    )

    # Build mock/real context managers
    if mock:
        mock_query_fn = _build_mock_query()
        async def mock_analyze(text, sys_prompt, exit_prompt, iteration_context=None):
            return MagicMock(should_stop=False, reason="continue", summary="continuing")

        query_ctx = patch("ralph_tui.orchestrator.query", side_effect=mock_query_fn)
        analyze_ctx = patch("ralph_tui.orchestrator.analyze_output", side_effect=mock_analyze)
    else:
        query_ctx = nullcontext()
        analyze_ctx = nullcontext()

    app = RalphApp(launch_cwd=str(project_dir))

    with query_ctx, analyze_ctx:
        async with app.run_test(size=(120, 40)) as pilot:
            # Wait for ConfigScreen to mount (pushed in on_mount)
            from ralph_tui.screens.config_screen import ConfigScreen
            from ralph_tui.screens.runner_screen import RunnerScreen
            from textual.widgets import Input, TextArea, Button

            # Poll until ConfigScreen is active
            for _ in range(20):
                await pilot.pause(0.2)
                if isinstance(pilot.app.screen, ConfigScreen):
                    break

            screen = pilot.app.screen
            assert isinstance(screen, ConfigScreen), (
                f"Expected ConfigScreen, got {type(screen).__name__}"
            )

            # Project path should already be set from launch_cwd
            # Fill prompts via screen.query_one (not app.query_one)
            screen.query_one("#ta-initial-prompt", TextArea).text = initial_prompt
            screen.query_one("#ta-rerun-prompt", TextArea).text = rerun_prompt

            # Set iteration counts — small for smoke test
            screen.query_one("#in-min-iter", Input).value = "1"
            screen.query_one("#in-max-iter", Input).value = "2"
            screen.query_one("#in-max-error-retries", Input).value = "2"
            screen.query_one("#in-max-consecutive-errors", Input).value = "5"

            await pilot.pause(0.2)

            # Click Start → validation + screen transition
            # Use screen.query_one to find the button and press it
            screen.query_one("#btn-start", Button).press()
            await pilot.pause(1.0)

            # --- RUNNER SCREEN ---
            # Wait for RunnerScreen to be pushed
            for _ in range(20):
                await pilot.pause(0.2)
                if isinstance(pilot.app.screen, RunnerScreen):
                    break

            runner_screen = pilot.app.screen
            if not isinstance(runner_screen, RunnerScreen):
                # Check for validation errors on config screen
                if isinstance(runner_screen, ConfigScreen):
                    err = runner_screen.query_one("#error-display").renderable
                    raise AssertionError(
                        f"Still on ConfigScreen. Error display: {err}"
                    )
                raise AssertionError(
                    f"Expected RunnerScreen, got {type(runner_screen).__name__}"
                )

            # Poll until orchestrator finishes or timeout
            timeout_sec = 30 if mock else 900
            for _ in range(timeout_sec):
                await pilot.pause(1.0)
                if runner_screen.orchestrator and runner_screen.orchestrator.state.status != "idle":
                    status = runner_screen.orchestrator.state.status
                    if any(kw in status for kw in ("Completed", "Stopped", "Reached max", "Circuit breaker")):
                        break

            assert runner_screen.orchestrator is not None, "Orchestrator never started"
            return runner_screen.orchestrator.state


# ---------------------------------------------------------------------------
# Section 3: Blind Verification (ZERO ralph-tui imports)
# ---------------------------------------------------------------------------


def verify_results(project_dir: Path) -> list[tuple[str, bool, str]]:
    """Run independent verification checks. Returns list of (name, passed, detail)."""
    results: list[tuple[str, bool, str]] = []

    # Find the iteration dir with actual output
    # The orchestrator creates iteration dirs in a sibling -ralph-runs directory
    runs_dir = project_dir.parent / f"{project_dir.name}-ralph-runs"

    # Find the latest iteration dir
    iter_dirs = sorted(runs_dir.glob("iteration-*")) if runs_dir.exists() else []
    if not iter_dirs:
        results.append(("iteration_dirs_exist", False, f"No iteration dirs found in {runs_dir}"))
        return results
    results.append(("iteration_dirs_exist", True, f"Found {len(iter_dirs)} iteration dir(s)"))

    # Use the last iteration directory for verification
    work_dir = iter_dirs[-1]

    # --- Layer A: File existence ---
    expected_files = [
        "strutils/__init__.py",
        "strutils/transform.py",
        "strutils/format.py",
        "tests/test_transform.py",
        "tests/test_format.py",
    ]
    for rel_path in expected_files:
        exists = (work_dir / rel_path).exists()
        results.append((f"file_exists:{rel_path}", exists, str(work_dir / rel_path)))

    # --- Layer B: Functional correctness (dynamic import) ---
    # We add the work dir to sys.path temporarily and import the module
    old_path = sys.path.copy()
    sys.path.insert(0, str(work_dir))
    try:
        # Clear any cached imports
        for mod_name in list(sys.modules):
            if mod_name.startswith("strutils"):
                del sys.modules[mod_name]

        try:
            from strutils.transform import slugify, truncate
            results.append(("import_transform", True, ""))

            # slugify checks
            r = slugify("Hello World")
            results.append(("slugify_hello_world", r == "hello-world", f"got: {r!r}"))

            r = truncate("hello world", 8)
            results.append(("truncate_cut", r == "hello...", f"got: {r!r}"))

            r = truncate("hi", 10)
            results.append(("truncate_no_change", r == "hi", f"got: {r!r}"))

        except Exception as e:
            results.append(("import_transform", False, str(e)))

        # Clear again before next import
        for mod_name in list(sys.modules):
            if mod_name.startswith("strutils"):
                del sys.modules[mod_name]

        try:
            from strutils.format import indent
            results.append(("import_format", True, ""))

            r = indent("a\nb", "  ")
            results.append(("indent_basic", r == "  a\n  b", f"got: {r!r}"))

        except Exception as e:
            results.append(("import_format", False, str(e)))

    finally:
        sys.path = old_path
        # Clean up imports
        for mod_name in list(sys.modules):
            if mod_name.startswith("strutils"):
                del sys.modules[mod_name]

    # --- Layer C: Run pytest independently ---
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "pytest", "tests/", "-v", "--tb=short"],
            cwd=work_dir,
            capture_output=True,
            text=True,
            timeout=60,
        )
        passed = proc.returncode == 0
        detail = proc.stdout[-500:] if proc.stdout else proc.stderr[-500:]
        results.append(("pytest_pass", passed, detail.strip()))
    except subprocess.TimeoutExpired:
        results.append(("pytest_pass", False, "pytest timed out"))
    except Exception as e:
        results.append(("pytest_pass", False, str(e)))

    # --- Layer D: Ralph log file ---
    log_file = runs_dir / "ralph-log.jsonl"
    if log_file.exists():
        lines = [l for l in log_file.read_text().strip().splitlines() if l.strip()]
        iteration_entries = []
        for line in lines:
            try:
                entry = json.loads(line)
                if "iteration" in entry and "cost_usd" in entry:
                    iteration_entries.append(entry)
            except json.JSONDecodeError:
                pass
        results.append(("log_has_iterations", len(iteration_entries) > 0,
                        f"{len(iteration_entries)} iteration log entries"))
    else:
        results.append(("log_has_iterations", False, "ralph-log.jsonl not found"))

    return results


# ---------------------------------------------------------------------------
# Section 4: Report & Main
# ---------------------------------------------------------------------------


def print_report(
    verification: list[tuple[str, bool, str]],
    state: object | None,
    project_dir: Path,
    elapsed: float,
    mock: bool,
    keep: bool,
) -> bool:
    """Print verification results. Returns True if all passed."""
    print("\n" + "=" * 70)
    print("  SMOKE TEST REPORT")
    print("=" * 70)
    print(f"  Mode:       {'MOCK' if mock else 'REAL (Claude)'}")
    print(f"  Project:    {project_dir}")
    print(f"  Elapsed:    {elapsed:.1f}s")

    if state:
        cost = getattr(state, "total_cost_usd", 0)
        eff = getattr(state, "effective_iterations", 0)
        status = getattr(state, "status", "unknown")
        num_results = len(getattr(state, "results", []))
        print(f"  Status:     {status}")
        print(f"  Effective:  {eff} iterations")
        print(f"  Raw iters:  {num_results}")
        print(f"  Cost:       ${cost:.4f}")

    print("\n  VERIFICATION RESULTS:")
    print("  " + "-" * 66)

    all_passed = True
    for name, passed, detail in verification:
        icon = "PASS" if passed else "FAIL"
        marker = "  " if passed else ">>"
        print(f"  {marker} [{icon}] {name}")
        if detail and not passed:
            # Show detail for failures
            for line in detail.splitlines()[:5]:
                print(f"           {line}")
        if not passed:
            all_passed = False

    print("  " + "-" * 66)
    overall = "ALL PASSED" if all_passed else "SOME FAILED"
    print(f"  OVERALL: {overall}")

    if keep:
        print(f"\n  Kept directory: {project_dir}")
        runs = project_dir.parent / f"{project_dir.name}-ralph-runs"
        if runs.exists():
            print(f"  Runs directory: {runs}")

    print("=" * 70)
    return all_passed


async def async_main(args: argparse.Namespace) -> int:
    """Async entry point."""
    start = time.time()

    # Setup
    base_dir = Path(tempfile.mkdtemp(prefix="ralph-smoke-"))
    project_dir = setup_project(base_dir)
    print(f"[setup] Project created at {project_dir}")

    state = None
    try:
        # Run through TUI
        print(f"[tui] Launching TUI ({'mock' if args.mock else 'real'} mode)...")
        state = await run_via_tui(project_dir, args.mock, args.model)
        print(f"[tui] Orchestrator finished: {state.status}")
        print(f"[tui] Effective iterations: {state.effective_iterations}, Cost: ${state.total_cost_usd:.4f}")

        # Verify
        print("[verify] Running blind verification...")
        verification = verify_results(project_dir)

        # Report
        elapsed = time.time() - start
        all_passed = print_report(verification, state, project_dir, elapsed, args.mock, args.keep)

        return 0 if all_passed else 1

    except Exception as e:
        print(f"\n[ERROR] Smoke test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return 1

    finally:
        if not args.keep:
            # Clean up temp dir and runs dir
            shutil.rmtree(base_dir, ignore_errors=True)
            runs_dir = project_dir.parent / f"{project_dir.name}-ralph-runs"
            if runs_dir.exists():
                shutil.rmtree(runs_dir, ignore_errors=True)
            print(f"[cleanup] Removed {base_dir}")
        else:
            print(f"[cleanup] Kept {base_dir}")


def main():
    parser = argparse.ArgumentParser(description="Ralph TUI E2E Smoke Test")
    parser.add_argument("--mock", action="store_true",
                        help="Use synthetic Claude results (free, fast)")
    parser.add_argument("--model", default="claude-sonnet-4-20250514",
                        help="Model to use for real runs")
    parser.add_argument("--keep", action="store_true",
                        help="Don't delete temp dir after test")
    args = parser.parse_args()

    # Ensure we're in the right directory for imports
    project_root = Path(__file__).parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    exit_code = asyncio.run(async_main(args))
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
