"""End-to-end test for the three-prompt system.

Runs a full 5-iteration orchestration with mocked Claude:
  - Iteration 1: initial_prompt ("Create a greeting module")
  - Iterations 2-3: rerun_prompt ("Add tests for the greeting module")
  - Iterations 4-5: final_prompt ("Polish and finalize")

Verifies:
  1. Correct prompt is sent at each iteration
  2. File copying chains correctly across all phases
  3. Each iteration gets its own directory
  4. Changes from Claude accumulate across iterations
  5. Iteration numbering is correct in directory names
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ralph_tui.config import RalphConfig
from ralph_tui.orchestrator import Orchestrator, IterationResult


INITIAL_PROMPT = "Create a greeting module"
RERUN_PROMPT = "Add tests for the greeting module"
FINAL_PROMPT = "Polish and finalize"


def _make_e2e_config(project_path: str) -> RalphConfig:
    """Build a config for the 5-iteration E2E test."""
    return RalphConfig(
        project_path=project_path,
        initial_prompt=INITIAL_PROMPT,
        rerun_prompt=RERUN_PROMPT,
        final_prompt=FINAL_PROMPT,
        transition_iteration=4,  # Switch to final at iteration 4
        min_iterations=1,        # Analyze from iteration 1 onward
        max_iterations=5,
    )


class TestEndToEndThreePrompt:
    """Full end-to-end test simulating 5 iterations with three prompts."""

    @pytest.mark.asyncio
    async def test_five_iterations_correct_prompt_per_phase(self, tmp_path):
        """Each iteration should receive the correct prompt based on its phase."""
        project = tmp_path / "myproject"
        project.mkdir()
        (project / "README.md").write_text("# My Project\n")

        cfg = _make_e2e_config(str(project))
        prompts_received = []
        cwds_received = []
        status_messages = []
        text_messages = []
        iteration_results: list[IterationResult] = []

        async def on_status(s):
            status_messages.append(s)

        async def on_text(t):
            text_messages.append(t)

        async def on_iter_done(r):
            iteration_results.append(r)

        orch = Orchestrator(
            cfg,
            on_status=on_status,
            on_text=on_text,
            on_iteration_done=on_iter_done,
        )

        async def mock_query(prompt, options):
            prompts_received.append(prompt)
            cwds_received.append(options.cwd)

            # Simulate Claude making a change based on the prompt
            cwd = Path(options.cwd)
            iteration_num = len(prompts_received)

            if iteration_num == 1:
                # Initial: create greeting.py
                (cwd / "greeting.py").write_text("def greet(name):\n    return f'Hello, {name}!'\n")
            elif iteration_num == 2:
                # Rerun phase: add test file
                (cwd / "test_greeting.py").write_text(
                    "from greeting import greet\n\ndef test_greet():\n    assert greet('World') == 'Hello, World!'\n"
                )
            elif iteration_num == 3:
                # Rerun phase: add another test
                existing = (cwd / "test_greeting.py").read_text()
                (cwd / "test_greeting.py").write_text(
                    existing + "\ndef test_greet_empty():\n    assert greet('') == 'Hello, !'\n"
                )
            elif iteration_num == 4:
                # Final phase: add docstring
                (cwd / "greeting.py").write_text(
                    '"""Greeting module."""\n\ndef greet(name: str) -> str:\n    """Return a greeting."""\n    return f\'Hello, {name}!\'\n'
                )
            elif iteration_num == 5:
                # Final phase: update README
                (cwd / "README.md").write_text("# My Project\n\nA greeting module with tests.\n")

            from claude_agent_sdk import ResultMessage
            result = MagicMock(spec=ResultMessage)
            result.is_error = False
            result.session_id = f"sess-{iteration_num}"
            result.result = f"Completed iteration {iteration_num}"
            result.total_cost_usd = 0.10 * iteration_num
            result.duration_ms = 1000 * iteration_num
            result.num_turns = iteration_num * 2
            yield result

        with patch("ralph_tui.orchestrator.query", side_effect=mock_query), \
             patch("ralph_tui.orchestrator.analyze_output", new_callable=AsyncMock) as mock_analyze:
            # Let all iterations run (don't stop early)
            mock_analyze.return_value = MagicMock(
                should_stop=False, reason="keep going", summary="not done yet"
            )
            state = await orch.run()

        # --- Verify prompt selection ---
        assert len(prompts_received) == 5, f"Expected 5 prompts, got {len(prompts_received)}"

        # Iteration 1: initial prompt
        assert prompts_received[0].startswith(INITIAL_PROMPT)

        # Iterations 2-3: rerun prompt
        assert prompts_received[1].startswith(RERUN_PROMPT)
        assert prompts_received[2].startswith(RERUN_PROMPT)

        # Iterations 4-5: final prompt (transition_iteration=4)
        assert prompts_received[3].startswith(FINAL_PROMPT)
        assert prompts_received[4].startswith(FINAL_PROMPT)

    @pytest.mark.asyncio
    async def test_five_iterations_file_chaining_across_phases(self, tmp_path):
        """Files created in earlier iterations should be available in later ones."""
        project = tmp_path / "myproject"
        project.mkdir()
        (project / "README.md").write_text("# My Project\n")

        cfg = _make_e2e_config(str(project))
        orch = Orchestrator(cfg)

        iteration_dirs: list[Path] = []

        async def mock_query(prompt, options):
            cwd = Path(options.cwd)
            iteration_dirs.append(cwd)
            iteration_num = len(iteration_dirs)

            # Each iteration creates a file named after its phase
            phase = "initial" if iteration_num == 1 else (
                "rerun" if iteration_num <= 3 else "final"
            )
            (cwd / f"iter{iteration_num}_{phase}.txt").write_text(
                f"Created in iteration {iteration_num} ({phase} phase)"
            )

            from claude_agent_sdk import ResultMessage
            result = MagicMock(spec=ResultMessage)
            result.is_error = False
            result.session_id = f"sess-{iteration_num}"
            result.result = f"Done iter {iteration_num}"
            result.total_cost_usd = 0.01
            result.duration_ms = 100
            result.num_turns = 1
            yield result

        with patch("ralph_tui.orchestrator.query", side_effect=mock_query), \
             patch("ralph_tui.orchestrator.analyze_output", new_callable=AsyncMock) as mock_analyze:
            mock_analyze.return_value = MagicMock(
                should_stop=False, reason="continue", summary="continue"
            )
            state = await orch.run()

        assert len(iteration_dirs) == 5

        # Verify directory naming
        runs_dir = project.parent / "myproject-ralph-runs"
        assert iteration_dirs[0] == runs_dir / "iteration-001"
        assert iteration_dirs[1] == runs_dir / "iteration-002"
        assert iteration_dirs[2] == runs_dir / "iteration-003"
        assert iteration_dirs[3] == runs_dir / "iteration-004"
        assert iteration_dirs[4] == runs_dir / "iteration-005"

        # Verify file chaining: each iteration has all files from previous iterations
        # Iteration 5 (the last) should have files from all 4 previous iterations
        final_dir = iteration_dirs[4]
        assert (final_dir / "README.md").exists(), "Original README should be preserved"
        assert (final_dir / "iter1_initial.txt").exists(), "File from iteration 1 should chain"
        assert (final_dir / "iter2_rerun.txt").exists(), "File from iteration 2 should chain"
        assert (final_dir / "iter3_rerun.txt").exists(), "File from iteration 3 should chain"
        assert (final_dir / "iter4_final.txt").exists(), "File from iteration 4 should chain"

        # Verify contents are correct
        assert "initial phase" in (final_dir / "iter1_initial.txt").read_text()
        assert "rerun phase" in (final_dir / "iter2_rerun.txt").read_text()
        assert "final phase" in (final_dir / "iter4_final.txt").read_text()

        # Verify CLAUDE.md is present in the last iteration directory
        # (prior iteration dirs are deleted after copy to save disk space)
        assert (iteration_dirs[-1] / "CLAUDE.md").exists(), "CLAUDE.md missing in final iteration"

    @pytest.mark.asyncio
    async def test_five_iterations_cost_and_metadata_tracking(self, tmp_path):
        """Verify costs, durations, and turn counts accumulate correctly across 5 iterations."""
        project = tmp_path / "myproject"
        project.mkdir()
        (project / "file.txt").write_text("content")

        cfg = _make_e2e_config(str(project))
        orch = Orchestrator(cfg)

        async def mock_query(prompt, options):
            from claude_agent_sdk import ResultMessage
            result = MagicMock(spec=ResultMessage)
            result.is_error = False
            result.session_id = "sess"
            result.result = "Done"
            result.total_cost_usd = 0.10
            result.duration_ms = 2000
            result.num_turns = 5
            yield result

        with patch("ralph_tui.orchestrator.query", side_effect=mock_query), \
             patch("ralph_tui.orchestrator.analyze_output", new_callable=AsyncMock) as mock_analyze:
            mock_analyze.return_value = MagicMock(
                should_stop=False, reason="continue", summary="continue"
            )
            state = await orch.run()

        # 5 iterations x $0.10 = $0.50 total
        assert abs(state.total_cost_usd - 0.50) < 0.001
        assert len(state.results) == 5

        # Verify each iteration result exists
        for i, result in enumerate(state.results, 1):
            assert result.iteration == i
            assert result.cost_usd == 0.10
            assert result.duration_ms == 2000
            assert result.num_turns == 5

    @pytest.mark.asyncio
    async def test_two_prompt_backward_compatibility(self, tmp_path):
        """With transition_iteration=0, the system behaves exactly as the old 2-prompt system."""
        project = tmp_path / "myproject"
        project.mkdir()
        (project / "file.txt").write_text("content")

        cfg = RalphConfig(
            project_path=str(project),
            initial_prompt="Start here",
            rerun_prompt="Continue here",
            final_prompt="",           # empty = disabled
            transition_iteration=0,    # 0 = disabled
            min_iterations=1,
            max_iterations=3,
        )
        orch = Orchestrator(cfg)
        prompts_used = []

        async def mock_query(prompt, options):
            prompts_used.append(prompt)
            from claude_agent_sdk import ResultMessage
            result = MagicMock(spec=ResultMessage)
            result.is_error = False
            result.session_id = "sess"
            result.result = "Done"
            result.total_cost_usd = 0.01
            result.duration_ms = 100
            result.num_turns = 1
            yield result

        with patch("ralph_tui.orchestrator.query", side_effect=mock_query), \
             patch("ralph_tui.orchestrator.analyze_output", new_callable=AsyncMock) as mock_analyze:
            mock_analyze.return_value = MagicMock(
                should_stop=False, reason="continue", summary="continue"
            )
            await orch.run()

        assert len(prompts_used) == 3
        assert prompts_used[0].startswith("Start here")
        assert prompts_used[1].startswith("Continue here")
        assert prompts_used[2].startswith("Continue here")

    @pytest.mark.asyncio
    async def test_original_project_is_never_modified(self, tmp_path):
        """The original project directory should never be modified, only copies."""
        project = tmp_path / "myproject"
        project.mkdir()
        (project / "original.txt").write_text("do not modify")

        cfg = _make_e2e_config(str(project))
        orch = Orchestrator(cfg)

        async def mock_query(prompt, options):
            cwd = Path(options.cwd)
            # "Claude" modifies files in the working copy
            (cwd / "original.txt").write_text("MODIFIED BY CLAUDE")
            (cwd / "new_file.txt").write_text("new content")

            from claude_agent_sdk import ResultMessage
            result = MagicMock(spec=ResultMessage)
            result.is_error = False
            result.session_id = "sess"
            result.result = "Done"
            result.total_cost_usd = 0.01
            result.duration_ms = 100
            result.num_turns = 1
            yield result

        with patch("ralph_tui.orchestrator.query", side_effect=mock_query), \
             patch("ralph_tui.orchestrator.analyze_output", new_callable=AsyncMock) as mock_analyze:
            mock_analyze.return_value = MagicMock(
                should_stop=False, reason="continue", summary="continue"
            )
            await orch.run()

        # Original project should be untouched
        assert (project / "original.txt").read_text() == "do not modify"
        assert not (project / "new_file.txt").exists()
