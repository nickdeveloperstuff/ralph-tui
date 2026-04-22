"""Tests for effective iteration counting, escalating backoff, consecutive stops,
verification directory management, and context recovery."""

import asyncio
import json
import shutil
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ralph_tui.config import RalphConfig, CONTEXT_RECOVERY_SUFFIX
from ralph_tui.orchestrator import Orchestrator, _copy_project


def _make_config(tmp_path: Path, **overrides) -> RalphConfig:
    """Return a valid RalphConfig pointing at a temp project dir."""
    project = tmp_path / "project"
    project.mkdir(exist_ok=True)
    (project / "main.py").write_text("print('hello')")
    defaults = dict(
        project_path=str(project),
        initial_prompt="Do something",
        rerun_prompt="Do more",
        min_iterations=1,
        max_iterations=3,
    )
    defaults.update(overrides)
    return RalphConfig(**defaults)


def _success_query_factory(cost=0.01):
    """Create a mock query that succeeds with given cost."""
    async def mock_query(prompt, options):
        from claude_agent_sdk import ResultMessage
        result = MagicMock(spec=ResultMessage)
        result.is_error = False
        result.session_id = "sess"
        result.result = "Done"
        result.total_cost_usd = cost
        result.duration_ms = 100
        result.num_turns = 1
        yield result
    return mock_query


def _error_query_factory(cost=0.0):
    """Create a mock query that always fails with a ProcessError."""
    async def mock_query(prompt, options):
        from claude_agent_sdk import ProcessError
        raise ProcessError("Command failed with exit code 1", exit_code=1)
    return mock_query


def _mixed_query_factory(fail_on: set[int], cost=0.01):
    """Create a mock query that fails on specified call numbers (1-indexed)."""
    call_count = {"n": 0}

    async def mock_query(prompt, options):
        call_count["n"] += 1
        if call_count["n"] in fail_on:
            from claude_agent_sdk import ProcessError
            raise ProcessError("Command failed with exit code 1", exit_code=1)
        from claude_agent_sdk import ResultMessage
        result = MagicMock(spec=ResultMessage)
        result.is_error = False
        result.session_id = "sess"
        result.result = "Done"
        result.total_cost_usd = cost
        result.duration_ms = 100
        result.num_turns = 1
        yield result
    return mock_query


class TestEffectiveIterationCounting:
    """Tests for effective iteration counting."""

    @pytest.mark.asyncio
    async def test_failed_iterations_not_counted(self, tmp_path):
        """Failed iterations (cost=0, error) should not count as effective."""
        cfg = _make_config(tmp_path, max_iterations=3, max_error_retries=0, max_consecutive_errors=3)
        orch = Orchestrator(cfg)

        # Fail calls 1 and 2 (raw iters 1 and 2), succeed on 3,4,5
        mock_query = _mixed_query_factory(fail_on={1, 2})

        with patch("ralph_tui.orchestrator.query", side_effect=mock_query), \
             patch("ralph_tui.orchestrator.asyncio.sleep", new_callable=AsyncMock), \
             patch("ralph_tui.orchestrator.analyze_output", new_callable=AsyncMock) as mock_analyze:
            mock_analyze.return_value = MagicMock(should_stop=False, reason="continue", summary="continue")
            state = await orch.run()

        assert state.effective_iterations == 3
        # Total raw iterations should be 5 (2 failed + 3 effective)
        assert len(state.results) == 5
        # First two results should be ineffective
        assert state.results[0].is_effective is False
        assert state.results[1].is_effective is False
        assert state.results[2].is_effective is True
        assert state.results[2].effective_iteration == 1

    @pytest.mark.asyncio
    async def test_prompt_selection_uses_effective_iteration(self, tmp_path):
        """Prompt selection should use effective iteration count, not raw."""
        cfg = _make_config(
            tmp_path, max_iterations=3, max_error_retries=0,
            max_consecutive_errors=5,
            final_prompt="Wrap it up",
            transition_iteration=3,
        )
        orch = Orchestrator(cfg)
        prompts_used = []

        call_count = {"n": 0}

        async def mock_query(prompt, options):
            call_count["n"] += 1
            prompts_used.append(prompt)
            if call_count["n"] == 2:
                # Fail on raw iteration 2
                from claude_agent_sdk import ProcessError
                raise ProcessError("fail", exit_code=1)
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
             patch("ralph_tui.orchestrator.asyncio.sleep", new_callable=AsyncMock), \
             patch("ralph_tui.orchestrator.analyze_output", new_callable=AsyncMock) as mock_analyze:
            mock_analyze.return_value = MagicMock(should_stop=False, reason="continue", summary="continue")
            state = await orch.run()

        # Raw iterations: 1(ok,eff1), 2(fail), 3(ok,eff2), 4(ok,eff3)
        # Effective iteration 1 → initial_prompt
        assert prompts_used[0].startswith("Do something")
        # Effective iteration 2 → rerun_prompt (transition at 3)
        # prompts_used[1] is the failed one — it tried for effective 2
        assert prompts_used[1].startswith("Do more")
        # After failure, next try is still for effective 2 → rerun
        assert prompts_used[2].startswith("Do more")
        # Effective iteration 3 → final_prompt (transition_iteration=3)
        assert prompts_used[3].startswith("Wrap it up")

    @pytest.mark.asyncio
    async def test_verification_scheduled_by_effective_count(self, tmp_path):
        """Verification should trigger on effective iteration N, not raw."""
        cfg = _make_config(
            tmp_path, max_iterations=4, max_error_retries=0,
            max_consecutive_errors=5,
            verification_prompt="Verify all citations",
            verification_interval=3,
        )
        orch = Orchestrator(cfg)
        prompts_used = []

        call_count = {"n": 0}

        async def mock_query(prompt, options):
            call_count["n"] += 1
            prompts_used.append(prompt)
            if call_count["n"] == 2:
                from claude_agent_sdk import ProcessError
                raise ProcessError("fail", exit_code=1)
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
             patch("ralph_tui.orchestrator.asyncio.sleep", new_callable=AsyncMock), \
             patch("ralph_tui.orchestrator.analyze_output", new_callable=AsyncMock) as mock_analyze:
            mock_analyze.return_value = MagicMock(should_stop=False, reason="continue", summary="continue")
            state = await orch.run()

        # Effective 3 should be verification (interval=3, effective_iter > 1, mod 3 == 0)
        # Find which prompt was the verification one
        verification_prompts = [p for p in prompts_used if "BLIND verification" in p]
        assert len(verification_prompts) == 1

    @pytest.mark.asyncio
    async def test_min_iterations_uses_effective_count(self, tmp_path):
        """Analysis should be skipped until effective count >= min_iterations."""
        cfg = _make_config(
            tmp_path, min_iterations=2, max_iterations=3,
            max_error_retries=0, max_consecutive_errors=5,
        )
        orch = Orchestrator(cfg)

        mock_query = _mixed_query_factory(fail_on={1})

        with patch("ralph_tui.orchestrator.query", side_effect=mock_query), \
             patch("ralph_tui.orchestrator.asyncio.sleep", new_callable=AsyncMock), \
             patch("ralph_tui.orchestrator.analyze_output", new_callable=AsyncMock) as mock_analyze:
            mock_analyze.return_value = MagicMock(should_stop=True, reason="done", summary="done")
            state = await orch.run()

        # Raw 1 fails (ineffective). Raw 2 succeeds (effective 1, < min 2 → skip).
        # Raw 3 succeeds (effective 2 → analyze). Raw 4 (effective 3 → 2nd stop) → break.
        analyzed_results = [r for r in state.results if r.analysis is not None]
        skipped_effective = [r for r in state.results if r.is_effective and r.skipped_analysis]
        assert len(skipped_effective) == 1  # effective iter 1 skipped analysis


class TestEscalatingBackoff:
    """Tests for escalating backoff schedule."""

    def test_backoff_schedule(self, tmp_path):
        """Verify wait times: 60s, 60s, 300s, 600s, 900s."""
        cfg = _make_config(tmp_path)
        orch = Orchestrator(cfg)
        assert orch._get_backoff_wait(1) == 60
        assert orch._get_backoff_wait(2) == 60
        assert orch._get_backoff_wait(3) == 300
        assert orch._get_backoff_wait(4) == 600
        assert orch._get_backoff_wait(5) == 900
        assert orch._get_backoff_wait(10) == 900

    @pytest.mark.asyncio
    async def test_circuit_breaker_at_max(self, tmp_path):
        """Circuit breaker fires at max_consecutive_errors."""
        cfg = _make_config(
            tmp_path, max_iterations=10,
            max_error_retries=0, max_consecutive_errors=3,
        )
        orch = Orchestrator(cfg)
        mock_query = _error_query_factory()

        with patch("ralph_tui.orchestrator.query", side_effect=mock_query), \
             patch("ralph_tui.orchestrator.asyncio.sleep", new_callable=AsyncMock):
            state = await orch.run()

        assert state.consecutive_errors == 3
        assert "Circuit breaker" in state.status

    @pytest.mark.asyncio
    async def test_consecutive_errors_reset_on_success(self, tmp_path):
        """Successful iteration resets consecutive_errors counter."""
        cfg = _make_config(
            tmp_path, max_iterations=3,
            max_error_retries=0, max_consecutive_errors=5,
        )
        orch = Orchestrator(cfg)
        # Fail on calls 1, 2; succeed on 3, 4, 5; fail on 6, 7; succeed on 8
        mock_query = _mixed_query_factory(fail_on={1, 2, 6, 7})

        with patch("ralph_tui.orchestrator.query", side_effect=mock_query), \
             patch("ralph_tui.orchestrator.asyncio.sleep", new_callable=AsyncMock), \
             patch("ralph_tui.orchestrator.analyze_output", new_callable=AsyncMock) as mock_analyze:
            mock_analyze.return_value = MagicMock(should_stop=False, reason="continue", summary="continue")
            state = await orch.run()

        # Should complete 3 effective iterations despite errors
        assert state.effective_iterations == 3
        assert state.consecutive_errors == 0  # reset after last success


class TestTwoConsecutiveStops:
    """Tests for 2-consecutive-stops end criteria."""

    @pytest.mark.asyncio
    async def test_two_consecutive_stops_required(self, tmp_path):
        """Loop should only break after 2 consecutive should_stop=True."""
        cfg = _make_config(tmp_path, max_iterations=10)
        orch = Orchestrator(cfg)
        mock_query = _success_query_factory()

        with patch("ralph_tui.orchestrator.query", side_effect=mock_query), \
             patch("ralph_tui.orchestrator.analyze_output", new_callable=AsyncMock) as mock_analyze:
            mock_analyze.return_value = MagicMock(should_stop=True, reason="done", summary="done")
            state = await orch.run()

        assert len(state.results) == 2
        assert state.consecutive_stops == 2

    @pytest.mark.asyncio
    async def test_single_stop_continues(self, tmp_path):
        """should_stop=True then False should reset counter and continue."""
        cfg = _make_config(tmp_path, max_iterations=4)
        orch = Orchestrator(cfg)
        mock_query = _success_query_factory()

        analyze_call_count = {"n": 0}

        async def mock_analyze(text, sys_prompt, exit_prompt, iteration_context=None):
            analyze_call_count["n"] += 1
            # Stop on call 1, continue on call 2, stop on calls 3 and 4
            if analyze_call_count["n"] == 1:
                return MagicMock(should_stop=True, reason="maybe done", summary="maybe")
            elif analyze_call_count["n"] == 2:
                return MagicMock(should_stop=False, reason="more work", summary="continue")
            else:
                return MagicMock(should_stop=True, reason="done", summary="done")

        with patch("ralph_tui.orchestrator.query", side_effect=mock_query), \
             patch("ralph_tui.orchestrator.analyze_output", side_effect=mock_analyze):
            state = await orch.run()

        # Effective 1: stop (1/2). Effective 2: no stop → reset to 0.
        # Effective 3: stop (1/2). Effective 4: stop (2/2) → break.
        assert len(state.results) == 4
        assert state.consecutive_stops == 2

    @pytest.mark.asyncio
    async def test_verification_no_affect_consecutive_stops(self, tmp_path):
        """Verification iterations should not affect consecutive_stops counter."""
        cfg = _make_config(
            tmp_path, max_iterations=6, min_iterations=1,
            verification_prompt="Verify citations",
            verification_interval=2,
        )
        orch = Orchestrator(cfg)
        mock_query = _success_query_factory()

        async def mock_analyze(text, sys_prompt, exit_prompt, iteration_context=None):
            return MagicMock(should_stop=True, reason="done", summary="done")

        with patch("ralph_tui.orchestrator.query", side_effect=mock_query), \
             patch("ralph_tui.orchestrator.analyze_output", side_effect=mock_analyze):
            state = await orch.run()

        # Eff 1: non-verify, stop (1/2). Eff 2: verify, stop overridden, no counter change.
        # Eff 3: non-verify, stop (2/2) → break.
        assert len(state.results) == 3
        assert state.consecutive_stops == 2


class TestVerificationDirectory:
    """Tests for blind verification directory preparation and restoration."""

    def test_prepare_verification_dir(self, tmp_path):
        """Main doc should be moved to hidden, manifest created, claims redacted."""
        iter_dir = tmp_path / "iter"
        iter_dir.mkdir()
        assignment = iter_dir / "assignment"
        assignment.mkdir()
        (assignment / "case1.md").write_text("Case content")
        (iter_dir / "Brief.MD").write_text("Legal brief content")
        (iter_dir / "_scratch_notes.md").write_text("My notes")
        state = {
            "iteration": 1,
            "tasks": [],
            "citations_to_verify": [
                {"citation": "Smith v Jones", "claim": "Ruled in favor of plaintiff", "status": "unverified"},
                {"citation": "42 USC 1983", "claim": "Civil rights statute", "status": "unverified"},
            ],
            "key_findings": [],
        }
        (iter_dir / "_ralph_state.json").write_text(json.dumps(state))
        (iter_dir / "CLAUDE.md").write_text("# Template")

        cfg = _make_config(tmp_path)
        orch = Orchestrator(cfg)
        orch._prepare_verification_dir(iter_dir)

        hidden = iter_dir / "._ralph_hidden"
        # Main doc moved
        assert not (iter_dir / "Brief.MD").exists()
        assert (hidden / "Brief.MD").exists()
        # Scratch moved
        assert not (iter_dir / "_scratch_notes.md").exists()
        assert (hidden / "_scratch_notes.md").exists()
        # Claims file
        claims = json.loads((hidden / "_claims.json").read_text())
        assert len(claims) == 2
        assert claims[0]["claim"] == "Ruled in favor of plaintiff"
        # Manifest (no claims)
        manifest = json.loads((iter_dir / "_verification_manifest.json").read_text())
        assert len(manifest) == 2
        assert "claim" not in manifest[0]
        assert manifest[0]["citation"] == "Smith v Jones"
        # State file redacted
        redacted = json.loads((iter_dir / "_ralph_state.json").read_text())
        for c in redacted["citations_to_verify"]:
            assert "claim" not in c

    def test_restore_from_hidden(self, tmp_path):
        """Files should be restored from hidden dir back to root."""
        iter_dir = tmp_path / "iter"
        iter_dir.mkdir()
        hidden = iter_dir / "._ralph_hidden"
        hidden.mkdir()
        (hidden / "Brief.MD").write_text("Legal brief")
        (hidden / "_scratch_notes.md").write_text("Notes")
        (hidden / "_claims.json").write_text("[]")
        (iter_dir / "_verification_manifest.json").write_text("[]")

        cfg = _make_config(tmp_path)
        orch = Orchestrator(cfg)
        orch._restore_from_hidden(iter_dir)

        assert (iter_dir / "Brief.MD").exists()
        assert (iter_dir / "_scratch_notes.md").exists()
        assert (iter_dir / "_claims.json").exists()
        assert not hidden.exists()
        assert not (iter_dir / "_verification_manifest.json").exists()

    def test_verification_feedback_generation(self, tmp_path):
        """Disputed citations should be formatted as prompt prefix."""
        iter_dir = tmp_path / "iter"
        iter_dir.mkdir()
        state = {
            "citations_to_verify": [
                {"citation": "Smith v Jones", "status": "verified"},
                {"citation": "Doe v Roe", "status": "disputed", "discrepancy": "Date was wrong"},
                {"citation": "42 USC 1983", "status": "unable_to_verify", "reason": "Source not found"},
            ],
        }
        (iter_dir / "_ralph_state.json").write_text(json.dumps(state))

        cfg = _make_config(tmp_path)
        orch = Orchestrator(cfg)
        feedback = orch._build_verification_feedback(iter_dir)

        assert feedback is not None
        assert "VERIFICATION FINDINGS" in feedback
        assert "Doe v Roe" in feedback
        assert "DISPUTED" in feedback
        assert "Date was wrong" in feedback
        assert "42 USC 1983" in feedback
        assert "UNABLE TO VERIFY" in feedback
        assert "Smith v Jones" not in feedback  # verified — no issue

    def test_verification_feedback_none_when_all_verified(self, tmp_path):
        """No feedback when all citations are verified."""
        iter_dir = tmp_path / "iter"
        iter_dir.mkdir()
        state = {
            "citations_to_verify": [
                {"citation": "Smith v Jones", "status": "verified"},
            ],
        }
        (iter_dir / "_ralph_state.json").write_text(json.dumps(state))

        cfg = _make_config(tmp_path)
        orch = Orchestrator(cfg)
        assert orch._build_verification_feedback(iter_dir) is None


class TestContextRecoverySuffix:
    """Tests for context recovery suffix injection."""

    @pytest.mark.asyncio
    async def test_context_recovery_suffix_injected(self, tmp_path):
        """After 3+ consecutive errors, context recovery suffix should appear in prompt."""
        cfg = _make_config(
            tmp_path, max_iterations=2,
            max_error_retries=0, max_consecutive_errors=5,
        )
        orch = Orchestrator(cfg)
        prompts_used = []

        call_count = {"n": 0}

        async def mock_query(prompt, options):
            call_count["n"] += 1
            prompts_used.append(prompt)
            # Fail first 3, then succeed
            if call_count["n"] <= 3:
                from claude_agent_sdk import ProcessError
                raise ProcessError("fail", exit_code=1)
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
             patch("ralph_tui.orchestrator.asyncio.sleep", new_callable=AsyncMock), \
             patch("ralph_tui.orchestrator.analyze_output", new_callable=AsyncMock) as mock_analyze:
            mock_analyze.return_value = MagicMock(should_stop=False, reason="continue", summary="continue")
            state = await orch.run()

        # After 3 consecutive errors, the 4th call should have recovery suffix
        assert "RECOVERY MODE" in prompts_used[3]

    @pytest.mark.asyncio
    async def test_context_recovery_suffix_cleared_on_success(self, tmp_path):
        """After a successful iteration, recovery suffix should not appear."""
        cfg = _make_config(
            tmp_path, max_iterations=3,
            max_error_retries=0, max_consecutive_errors=5,
        )
        orch = Orchestrator(cfg)
        prompts_used = []

        call_count = {"n": 0}

        async def mock_query(prompt, options):
            call_count["n"] += 1
            prompts_used.append(prompt)
            # Fail 1-3, succeed 4+
            if call_count["n"] <= 3:
                from claude_agent_sdk import ProcessError
                raise ProcessError("fail", exit_code=1)
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
             patch("ralph_tui.orchestrator.asyncio.sleep", new_callable=AsyncMock), \
             patch("ralph_tui.orchestrator.analyze_output", new_callable=AsyncMock) as mock_analyze:
            mock_analyze.return_value = MagicMock(should_stop=False, reason="continue", summary="continue")
            state = await orch.run()

        # Call 4 has recovery suffix (consecutive_errors was 3)
        assert "RECOVERY MODE" in prompts_used[3]
        # Call 5 should NOT have recovery suffix (consecutive_errors reset to 0)
        assert "RECOVERY MODE" not in prompts_used[4]


class TestIsEffective:
    """Tests for the _is_effective helper."""

    def test_effective_with_cost_and_no_error(self, tmp_path):
        cfg = _make_config(tmp_path)
        orch = Orchestrator(cfg)
        assert orch._is_effective(0.50, None) is True

    def test_not_effective_zero_cost(self, tmp_path):
        cfg = _make_config(tmp_path)
        orch = Orchestrator(cfg)
        assert orch._is_effective(0.0, None) is False

    def test_not_effective_with_error(self, tmp_path):
        cfg = _make_config(tmp_path)
        orch = Orchestrator(cfg)
        assert orch._is_effective(0.50, "process_error") is False

    def test_not_effective_zero_cost_with_error(self, tmp_path):
        cfg = _make_config(tmp_path)
        orch = Orchestrator(cfg)
        assert orch._is_effective(0.0, "unknown") is False
