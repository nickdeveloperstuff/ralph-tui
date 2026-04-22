"""Full-simulation E2E tests for the orchestrator.

These tests go beyond behavioral checks — the mock_query callbacks inspect and
assert against the **real filesystem** during execution (iteration dirs get
cleaned up later, so assertions must happen while Claude would see them).
"""

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ralph_tui.config import (
    RalphConfig,
    CLAUDE_MD_VERIFICATION_TEMPLATE,
    CONTEXT_RECOVERY_SUFFIX,
    VERIFICATION_METHODOLOGY_TEMPLATE,
    CONTEXT_MANAGEMENT_SUFFIX,
)
from ralph_tui.orchestrator import Orchestrator


def _setup_test_project(tmp_path: Path) -> Path:
    """Create a test project with legal documents and source materials."""
    project = tmp_path / "test_project"
    project.mkdir()

    # Main legal document
    (project / "Brief.MD").write_text("""\
# Legal Brief

## Introduction
This brief addresses the claims in Smith v. Jones, 456 U.S. 123 (2024).

## Argument
According to the deposition transcript (Exhibit A), the defendant admitted fault.
Furthermore, 42 U.S.C. § 1983 provides the statutory basis for this action.

## Conclusion
The evidence clearly supports the plaintiff's position.
""")

    # Source materials
    assignment = project / "assignment"
    assignment.mkdir()
    (assignment / "case1.md").write_text("""\
# Smith v. Jones, 456 U.S. 123 (2024)

The Supreme Court held that employers have a duty of care when supervising
remote workers. The ruling was 5-4 in favor of the plaintiff Smith.

Key holding: Employers must provide reasonable accommodations for remote work.
""")

    (assignment / "case2.md").write_text("""\
# 42 U.S.C. § 1983 - Civil Rights Act

Every person who, under color of any statute, ordinance, regulation, custom,
or usage, subjects any citizen to the deprivation of any rights, privileges,
or immunities secured by the Constitution and laws, shall be liable to the
party injured in an action at law.
""")

    (assignment / "exhibit_a.md").write_text("""\
# Exhibit A - Deposition Transcript

Q: Did you acknowledge responsibility for the incident?
A: I understood that our company procedures were not followed.

Q: Were safety protocols in place?
A: The protocols existed but were not consistently enforced.
""")

    return project


def _make_e2e_config(project_path: Path, **overrides) -> RalphConfig:
    """Create a config for E2E testing."""
    defaults = dict(
        project_path=str(project_path),
        initial_prompt="Review the brief and begin drafting improvements. Focus on strengthening citations.",
        rerun_prompt="Continue refining the brief. Address any weak citations.",
        final_prompt="Final review pass. Ensure all citations are properly supported.",
        min_iterations=2,
        max_iterations=6,
        transition_iteration=4,
        verification_prompt="Verify all legal citations against source materials",
        verification_interval=3,
        max_consecutive_errors=5,
        max_error_retries=2,
    )
    defaults.update(overrides)
    return RalphConfig(**defaults)


def _make_mock_result(session_id: str, text: str, cost: float = 0.50):
    """Create a mock ResultMessage."""
    from claude_agent_sdk import ResultMessage
    result = MagicMock(spec=ResultMessage)
    result.is_error = False
    result.session_id = session_id
    result.result = text
    result.total_cost_usd = cost
    result.duration_ms = 5000
    result.num_turns = 10
    return result


class TestFullSimulationNormalFlow:
    """Full simulation with filesystem artifact verification."""

    @pytest.mark.asyncio
    async def test_full_simulation_normal_flow(self, tmp_path):
        """Run 5 effective iterations with verification at #3 and transition at #4.

        The mock_query inspects the real filesystem at each iteration to verify:
        - CLAUDE.md template switching (standard vs verification)
        - Brief.MD presence/absence during verification
        - ._ralph_hidden/ structure during verification
        - _verification_manifest.json contents (no claims)
        - _ralph_state.json redaction during verification
        - Restoration of files post-verification
        - _document_index.md generation
        """
        project = _setup_test_project(tmp_path)
        cfg = _make_e2e_config(project, max_iterations=5)
        orch = Orchestrator(cfg)

        effective_count = {"n": 0}
        captured_checks = []
        prompts_used = []
        analyze_contexts = []

        async def mock_query(prompt, options):
            effective_count["n"] += 1
            eff_n = effective_count["n"]
            cwd = Path(options.cwd)
            prompts_used.append(prompt)
            checks = {"effective": eff_n, "checks": {}}

            claude_md = cwd / "CLAUDE.md"
            brief = cwd / "Brief.MD"
            hidden = cwd / "._ralph_hidden"
            manifest = cwd / "_verification_manifest.json"
            state_file = cwd / "_ralph_state.json"
            doc_index = cwd / "_document_index.md"
            assignment_dir = cwd / "assignment"

            if eff_n == 1:
                # EFFECTIVE 1 (initial): standard CLAUDE.md, Brief.MD present, doc index exists
                checks["checks"]["claude_md_is_standard"] = (
                    claude_md.exists() and "Iteration Guidelines" in claude_md.read_text()
                )
                checks["checks"]["brief_exists"] = brief.exists()
                checks["checks"]["doc_index_exists"] = doc_index.exists()
                checks["checks"]["hidden_not_exists"] = not hidden.exists()
                checks["checks"]["manifest_not_exists"] = not manifest.exists()

                # ACTION: edit Brief.MD and write citations to state
                content = brief.read_text()
                brief.write_text(content + "\n<!-- Effective 1 edit -->\n")
                state = json.loads(state_file.read_text())
                state["citations_to_verify"] = [
                    {"citation": "Smith v. Jones, 456 U.S. 123", "claim": "Ruled 5-4 for plaintiff", "status": "unverified"},
                    {"citation": "42 U.S.C. § 1983", "claim": "Civil rights statute basis", "status": "unverified"},
                ]
                state["tasks"] = [{"name": "Review citations", "status": "completed"}]
                state_file.write_text(json.dumps(state, indent=2))
                result_text = "Drafted improvements, added citations."

            elif eff_n == 2:
                # EFFECTIVE 2 (rerun): standard CLAUDE.md, Brief has edits from eff 1
                checks["checks"]["claude_md_is_standard"] = (
                    claude_md.exists() and "Iteration Guidelines" in claude_md.read_text()
                )
                checks["checks"]["brief_has_eff1_edit"] = (
                    brief.exists() and "Effective 1 edit" in brief.read_text()
                )
                checks["checks"]["hidden_not_exists"] = not hidden.exists()

                # ACTION: more edits
                brief.write_text(brief.read_text() + "\n<!-- Effective 2 edit -->\n")
                result_text = "Continued refining brief."

            elif eff_n == 3:
                # EFFECTIVE 3 (VERIFICATION): verification CLAUDE.md, Brief.MD HIDDEN
                checks["checks"]["claude_md_is_verification"] = (
                    claude_md.exists() and "VERIFICATION Iteration (BLIND)" in claude_md.read_text()
                )
                checks["checks"]["brief_NOT_exists"] = not brief.exists()
                checks["checks"]["hidden_exists"] = hidden.exists()
                checks["checks"]["hidden_has_brief"] = (hidden / "Brief.MD").exists() if hidden.exists() else False
                checks["checks"]["hidden_has_claims"] = (hidden / "_claims.json").exists() if hidden.exists() else False
                checks["checks"]["manifest_exists"] = manifest.exists()
                checks["checks"]["assignment_accessible"] = assignment_dir.exists()

                # Verify manifest has NO "claim" field
                if manifest.exists():
                    manifest_data = json.loads(manifest.read_text())
                    checks["checks"]["manifest_no_claims"] = all(
                        "claim" not in entry for entry in manifest_data
                    )
                    checks["checks"]["manifest_has_citations"] = all(
                        "citation" in entry for entry in manifest_data
                    )
                else:
                    checks["checks"]["manifest_no_claims"] = False
                    checks["checks"]["manifest_has_citations"] = False

                # Verify _ralph_state.json is redacted (no claim field on citations)
                if state_file.exists():
                    state = json.loads(state_file.read_text())
                    citations = state.get("citations_to_verify", [])
                    checks["checks"]["state_redacted"] = all(
                        "claim" not in c for c in citations if isinstance(c, dict)
                    )
                else:
                    checks["checks"]["state_redacted"] = False

                # Verify _claims.json has claims
                if hidden.exists() and (hidden / "_claims.json").exists():
                    claims_data = json.loads((hidden / "_claims.json").read_text())
                    checks["checks"]["claims_have_claims"] = all(
                        "claim" in entry and entry["claim"] for entry in claims_data
                    )
                else:
                    checks["checks"]["claims_have_claims"] = False

                # ACTION: write verification report, update statuses
                (cwd / "_verification_report.md").write_text(
                    "# Verification Report\n\n- 2 citations verified\n- 0 disputed\n"
                )
                if state_file.exists():
                    state = json.loads(state_file.read_text())
                    for c in state.get("citations_to_verify", []):
                        c["status"] = "verified"
                    state_file.write_text(json.dumps(state, indent=2))
                result_text = "Verified all citations. 2 verified, 0 disputed."

            elif eff_n == 4:
                # EFFECTIVE 4 (final, transition_iteration=4): Brief.MD RESTORED
                checks["checks"]["brief_exists_restored"] = brief.exists()
                checks["checks"]["hidden_not_exists"] = not hidden.exists()
                checks["checks"]["manifest_not_exists"] = not manifest.exists()
                checks["checks"]["claude_md_is_standard"] = (
                    claude_md.exists() and "Iteration Guidelines" in claude_md.read_text()
                )
                # Brief should have prior edits
                if brief.exists():
                    checks["checks"]["brief_has_prior_edits"] = "Effective 2 edit" in brief.read_text()
                else:
                    checks["checks"]["brief_has_prior_edits"] = False

                # ACTION: final edits
                if brief.exists():
                    brief.write_text(brief.read_text() + "\n<!-- Effective 4 edit -->\n")
                result_text = "Final review edits applied."

            else:
                # EFFECTIVE 5 (final): standard template, no hidden
                checks["checks"]["claude_md_is_standard"] = (
                    claude_md.exists() and "Iteration Guidelines" in claude_md.read_text()
                )
                checks["checks"]["hidden_not_exists"] = not hidden.exists()
                if brief.exists():
                    brief.write_text(brief.read_text() + "\n<!-- Effective 5 edit -->\n")
                result_text = "More final edits."

            captured_checks.append(checks)
            yield _make_mock_result(f"sess-{eff_n}", result_text)

        async def mock_analyze(text, sys_prompt, exit_prompt, iteration_context=None):
            if iteration_context:
                analyze_contexts.append(iteration_context)
            return MagicMock(should_stop=False, reason="more work", summary="continuing")

        with patch("ralph_tui.orchestrator.query", side_effect=mock_query), \
             patch("ralph_tui.orchestrator.analyze_output", side_effect=mock_analyze):
            state = await orch.run()

        # --- Post-run assertions ---

        # 1. Effective iteration count
        assert state.effective_iterations == 5

        # 2. All filesystem checks passed
        for check_group in captured_checks:
            eff = check_group["effective"]
            for name, passed in check_group["checks"].items():
                assert passed, f"Effective {eff}: check '{name}' FAILED"

        # 3. Log file correctness
        runs_dir = Path(cfg.project_path).resolve().parent / f"{Path(cfg.project_path).resolve().name}-ralph-runs"
        log_file = runs_dir / "ralph-log.jsonl"
        assert log_file.exists()

        log_entries = []
        for line in log_file.read_text().strip().split("\n"):
            if line.strip():
                try:
                    entry = json.loads(line)
                    if "effective_iteration" in entry:
                        log_entries.append(entry)
                except json.JSONDecodeError:
                    pass

        assert len(log_entries) == 5
        assert [e["effective_iteration"] for e in log_entries] == [1, 2, 3, 4, 5]
        assert all(e["is_effective"] for e in log_entries)

        # 4. Prompt selection sequence
        # Eff 1: initial, Eff 2: rerun, Eff 3: verification methodology,
        # Eff 4-5: final
        assert prompts_used[0].startswith("Review the brief")  # initial
        assert prompts_used[1].startswith("Continue refining")  # rerun
        assert "BLIND verification" in prompts_used[2]  # verification methodology
        assert prompts_used[3].startswith("Final review")  # final (transition=4)
        assert prompts_used[4].startswith("Final review")

        # 5. Analyzer received correct iteration_context
        # Analysis skipped for eff 1 (below min_iterations=2), so first context is eff 2
        assert len(analyze_contexts) >= 3  # eff 2,3,4,5 (4 analyzed, 1 skipped)
        # First analyzed is effective 2
        assert analyze_contexts[0]["iteration"] == 2
        assert analyze_contexts[0]["phase"] == "rerun"
        # Verification iteration
        verif_ctx = [c for c in analyze_contexts if c["is_verification"]]
        assert len(verif_ctx) == 1
        assert verif_ctx[0]["iteration"] == 3
        assert verif_ctx[0]["phase"] == "verification"
        # Final phase iterations
        final_ctxs = [c for c in analyze_contexts if c["phase"] == "final"]
        assert len(final_ctxs) >= 1
        assert all(c["iteration"] >= 4 for c in final_ctxs)

        # 6. Results match expectations
        assert len(state.results) == 5
        assert all(r.is_effective for r in state.results)
        assert state.results[0].effective_iteration == 1
        assert state.results[2].effective_iteration == 3  # verification


class TestFullSimulationVerificationFeedback:
    """Verification feedback loop: disputed citations get fed back to next drafting prompt."""

    @pytest.mark.asyncio
    async def test_full_simulation_verification_feedback_loop(self, tmp_path):
        """After verification with DISPUTED citations, next drafting prompt has feedback."""
        project = _setup_test_project(tmp_path)
        cfg = _make_e2e_config(
            project, max_iterations=5, min_iterations=1,
            verification_interval=3, transition_iteration=0,
        )
        orch = Orchestrator(cfg)

        effective_count = {"n": 0}
        prompts_used = []
        captured_checks = []

        async def mock_query(prompt, options):
            effective_count["n"] += 1
            eff_n = effective_count["n"]
            cwd = Path(options.cwd)
            prompts_used.append(prompt)
            checks = {"effective": eff_n, "checks": {}}

            state_file = cwd / "_ralph_state.json"
            brief = cwd / "Brief.MD"

            if eff_n == 1:
                # Write citations
                state = json.loads(state_file.read_text())
                state["citations_to_verify"] = [
                    {"citation": "Smith v. Jones", "claim": "Ruled for plaintiff", "status": "unverified"},
                    {"citation": "Doe v. Roe", "claim": "Damages awarded $1M", "status": "unverified"},
                ]
                state_file.write_text(json.dumps(state, indent=2))
                if brief.exists():
                    brief.write_text(brief.read_text() + "\n<!-- eff1 -->")
                result_text = "Added citations."

            elif eff_n == 2:
                if brief.exists():
                    brief.write_text(brief.read_text() + "\n<!-- eff2 -->")
                result_text = "More work."

            elif eff_n == 3:
                # VERIFICATION — mark one citation as DISPUTED
                if state_file.exists():
                    state = json.loads(state_file.read_text())
                    for c in state.get("citations_to_verify", []):
                        if c.get("citation") == "Smith v. Jones":
                            c["status"] = "verified"
                        elif c.get("citation") == "Doe v. Roe":
                            c["status"] = "disputed"
                            c["discrepancy"] = "Source says $500K not $1M"
                    state_file.write_text(json.dumps(state, indent=2))
                (cwd / "_verification_report.md").write_text(
                    "# Report\n1 verified, 1 disputed\n"
                )
                result_text = "Verification complete."

            elif eff_n == 4:
                # Post-verification drafting — should have VERIFICATION FINDINGS in prompt
                checks["checks"]["has_verification_feedback"] = (
                    "VERIFICATION FINDINGS" in prompt
                )
                checks["checks"]["feedback_mentions_disputed"] = (
                    "DISPUTED" in prompt and "Doe v. Roe" in prompt
                )
                checks["checks"]["feedback_mentions_discrepancy"] = (
                    "$500K not $1M" in prompt
                )
                if brief.exists():
                    brief.write_text(brief.read_text() + "\n<!-- eff4 -->")
                result_text = "Addressed verification feedback."

            else:
                # eff_n == 5: feedback should be CONSUMED (not repeated)
                checks["checks"]["no_verification_feedback"] = (
                    "VERIFICATION FINDINGS" not in prompt
                )
                if brief.exists():
                    brief.write_text(brief.read_text() + "\n<!-- eff5 -->")
                result_text = "Final work."

            captured_checks.append(checks)
            yield _make_mock_result(f"sess-{eff_n}", result_text)

        async def mock_analyze(text, sys_prompt, exit_prompt, iteration_context=None):
            return MagicMock(should_stop=False, reason="continue", summary="continue")

        with patch("ralph_tui.orchestrator.query", side_effect=mock_query), \
             patch("ralph_tui.orchestrator.analyze_output", side_effect=mock_analyze):
            state = await orch.run()

        assert state.effective_iterations == 5

        # Verify all filesystem checks
        for check_group in captured_checks:
            eff = check_group["effective"]
            for name, passed in check_group["checks"].items():
                assert passed, f"Effective {eff}: check '{name}' FAILED"

        # Verify prompt sequence
        assert prompts_used[0].startswith("Review the brief")  # initial
        assert prompts_used[1].startswith("Continue refining")  # rerun
        assert "BLIND verification" in prompts_used[2]  # verification
        # Eff 4 prompt starts with verification feedback
        assert "VERIFICATION FINDINGS" in prompts_used[3]
        # Eff 5 has no feedback
        assert "VERIFICATION FINDINGS" not in prompts_used[4]


class TestFullSimulationErrorRecovery:
    """Error recovery with filesystem artifact verification."""

    @pytest.mark.asyncio
    async def test_full_simulation_error_recovery_with_artifacts(self, tmp_path):
        """Errors don't consume iteration budget; context recovery suffix appears after 3+ errors."""
        project = _setup_test_project(tmp_path)
        cfg = _make_e2e_config(
            project, max_iterations=3, min_iterations=1,
            max_error_retries=0, max_consecutive_errors=5,
            verification_interval=0,  # disable verification
            transition_iteration=0,  # two-prompt mode
        )
        orch = Orchestrator(cfg)

        call_count = {"n": 0}
        prompts_used = []
        sleep_durations = []
        zombie_kill_count = {"n": 0}

        original_zombie_kill = orch._kill_zombie_claude_processes

        async def tracking_zombie_kill():
            zombie_kill_count["n"] += 1

        orch._kill_zombie_claude_processes = tracking_zombie_kill

        async def mock_query(prompt, options):
            call_count["n"] += 1
            prompts_used.append(prompt)

            # Fail calls 2, 3, 4 (after first success)
            if call_count["n"] in {2, 3, 4}:
                from claude_agent_sdk import ProcessError
                raise ProcessError("Command failed with exit code 1", exit_code=1)

            cwd = Path(options.cwd)
            brief = cwd / "Brief.MD"
            if brief.exists():
                brief.write_text(brief.read_text() + f"\n<!-- call {call_count['n']} -->")

            yield _make_mock_result(f"sess-{call_count['n']}", "Work done.")

        async def tracking_sleep(duration):
            sleep_durations.append(duration)

        with patch("ralph_tui.orchestrator.query", side_effect=mock_query), \
             patch("ralph_tui.orchestrator.asyncio.sleep", side_effect=tracking_sleep), \
             patch("ralph_tui.orchestrator.analyze_output", new_callable=AsyncMock) as mock_analyze:
            mock_analyze.return_value = MagicMock(should_stop=False, reason="continue", summary="continue")
            state = await orch.run()

        # 3 effective iterations despite 3 failures (6 raw total)
        assert state.effective_iterations == 3
        assert len(state.results) == 6

        effective_results = [r for r in state.results if r.is_effective]
        ineffective_results = [r for r in state.results if not r.is_effective]
        assert len(effective_results) == 3
        assert len(ineffective_results) == 3

        # Effective iterations have correct effective_iteration numbers
        assert [r.effective_iteration for r in effective_results] == [1, 2, 3]
        # Ineffective iterations have effective_iteration = 0
        assert all(r.effective_iteration == 0 for r in ineffective_results)

        # Log file verification
        runs_dir = Path(cfg.project_path).resolve().parent / f"{Path(cfg.project_path).resolve().name}-ralph-runs"
        log_file = runs_dir / "ralph-log.jsonl"
        log_entries = []
        for line in log_file.read_text().strip().split("\n"):
            try:
                entry = json.loads(line)
                if "effective_iteration" in entry:
                    log_entries.append(entry)
            except json.JSONDecodeError:
                pass

        assert len(log_entries) == 6
        eff_entries = [e for e in log_entries if e["is_effective"]]
        ineff_entries = [e for e in log_entries if not e["is_effective"]]
        assert len(eff_entries) == 3
        assert [e["effective_iteration"] for e in eff_entries] == [1, 2, 3]
        assert len(ineff_entries) == 3
        assert all(e["effective_iteration"] == 0 for e in ineff_entries)

        # Context recovery suffix appears after 3rd consecutive error
        # Calls: 1=success, 2=fail, 3=fail, 4=fail(3rd consec), 5=retry(has recovery suffix)
        # After 3 consecutive errors, consecutive_errors==3, so next prompt gets suffix
        recovery_prompts = [p for p in prompts_used if "RECOVERY MODE" in p]
        assert len(recovery_prompts) >= 1, "Context recovery suffix should appear after 3+ errors"

        # Zombie cleanup called at least once (at 3rd consecutive error)
        assert zombie_kill_count["n"] >= 1

        # Backoff waits were called (escalating: 60s, 60s, 300s per-second countdown)
        assert len(sleep_durations) > 0

        # Prompt selection uses effective count despite errors
        # Call 1 (eff 1): initial, Call 5 (eff 2): rerun, Call 6 (eff 3): rerun
        assert prompts_used[0].startswith("Review the brief")  # initial

    @pytest.mark.asyncio
    async def test_full_simulation_circuit_breaker(self, tmp_path):
        """All calls fail — circuit breaker triggers after max_consecutive_errors."""
        project = _setup_test_project(tmp_path)
        cfg = _make_e2e_config(
            project, max_iterations=10,
            max_error_retries=0, max_consecutive_errors=3,
            verification_interval=0,
        )
        orch = Orchestrator(cfg)

        async def mock_query(prompt, options):
            from claude_agent_sdk import ProcessError
            raise ProcessError("Command failed with exit code 1", exit_code=1)
            yield  # make this an async generator

        with patch("ralph_tui.orchestrator.query", side_effect=mock_query), \
             patch("ralph_tui.orchestrator.asyncio.sleep", new_callable=AsyncMock), \
             patch.object(orch, "_kill_zombie_claude_processes", new_callable=AsyncMock):
            state = await orch.run()

        # Circuit breaker fired
        assert state.effective_iterations == 0
        assert "Circuit breaker" in state.status
        assert state.consecutive_errors == 3

        # Log file: circuit breaker breaks BEFORE recording the triggering iteration,
        # so we get N-1 entries (2 logged, 3rd triggers the breaker before recording)
        runs_dir = Path(cfg.project_path).resolve().parent / f"{Path(cfg.project_path).resolve().name}-ralph-runs"
        log_file = runs_dir / "ralph-log.jsonl"
        log_entries = []
        for line in log_file.read_text().strip().split("\n"):
            try:
                entry = json.loads(line)
                if "effective_iteration" in entry:
                    log_entries.append(entry)
            except json.JSONDecodeError:
                pass

        assert len(log_entries) == 2
        assert all(not e["is_effective"] for e in log_entries)
        assert all(e["effective_iteration"] == 0 for e in log_entries)

        # Results: same — 2 recorded, 3rd triggered circuit breaker before recording
        assert len(state.results) == 2
        assert all(not r.is_effective for r in state.results)


class TestFullSimulationTwoConsecutiveStops:
    """The 2-consecutive-stops end criteria with full artifact verification."""

    @pytest.mark.asyncio
    async def test_full_simulation_two_consecutive_stops(self, tmp_path):
        """Analyzer: stop, continue, stop, stop → ends at effective 4."""
        project = _setup_test_project(tmp_path)
        cfg = _make_e2e_config(
            project, max_iterations=10, min_iterations=1,
            verification_interval=0, transition_iteration=0,
        )
        orch = Orchestrator(cfg)

        effective_count = {"n": 0}
        # Analyzer pattern: stop, continue, stop, stop
        stop_pattern = [True, False, True, True]

        async def mock_query(prompt, options):
            effective_count["n"] += 1
            cwd = Path(options.cwd)
            brief = cwd / "Brief.MD"
            if brief.exists():
                brief.write_text(brief.read_text() + f"\n<!-- eff {effective_count['n']} -->")
            yield _make_mock_result(f"sess-{effective_count['n']}", "Work done.")

        analyze_count = {"n": 0}

        async def mock_analyze(text, sys_prompt, exit_prompt, iteration_context=None):
            idx = analyze_count["n"]
            analyze_count["n"] += 1
            should_stop = stop_pattern[idx] if idx < len(stop_pattern) else True
            return MagicMock(
                should_stop=should_stop,
                reason="done" if should_stop else "more work",
                summary="complete" if should_stop else "continuing",
            )

        with patch("ralph_tui.orchestrator.query", side_effect=mock_query), \
             patch("ralph_tui.orchestrator.analyze_output", side_effect=mock_analyze):
            state = await orch.run()

        # Should stop at effective 4 (stop, reset, stop, stop=2 consecutive)
        assert state.effective_iterations == 4
        assert state.consecutive_stops == 2
        assert "2 consecutive stops" in state.status

        # Log has exactly 4 entries, all effective
        runs_dir = Path(cfg.project_path).resolve().parent / f"{Path(cfg.project_path).resolve().name}-ralph-runs"
        log_file = runs_dir / "ralph-log.jsonl"
        log_entries = []
        for line in log_file.read_text().strip().split("\n"):
            try:
                entry = json.loads(line)
                if "effective_iteration" in entry:
                    log_entries.append(entry)
            except json.JSONDecodeError:
                pass

        assert len(log_entries) == 4
        assert all(e["is_effective"] for e in log_entries)
        assert [e["effective_iteration"] for e in log_entries] == [1, 2, 3, 4]

    @pytest.mark.asyncio
    async def test_verification_does_not_affect_consecutive_stops(self, tmp_path):
        """Verification iterations should NOT reset or increment consecutive_stops."""
        project = _setup_test_project(tmp_path)
        cfg = _make_e2e_config(
            project, max_iterations=10, min_iterations=1,
            verification_interval=2, transition_iteration=0,
        )
        orch = Orchestrator(cfg)

        effective_count = {"n": 0}
        prompts_used = []

        async def mock_query(prompt, options):
            effective_count["n"] += 1
            cwd = Path(options.cwd)
            prompts_used.append(prompt)

            state_file = cwd / "_ralph_state.json"
            brief = cwd / "Brief.MD"

            if effective_count["n"] == 1:
                # Write citations for verification
                state = json.loads(state_file.read_text())
                state["citations_to_verify"] = [
                    {"citation": "Test v. Case", "claim": "Test claim", "status": "unverified"},
                ]
                state_file.write_text(json.dumps(state, indent=2))
                if brief.exists():
                    brief.write_text(brief.read_text() + "\n<!-- e1 -->")
            elif "BLIND verification" in prompt or "VERIFICATION" in (cwd / "CLAUDE.md").read_text()[:50]:
                # Verification iteration
                if state_file.exists():
                    state = json.loads(state_file.read_text())
                    for c in state.get("citations_to_verify", []):
                        c["status"] = "verified"
                    state_file.write_text(json.dumps(state, indent=2))
                (cwd / "_verification_report.md").write_text("# Report\nAll verified.\n")
            else:
                if brief.exists():
                    brief.write_text(brief.read_text() + f"\n<!-- e{effective_count['n']} -->")

            yield _make_mock_result(f"sess-{effective_count['n']}", "Done.")

        analyze_count = {"n": 0}

        async def mock_analyze(text, sys_prompt, exit_prompt, iteration_context=None):
            analyze_count["n"] += 1
            ctx = iteration_context or {}
            # Eff 1: stop (1/2). Eff 2: verification (ignored). Eff 3: stop (should be 2/2)
            if ctx.get("iteration") == 1:
                return MagicMock(should_stop=True, reason="stop", summary="stop")
            elif ctx.get("is_verification"):
                return MagicMock(should_stop=True, reason="verif done", summary="verif")
            else:
                return MagicMock(should_stop=True, reason="stop", summary="stop")

        with patch("ralph_tui.orchestrator.query", side_effect=mock_query), \
             patch("ralph_tui.orchestrator.analyze_output", side_effect=mock_analyze):
            state = await orch.run()

        # Eff 1: stop (1/2), Eff 2: verification (no effect on stops), Eff 3: stop (2/2)
        assert state.effective_iterations == 3
        assert state.consecutive_stops == 2
        assert "2 consecutive stops" in state.status
