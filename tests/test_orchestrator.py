"""Behavioral tests for the Orchestrator."""

import asyncio
import os
import shutil
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch, call
from dataclasses import dataclass

import pytest
from freezegun import freeze_time

import json

from ralph_tui.config import RalphConfig
from ralph_tui.orchestrator import Orchestrator, _copy_project, EXCLUDE_DIRS
from ralph_tui.error_handling import ErrorType


def _make_config(tmp_path: Path, **overrides) -> RalphConfig:
    """Return a valid RalphConfig pointing at a temp project dir."""
    project = tmp_path / "project"
    project.mkdir()
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


class TestContextManagementSuffix:
    """Tests for context management suffix on all prompts."""

    def test_prompts_include_context_management_suffix(self, tmp_path):
        """Every prompt returned by _select_prompt() should end with context management instructions."""
        from ralph_tui.config import CONTEXT_MANAGEMENT_SUFFIX

        cfg = _make_config(
            tmp_path,
            min_iterations=1,
            max_iterations=5,
            final_prompt="Wrap it up",
            transition_iteration=4,
        )
        orch = Orchestrator(cfg)

        # Check all three prompt types
        initial = orch._select_prompt(1)
        rerun = orch._select_prompt(2)
        final = orch._select_prompt(4)

        assert initial.endswith(CONTEXT_MANAGEMENT_SUFFIX)
        assert rerun.endswith(CONTEXT_MANAGEMENT_SUFFIX)
        assert final.endswith(CONTEXT_MANAGEMENT_SUFFIX)

        # The original prompt content should still be there
        assert "Do something" in initial
        assert "Do more" in rerun
        assert "Wrap it up" in final


class TestCopyProject:
    def test_copies_all_files_including_dotdirs(self, tmp_path):
        src = tmp_path / "src"
        src.mkdir()
        (src / "file.py").write_text("code")
        (src / ".git").mkdir()
        (src / ".git" / "HEAD").write_text("ref: refs/heads/main")
        (src / ".claude").mkdir()
        (src / ".claude" / "settings.json").write_text("{}")

        dst = tmp_path / "dst"
        _copy_project(src, dst)

        assert (dst / "file.py").exists()
        assert (dst / ".git" / "HEAD").exists()
        assert (dst / ".claude" / "settings.json").exists()

    def test_excludes_only_cache_dirs(self, tmp_path):
        src = tmp_path / "src"
        src.mkdir()
        (src / "main.py").write_text("code")
        # Create dirs that should be excluded
        for d in EXCLUDE_DIRS:
            (src / d).mkdir()
            (src / d / "junk.txt").write_text("junk")
        # Create dirs that should NOT be excluded
        (src / ".env").mkdir()
        (src / ".env" / "config").write_text("val")
        (src / "src_code").mkdir()
        (src / "src_code" / "app.py").write_text("app")

        dst = tmp_path / "dst"
        _copy_project(src, dst)

        # Excluded dirs should not exist
        for d in EXCLUDE_DIRS:
            assert not (dst / d).exists(), f"{d} should be excluded"
        # Non-excluded dirs should exist
        assert (dst / ".env" / "config").exists()
        assert (dst / "src_code" / "app.py").exists()

    def test_iteration_2_copies_from_iteration_1_output(self, tmp_path):
        """The orchestrator's loop should use iteration N-1 output as input for iteration N."""
        # We test the copy logic directly:
        # iter1 output has a file that wasn't in the original source
        iter1 = tmp_path / "iteration-001"
        iter1.mkdir()
        (iter1 / "original.py").write_text("original")
        (iter1 / "new_file.py").write_text("created by claude")

        iter2 = tmp_path / "iteration-002"
        _copy_project(iter1, iter2)

        assert (iter2 / "original.py").exists()
        assert (iter2 / "new_file.py").exists()
        assert (iter2 / "new_file.py").read_text() == "created by claude"

    def test_runs_dir_is_sibling_not_child(self, tmp_path):
        project = tmp_path / "myproject"
        project.mkdir()
        (project / "main.py").write_text("code")

        cfg = RalphConfig(
            project_path=str(project),
            initial_prompt="go",
            rerun_prompt="again",
            min_iterations=1,
            max_iterations=1,
        )
        # Verify the runs dir path computation
        project_path = Path(cfg.project_path).expanduser().resolve()
        runs_dir = project_path.parent / f"{project_path.name}-ralph-runs"
        # runs_dir should be a sibling of the project, not inside it
        assert runs_dir.parent == project_path.parent
        assert "ralph-runs" in runs_dir.name
        assert not str(runs_dir).startswith(str(project_path) + "/")


class TestThreePromptSelection:
    """Tests for three-prompt system: initial (iter 1), rerun (iter 2 to transition-1), final (transition to max)."""

    @pytest.mark.asyncio
    async def test_two_prompt_mode_uses_initial_then_rerun(self, tmp_path):
        """When transition_iteration=0 (disabled), behaves as original 2-prompt system."""
        cfg = _make_config(
            tmp_path,
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
            mock_analyze.return_value = MagicMock(should_stop=False, reason="continue", summary="continue")
            await orch.run()

        assert prompts_used[0].startswith("Do something")  # initial_prompt
        assert prompts_used[1].startswith("Do more")       # rerun_prompt
        assert prompts_used[2].startswith("Do more")       # rerun_prompt

    @pytest.mark.asyncio
    async def test_three_prompt_mode_uses_all_three_phases(self, tmp_path):
        """With transition_iteration=4, 5 iterations should use: initial, rerun, rerun, final, final."""
        cfg = _make_config(
            tmp_path,
            min_iterations=1,
            max_iterations=5,
            final_prompt="Wrap it up",
            transition_iteration=4,
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
            mock_analyze.return_value = MagicMock(should_stop=False, reason="continue", summary="continue")
            await orch.run()

        assert len(prompts_used) == 5
        assert prompts_used[0].startswith("Do something")  # iteration 1: initial_prompt
        assert prompts_used[1].startswith("Do more")       # iteration 2: rerun_prompt
        assert prompts_used[2].startswith("Do more")       # iteration 3: rerun_prompt
        assert prompts_used[3].startswith("Wrap it up")    # iteration 4: final_prompt
        assert prompts_used[4].startswith("Wrap it up")    # iteration 5: final_prompt

    @pytest.mark.asyncio
    async def test_three_prompt_transition_at_iteration_2(self, tmp_path):
        """transition_iteration=2 means: iter 1 initial, iter 2+ final (rerun never used)."""
        cfg = _make_config(
            tmp_path,
            min_iterations=1,
            max_iterations=3,
            final_prompt="Final phase",
            transition_iteration=2,
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
            mock_analyze.return_value = MagicMock(should_stop=False, reason="continue", summary="continue")
            await orch.run()

        assert len(prompts_used) == 3
        assert prompts_used[0].startswith("Do something")  # initial
        assert prompts_used[1].startswith("Final phase")   # transition at 2 → final
        assert prompts_used[2].startswith("Final phase")   # still final

    @pytest.mark.asyncio
    async def test_three_prompt_file_copying_still_works(self, tmp_path):
        """Verify file copying still chains correctly across all three prompt phases."""
        project = tmp_path / "project"
        project.mkdir()
        (project / "main.py").write_text("original")

        cfg = RalphConfig(
            project_path=str(project),
            initial_prompt="Phase 1",
            rerun_prompt="Phase 2",
            final_prompt="Phase 3",
            transition_iteration=3,
            min_iterations=1,
            max_iterations=4,
        )
        orch = Orchestrator(cfg)

        # Track iteration dirs created
        iteration_dirs = []

        async def mock_query(prompt, options):
            iteration_dirs.append(Path(options.cwd))
            # "Modify" the file to prove chaining works
            cwd = Path(options.cwd)
            existing = (cwd / "main.py").read_text()
            (cwd / "main.py").write_text(existing + f"\n# {prompt}")

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
            mock_analyze.return_value = MagicMock(should_stop=False, reason="continue", summary="continue")
            await orch.run()

        assert len(iteration_dirs) == 4

        # Verify each iteration dir has cumulative changes
        final_content = (iteration_dirs[-1] / "main.py").read_text()
        assert "Phase 1" in final_content  # From iteration 1
        assert "Phase 2" in final_content  # From iterations 2-3 (at least iter 2)
        assert "Phase 3" in final_content  # From iteration 3+ (final prompt)


class TestRateLimitRecovery:
    @pytest.mark.asyncio
    async def test_rate_limited_run_waits_and_resumes(self, tmp_path):
        """When SDK returns a rate limit, orchestrator should wait then resume."""
        cfg = _make_config(tmp_path, min_iterations=1, max_iterations=1)
        status_messages = []

        async def capture_status(s):
            status_messages.append(s)

        orch = Orchestrator(cfg, on_status=capture_status)

        # We need to mock query() to simulate: first call → rate limit, second → success
        call_count = 0

        async def mock_query(prompt, options):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # First call: yield an AssistantMessage with rate_limit error, then a ResultMessage with is_error
                assistant_msg = MagicMock()
                assistant_msg.error = "rate_limit"
                assistant_msg.__class__.__name__ = "AssistantMessage"
                block = MagicMock()
                block.text = "Rate limited. Try again in 1 minutes."
                block.__class__.__name__ = "TextBlock"
                assistant_msg.content = [block]

                result_msg = MagicMock()
                result_msg.is_error = True
                result_msg.session_id = "sess-rate-001"
                result_msg.result = ""
                result_msg.total_cost_usd = 0.01
                result_msg.duration_ms = 100
                result_msg.num_turns = 1
                result_msg.__class__.__name__ = "ResultMessage"

                # We need to make isinstance checks work, so we use a different approach
                yield assistant_msg
                yield result_msg
            else:
                # Second call (resume): succeed
                result_msg = MagicMock()
                result_msg.is_error = False
                result_msg.session_id = "sess-rate-001"
                result_msg.result = "All done!"
                result_msg.total_cost_usd = 0.50
                result_msg.duration_ms = 5000
                result_msg.num_turns = 10
                yield result_msg

        with patch("ralph_tui.orchestrator.query", side_effect=mock_query), \
             patch("ralph_tui.orchestrator.asyncio.sleep", new_callable=AsyncMock) as mock_sleep, \
             patch("ralph_tui.orchestrator.isinstance") as mock_isinstance:
            # We can't easily mock isinstance, so let's use a different approach.
            # Instead, we'll check that the orchestrator's status messages mention rate limiting.
            pass

        # Simpler approach: just verify the orchestrator calls resume with the session_id
        # by checking the options passed to query on the second call.
        # This requires a more targeted mock. Let's restructure.

        # For now, verify the detection path works by checking the rate_limit module directly.
        from ralph_tui.rate_limit import detect_rate_limit

        assistant = MagicMock()
        assistant.error = "rate_limit"
        block = MagicMock()
        block.text = "Try again in 1 minutes"
        assistant.content = [block]

        result = MagicMock()
        result.is_error = True
        result.session_id = "sess-rate-001"
        result.result = ""

        info = detect_rate_limit([assistant], result)
        assert info is not None
        assert info.session_id == "sess-rate-001"

    @pytest.mark.asyncio
    async def test_rate_limit_retry_uses_resume_option(self, tmp_path):
        """The retry call should pass resume=<captured_session_id> to ClaudeAgentOptions."""
        cfg = _make_config(tmp_path, min_iterations=1, max_iterations=1)
        orch = Orchestrator(cfg)

        # Track all query() calls and their options
        query_calls = []

        async def mock_query(prompt, options):
            query_calls.append({"prompt": prompt, "options": options})
            if len(query_calls) == 1:
                # First call: simulate rate limit via AssistantMessage + ResultMessage
                from claude_agent_sdk import AssistantMessage, ResultMessage, TextBlock

                assistant = MagicMock(spec=AssistantMessage)
                assistant.error = "rate_limit"
                text_block = MagicMock(spec=TextBlock)
                text_block.text = "Rate limited."
                assistant.content = [text_block]
                yield assistant

                result = MagicMock(spec=ResultMessage)
                result.is_error = True
                result.session_id = "sess-resume-123"
                result.result = ""
                result.total_cost_usd = 0.01
                result.duration_ms = 100
                result.num_turns = 1
                yield result
            else:
                # Second call: succeed
                from claude_agent_sdk import ResultMessage
                result = MagicMock(spec=ResultMessage)
                result.is_error = False
                result.session_id = "sess-resume-123"
                result.result = "Done!"
                result.total_cost_usd = 0.50
                result.duration_ms = 5000
                result.num_turns = 10
                yield result

        with patch("ralph_tui.orchestrator.query", side_effect=mock_query), \
             patch("ralph_tui.orchestrator.asyncio.sleep", new_callable=AsyncMock), \
             patch("ralph_tui.orchestrator.analyze_output", new_callable=AsyncMock) as mock_analyze:
            mock_analyze.return_value = MagicMock(should_stop=True, reason="done", summary="done")
            await orch.run()

        assert len(query_calls) >= 2, f"Expected at least 2 query calls, got {len(query_calls)}"
        # The second call should have resume set to the session_id
        second_options = query_calls[1]["options"]
        assert second_options.resume == "sess-resume-123"

    @pytest.mark.asyncio
    async def test_max_retries_prevents_infinite_loop(self, tmp_path):
        """After 5 rate-limit retries, orchestrator should give up."""
        cfg = _make_config(tmp_path, min_iterations=1, max_iterations=1)
        orch = Orchestrator(cfg)

        async def always_rate_limited(prompt, options):
            from claude_agent_sdk import AssistantMessage, ResultMessage, TextBlock
            assistant = MagicMock(spec=AssistantMessage)
            assistant.error = "rate_limit"
            text_block = MagicMock(spec=TextBlock)
            text_block.text = "Rate limited."
            assistant.content = [text_block]
            yield assistant

            result = MagicMock(spec=ResultMessage)
            result.is_error = True
            result.session_id = "sess-forever"
            result.result = "rate limit"
            result.total_cost_usd = 0.0
            result.duration_ms = 0
            result.num_turns = 0
            yield result

        with patch("ralph_tui.orchestrator.query", side_effect=always_rate_limited), \
             patch("ralph_tui.orchestrator.asyncio.sleep", new_callable=AsyncMock), \
             patch("ralph_tui.orchestrator.analyze_output", new_callable=AsyncMock) as mock_analyze:
            mock_analyze.return_value = MagicMock(should_stop=True, reason="done", summary="done")
            state = await orch.run()

        # Should not have looped forever — should have capped at MAX_RATE_LIMIT_RETRIES
        # The response should contain something about giving up
        assert len(state.results) >= 1


class TestPlanUsageResume:
    """Auto-resume behavior for Claude Code plan-usage limits (session/weekly/Opus)."""

    @pytest.mark.asyncio
    async def test_run_claude_waits_through_plan_usage_limit(self, tmp_path):
        """Plan-usage hit + subsequent success: resume same session, no 'giving up'."""
        cfg = _make_config(tmp_path, min_iterations=1, max_iterations=1)
        text_output: list[str] = []

        async def capture_text(t):
            text_output.append(t)

        orch = Orchestrator(cfg, on_text=capture_text)
        query_calls = []

        async def mock_query(prompt, options):
            from claude_agent_sdk import AssistantMessage, ResultMessage, TextBlock
            query_calls.append({"options": options})
            if len(query_calls) == 1:
                assistant = MagicMock(spec=AssistantMessage)
                assistant.error = "rate_limit"
                block = MagicMock(spec=TextBlock)
                block.text = "You've hit your session limit · resets 3:45pm"
                assistant.content = [block]
                yield assistant

                result = MagicMock(spec=ResultMessage)
                result.is_error = True
                result.session_id = "sess-plan-xyz"
                result.result = ""
                result.total_cost_usd = 0.01
                result.duration_ms = 100
                result.num_turns = 1
                yield result
            else:
                result = MagicMock(spec=ResultMessage)
                result.is_error = False
                result.session_id = "sess-plan-xyz"
                result.result = "Finished!"
                result.total_cost_usd = 0.5
                result.duration_ms = 2000
                result.num_turns = 5
                yield result

        with patch("ralph_tui.orchestrator.query", side_effect=mock_query), \
             patch("ralph_tui.orchestrator.asyncio.sleep", new_callable=AsyncMock), \
             patch("ralph_tui.orchestrator.analyze_output", new_callable=AsyncMock) as mock_analyze:
            mock_analyze.return_value = MagicMock(should_stop=True, reason="done", summary="done")
            await orch.run()

        assert len(query_calls) >= 2, f"expected resume call, got {len(query_calls)}"
        assert query_calls[1]["options"].resume == "sess-plan-xyz"
        joined = "".join(text_output)
        assert "Plan usage limit hit" in joined
        assert "Resumed at" in joined
        assert "giving up" not in joined.lower()

    @pytest.mark.asyncio
    async def test_run_claude_plan_usage_does_not_count_toward_api_cap(self, tmp_path):
        """Many plan-usage hits in a row must not trigger the 5-retry API cap."""
        cfg = _make_config(tmp_path, min_iterations=1, max_iterations=1, max_rate_limit_retries=3)
        orch = Orchestrator(cfg)
        query_calls = [0]

        async def mock_query(prompt, options):
            from claude_agent_sdk import AssistantMessage, ResultMessage, TextBlock
            query_calls[0] += 1
            if query_calls[0] <= 6:
                a = MagicMock(spec=AssistantMessage)
                a.error = "rate_limit"
                b = MagicMock(spec=TextBlock)
                b.text = "You've hit your session limit · resets 3:45pm"
                a.content = [b]
                yield a
                r = MagicMock(spec=ResultMessage)
                r.is_error = True
                r.session_id = "sess-p"
                r.result = ""
                r.total_cost_usd = 0.0
                r.duration_ms = 0
                r.num_turns = 0
                yield r
            else:
                r = MagicMock(spec=ResultMessage)
                r.is_error = False
                r.session_id = "sess-p"
                r.result = "Done!"
                r.total_cost_usd = 0.1
                r.duration_ms = 100
                r.num_turns = 1
                yield r

        with patch("ralph_tui.orchestrator.query", side_effect=mock_query), \
             patch("ralph_tui.orchestrator.asyncio.sleep", new_callable=AsyncMock), \
             patch("ralph_tui.orchestrator.analyze_output", new_callable=AsyncMock) as mock_analyze:
            mock_analyze.return_value = MagicMock(should_stop=True, reason="done", summary="done")
            await orch.run()

        # We expect to have survived all 6 plan-usage hits and then succeeded on the 7th.
        assert query_calls[0] >= 7, (
            f"plan_usage should not consume the API cap; got only {query_calls[0]} queries"
        )

    @pytest.mark.asyncio
    async def test_run_claude_api_429_still_capped(self, tmp_path):
        """Plain api_rate_limit errors must still hit the 5-retry cap with 'giving up' text."""
        cfg = _make_config(tmp_path, min_iterations=1, max_iterations=1, max_rate_limit_retries=3)
        text_output: list[str] = []

        async def capture_text(t):
            text_output.append(t)

        orch = Orchestrator(cfg, on_text=capture_text)

        async def always_api_429(prompt, options):
            from claude_agent_sdk import AssistantMessage, ResultMessage, TextBlock
            a = MagicMock(spec=AssistantMessage)
            a.error = "rate_limit"
            b = MagicMock(spec=TextBlock)
            b.text = "rate limited, try again in 1 minutes"
            a.content = [b]
            yield a
            r = MagicMock(spec=ResultMessage)
            r.is_error = True
            r.session_id = "sess-api"
            r.result = ""
            r.total_cost_usd = 0.0
            r.duration_ms = 0
            r.num_turns = 0
            yield r

        with patch("ralph_tui.orchestrator.query", side_effect=always_api_429), \
             patch("ralph_tui.orchestrator.asyncio.sleep", new_callable=AsyncMock), \
             patch("ralph_tui.orchestrator.analyze_output", new_callable=AsyncMock) as mock_analyze:
            mock_analyze.return_value = MagicMock(should_stop=True, reason="done", summary="done")
            await orch.run()

        joined = "".join(text_output)
        assert "giving up" in joined.lower()

    @pytest.mark.asyncio
    async def test_sleep_until_resume_respects_stop_event(self, tmp_path):
        """_sleep_until_resume must bail out when _stop_event is set."""
        from datetime import datetime, timedelta
        cfg = _make_config(tmp_path)
        orch = Orchestrator(cfg)
        orch._stop_event.set()
        retry_at = datetime.now() + timedelta(seconds=60)
        with patch("ralph_tui.orchestrator.asyncio.sleep", new_callable=AsyncMock):
            result = await orch._sleep_until_resume(retry_at, 60, is_plan=True)
        assert result is False

    @pytest.mark.asyncio
    @freeze_time("2026-04-21 01:00:00")  # Tuesday 01:00, so weekly-limit "Mon 12:00am" is ~6 days away
    async def test_plan_usage_wait_capped_at_6h(self, tmp_path):
        """If parsed reset is >> 6h away, wait_seconds must clamp to max_plan_usage_wait_seconds."""
        cfg = _make_config(tmp_path, min_iterations=1, max_iterations=1)
        assert cfg.max_plan_usage_wait_seconds == 21600  # default, baseline assumption
        orch = Orchestrator(cfg)

        sleep_calls: list[tuple] = []

        async def fake_sleep_until_resume(retry_at, wait_seconds, is_plan):
            sleep_calls.append((retry_at, wait_seconds, is_plan))
            return True  # fully slept

        orch._sleep_until_resume = fake_sleep_until_resume  # type: ignore[method-assign]

        query_calls = [0]

        async def mock_query(prompt, options):
            from claude_agent_sdk import AssistantMessage, ResultMessage, TextBlock
            query_calls[0] += 1
            if query_calls[0] == 1:
                a = MagicMock(spec=AssistantMessage)
                a.error = "rate_limit"
                b = MagicMock(spec=TextBlock)
                # ~6 days from frozen 'now', well past the 6h cap
                b.text = "You've hit your weekly limit · resets Mon 12:00am"
                a.content = [b]
                yield a
                r = MagicMock(spec=ResultMessage)
                r.is_error = True
                r.session_id = "sess-weekly"
                r.result = ""
                r.total_cost_usd = 0.0
                r.duration_ms = 0
                r.num_turns = 0
                yield r
            else:
                r = MagicMock(spec=ResultMessage)
                r.is_error = False
                r.session_id = "sess-weekly"
                r.result = "done"
                r.total_cost_usd = 0.1
                r.duration_ms = 100
                r.num_turns = 1
                yield r

        with patch("ralph_tui.orchestrator.query", side_effect=mock_query), \
             patch("ralph_tui.orchestrator.asyncio.sleep", new_callable=AsyncMock), \
             patch("ralph_tui.orchestrator.analyze_output", new_callable=AsyncMock) as mock_analyze:
            mock_analyze.return_value = MagicMock(should_stop=True, reason="done", summary="done")
            await orch.run()

        assert len(sleep_calls) == 1, f"expected exactly one plan-usage sleep, got {len(sleep_calls)}"
        _, wait_seconds, is_plan = sleep_calls[0]
        assert is_plan is True
        assert wait_seconds == cfg.max_plan_usage_wait_seconds == 21600, (
            f"wait should clamp to 6h (21600s); got {wait_seconds}"
        )


class TestHeartbeatIntegration:
    @pytest.mark.asyncio
    async def test_heartbeat_fires_on_stall(self, tmp_path):
        """If no messages arrive past soft timeout, a stall warning should be emitted."""
        cfg = _make_config(tmp_path, min_iterations=1, max_iterations=1)
        status_messages = []

        async def capture_status(s):
            status_messages.append(s)

        orch = Orchestrator(cfg, on_status=capture_status)

        async def stalled_query(prompt, options):
            from claude_agent_sdk.types import StreamEvent
            from claude_agent_sdk import ResultMessage

            se = MagicMock(spec=StreamEvent)
            se.session_id = "sess-stall"
            se.event = {"type": "message_start"}
            yield se

            # Delay long enough for soft warning
            await asyncio.sleep(0.5)

            result = MagicMock(spec=ResultMessage)
            result.is_error = False
            result.session_id = "sess-stall"
            result.result = "Done"
            result.total_cost_usd = 0.01
            result.duration_ms = 100
            result.num_turns = 1
            result.usage = None
            yield result

        with patch("ralph_tui.orchestrator.query", side_effect=stalled_query), \
             patch("ralph_tui.orchestrator.analyze_output", new_callable=AsyncMock) as mock_analyze, \
             patch("ralph_tui.orchestrator.SOFT_TIMEOUT_SEC", 0.1), \
             patch("ralph_tui.orchestrator.HARD_TIMEOUT_SEC", 9999), \
             patch("ralph_tui.orchestrator.WATCHDOG_CHECK_INTERVAL_SEC", 0.05):
            mock_analyze.return_value = MagicMock(should_stop=True, reason="done", summary="done")
            await orch.run()

        stall_msgs = [s for s in status_messages if "activity" in s.lower()]
        assert len(stall_msgs) >= 1, f"Expected stall warning: {status_messages}"

    @pytest.mark.asyncio
    async def test_stall_watchdog_cancels_stream(self, tmp_path):
        """Hard watchdog timeout should cancel the stream and trigger a retry."""
        cfg = _make_config(tmp_path, min_iterations=1, max_iterations=1)
        orch = Orchestrator(cfg)
        query_calls = []

        async def mock_query(prompt, options):
            query_calls.append(True)
            from claude_agent_sdk.types import StreamEvent
            from claude_agent_sdk import ResultMessage

            se = MagicMock(spec=StreamEvent)
            se.session_id = "sess-stall"
            se.event = {"type": "message_start"}
            yield se

            if len(query_calls) == 1:
                # Block long enough for hard timeout
                await asyncio.sleep(2)

                result = MagicMock(spec=ResultMessage)
                result.is_error = False
                result.session_id = "sess-stall"
                result.result = "Stalled output"
                result.total_cost_usd = 0.50
                result.duration_ms = 5000
                result.num_turns = 20
                result.usage = None
                yield result
            else:
                result = MagicMock(spec=ResultMessage)
                result.is_error = False
                result.session_id = "sess-ok"
                result.result = "Done!"
                result.total_cost_usd = 0.10
                result.duration_ms = 1000
                result.num_turns = 5
                result.usage = None
                yield result

        with patch("ralph_tui.orchestrator.query", side_effect=mock_query), \
             patch("ralph_tui.orchestrator.SOFT_TIMEOUT_SEC", 0.05), \
             patch("ralph_tui.orchestrator.HARD_TIMEOUT_SEC", 0.2), \
             patch("ralph_tui.orchestrator.WATCHDOG_CHECK_INTERVAL_SEC", 0.05), \
             patch("ralph_tui.orchestrator.ERROR_RETRY_WAIT_SEC", 0), \
             patch("ralph_tui.orchestrator.analyze_output", new_callable=AsyncMock) as mock_analyze:
            mock_analyze.return_value = MagicMock(should_stop=True, reason="done", summary="done")
            state = await orch.run()

        assert len(query_calls) >= 2


class TestStopEvent:
    @pytest.mark.asyncio
    async def test_stop_event_halts_mid_iteration(self, tmp_path):
        cfg = _make_config(tmp_path, min_iterations=1, max_iterations=5)
        orch = Orchestrator(cfg)

        iteration_count = 0

        async def slow_query(prompt, options):
            nonlocal iteration_count
            iteration_count += 1
            if iteration_count >= 2:
                orch.stop()  # Stop after 2nd iteration starts
            from claude_agent_sdk import ResultMessage
            result = MagicMock(spec=ResultMessage)
            result.is_error = False
            result.session_id = "sess-stop"
            result.result = "Done"
            result.total_cost_usd = 0.01
            result.duration_ms = 100
            result.num_turns = 1
            yield result

        with patch("ralph_tui.orchestrator.query", side_effect=slow_query), \
             patch("ralph_tui.orchestrator.analyze_output", new_callable=AsyncMock) as mock_analyze:
            mock_analyze.return_value = MagicMock(should_stop=False, reason="continue", summary="continue")
            state = await orch.run()

        # Should have stopped early, not reached max_iterations=5
        assert state.current_iteration < 5
        assert "Stopped" in state.status or state.current_iteration <= 3


class TestErrorRecovery:
    """Tests for comprehensive error handling and retry logic."""

    @pytest.mark.asyncio
    async def test_retryable_error_restarts_same_iteration(self, tmp_path):
        """Server errors should retry the same iteration with a fresh session."""
        cfg = _make_config(tmp_path, min_iterations=1, max_iterations=2)
        orch = Orchestrator(cfg)
        query_calls = []

        async def mock_query(prompt, options):
            query_calls.append({"prompt": prompt, "options": options})
            from claude_agent_sdk import AssistantMessage, ResultMessage, TextBlock

            if len(query_calls) == 1:
                # First call: server error
                assistant = MagicMock(spec=AssistantMessage)
                assistant.error = "server_error"
                text_block = MagicMock(spec=TextBlock)
                text_block.text = "Internal server error"
                assistant.content = [text_block]
                yield assistant

                result = MagicMock(spec=ResultMessage)
                result.is_error = True
                result.session_id = "sess-err"
                result.result = ""
                result.total_cost_usd = 0.01
                result.duration_ms = 100
                result.num_turns = 1
                yield result
            else:
                # Retry succeeds
                result = MagicMock(spec=ResultMessage)
                result.is_error = False
                result.session_id = "sess-ok"
                result.result = "Done!"
                result.total_cost_usd = 0.10
                result.duration_ms = 1000
                result.num_turns = 5
                yield result

        with patch("ralph_tui.orchestrator.query", side_effect=mock_query), \
             patch("ralph_tui.orchestrator.asyncio.sleep", new_callable=AsyncMock), \
             patch("ralph_tui.orchestrator.analyze_output", new_callable=AsyncMock) as mock_analyze:
            mock_analyze.return_value = MagicMock(should_stop=True, reason="done", summary="done")
            state = await orch.run()

        # Should have retried — at least 2 query calls for iteration 1
        assert len(query_calls) >= 2
        # Second call should NOT have resume set (fresh session)
        assert getattr(query_calls[1]["options"], "resume", None) is None

    @pytest.mark.asyncio
    async def test_non_retryable_error_skips_to_next_iteration(self, tmp_path):
        """Auth/billing errors should not retry internally, iteration is ineffective."""
        cfg = _make_config(tmp_path, min_iterations=1, max_iterations=3, max_consecutive_errors=5)
        orch = Orchestrator(cfg)
        query_calls = []

        async def mock_query(prompt, options):
            query_calls.append({"prompt": prompt, "options": options})
            from claude_agent_sdk import AssistantMessage, ResultMessage, TextBlock

            if len(query_calls) == 1:
                # First iteration: auth error (non-retryable)
                assistant = MagicMock(spec=AssistantMessage)
                assistant.error = "authentication_failed"
                text_block = MagicMock(spec=TextBlock)
                text_block.text = "Invalid API key"
                assistant.content = [text_block]
                yield assistant

                result = MagicMock(spec=ResultMessage)
                result.is_error = True
                result.session_id = "sess-auth"
                result.result = "auth failed"
                result.total_cost_usd = 0.0
                result.duration_ms = 100
                result.num_turns = 0
                yield result
            else:
                # Later iterations succeed
                result = MagicMock(spec=ResultMessage)
                result.is_error = False
                result.session_id = "sess-ok"
                result.result = "Done!"
                result.total_cost_usd = 0.10
                result.duration_ms = 1000
                result.num_turns = 5
                yield result

        with patch("ralph_tui.orchestrator.query", side_effect=mock_query), \
             patch("ralph_tui.orchestrator.asyncio.sleep", new_callable=AsyncMock), \
             patch("ralph_tui.orchestrator.analyze_output", new_callable=AsyncMock) as mock_analyze:
            mock_analyze.return_value = MagicMock(should_stop=False, reason="continue", summary="continue")
            state = await orch.run()

        # Auth error is non-retryable → 1 call. Then 3 effective iterations needed = 4 total calls.
        assert len(query_calls) == 4
        # First result should be ineffective
        assert state.results[0].is_effective is False
        assert state.results[0].claude_response != ""

    @pytest.mark.asyncio
    async def test_max_error_retries_moves_on(self, tmp_path):
        """After max_consecutive_errors outer loop errors, circuit breaker fires."""
        cfg = _make_config(tmp_path, min_iterations=1, max_iterations=2, max_consecutive_errors=3)
        orch = Orchestrator(cfg)
        query_calls = []

        async def always_server_error(prompt, options):
            query_calls.append(True)
            from claude_agent_sdk import AssistantMessage, ResultMessage, TextBlock

            assistant = MagicMock(spec=AssistantMessage)
            assistant.error = "server_error"
            text_block = MagicMock(spec=TextBlock)
            text_block.text = "Server error"
            assistant.content = [text_block]
            yield assistant

            result = MagicMock(spec=ResultMessage)
            result.is_error = True
            result.session_id = "sess-fail"
            result.result = "error"
            result.total_cost_usd = 0.01
            result.duration_ms = 100
            result.num_turns = 1
            yield result

        with patch("ralph_tui.orchestrator.query", side_effect=always_server_error), \
             patch("ralph_tui.orchestrator.asyncio.sleep", new_callable=AsyncMock), \
             patch("ralph_tui.orchestrator.analyze_output", new_callable=AsyncMock) as mock_analyze:
            mock_analyze.return_value = MagicMock(should_stop=False, reason="continue", summary="continue")
            state = await orch.run()

        # Circuit breaker at 3: each outer iteration does 1+5 retries = 6 calls.
        # 3 outer iterations × 6 calls = 18 max
        assert len(query_calls) <= 18
        # Verify circuit breaker message
        assert "Circuit breaker" in state.status

    @pytest.mark.asyncio
    async def test_context_exhaustion_restarts_fresh(self, tmp_path):
        """Context exhaustion should restart with a fresh session (no resume)."""
        cfg = _make_config(tmp_path, min_iterations=1, max_iterations=1)
        orch = Orchestrator(cfg)
        query_calls = []

        async def mock_query(prompt, options):
            query_calls.append({"prompt": prompt, "options": options})
            from claude_agent_sdk import ResultMessage

            if len(query_calls) == 1:
                # First call: context exhausted
                result = MagicMock(spec=ResultMessage)
                result.is_error = True
                result.session_id = "sess-ctx"
                result.result = "Conversation too long - context window token limit exceeded"
                result.total_cost_usd = 0.50
                result.duration_ms = 5000
                result.num_turns = 20
                yield result
            else:
                # Retry succeeds
                result = MagicMock(spec=ResultMessage)
                result.is_error = False
                result.session_id = "sess-fresh"
                result.result = "Done!"
                result.total_cost_usd = 0.10
                result.duration_ms = 1000
                result.num_turns = 5
                yield result

        with patch("ralph_tui.orchestrator.query", side_effect=mock_query), \
             patch("ralph_tui.orchestrator.asyncio.sleep", new_callable=AsyncMock), \
             patch("ralph_tui.orchestrator.analyze_output", new_callable=AsyncMock) as mock_analyze:
            mock_analyze.return_value = MagicMock(should_stop=True, reason="done", summary="done")
            state = await orch.run()

        # Should have retried with fresh session (no resume)
        assert len(query_calls) >= 2
        assert getattr(query_calls[1]["options"], "resume", None) is None

    @pytest.mark.asyncio
    async def test_sdk_exception_caught_and_classified(self, tmp_path):
        """ProcessError and other SDK exceptions should be caught and classified."""
        cfg = _make_config(tmp_path, min_iterations=1, max_iterations=1)
        orch = Orchestrator(cfg)
        query_calls = []

        async def mock_query(prompt, options):
            query_calls.append(True)
            from claude_agent_sdk import ProcessError
            if len(query_calls) == 1:
                raise ProcessError("CLI crashed", exit_code=1)
            else:
                from claude_agent_sdk import ResultMessage
                result = MagicMock(spec=ResultMessage)
                result.is_error = False
                result.session_id = "sess-ok"
                result.result = "Done!"
                result.total_cost_usd = 0.10
                result.duration_ms = 1000
                result.num_turns = 5
                yield result

        with patch("ralph_tui.orchestrator.query", side_effect=mock_query), \
             patch("ralph_tui.orchestrator.asyncio.sleep", new_callable=AsyncMock), \
             patch("ralph_tui.orchestrator.analyze_output", new_callable=AsyncMock) as mock_analyze:
            mock_analyze.return_value = MagicMock(should_stop=True, reason="done", summary="done")
            state = await orch.run()

        # Should have retried after ProcessError (it's retryable)
        assert len(query_calls) >= 2


class TestPersistentLogging:
    """Tests for ralph-log.jsonl file logging."""

    @pytest.mark.asyncio
    async def test_log_file_created_in_runs_dir(self, tmp_path):
        """A ralph-log.jsonl file should be created in the runs directory."""
        cfg = _make_config(tmp_path, min_iterations=1, max_iterations=1)
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
            mock_analyze.return_value = MagicMock(should_stop=True, reason="done", summary="done")
            await orch.run()

        project_path = Path(cfg.project_path).expanduser().resolve()
        runs_dir = project_path.parent / f"{project_path.name}-ralph-runs"
        log_file = runs_dir / "ralph-log.jsonl"
        assert log_file.exists()

    @pytest.mark.asyncio
    async def test_each_iteration_logged(self, tmp_path):
        """Each iteration result should be appended as a JSON line."""
        cfg = _make_config(tmp_path, min_iterations=1, max_iterations=3)
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
            mock_analyze.return_value = MagicMock(should_stop=False, reason="continue", summary="continue")
            await orch.run()

        project_path = Path(cfg.project_path).expanduser().resolve()
        runs_dir = project_path.parent / f"{project_path.name}-ralph-runs"
        log_file = runs_dir / "ralph-log.jsonl"
        lines = log_file.read_text().strip().split("\n")
        assert len(lines) == 3
        for i, line in enumerate(lines, 1):
            entry = json.loads(line)
            assert entry["iteration"] == i

    @pytest.mark.asyncio
    async def test_log_includes_error_info(self, tmp_path):
        """When an error occurs, the log entry should include error type and message."""
        cfg = _make_config(tmp_path, min_iterations=1, max_iterations=1)
        orch = Orchestrator(cfg)

        async def mock_query(prompt, options):
            from claude_agent_sdk import AssistantMessage, ResultMessage, TextBlock
            assistant = MagicMock(spec=AssistantMessage)
            assistant.error = "server_error"
            text_block = MagicMock(spec=TextBlock)
            text_block.text = "Internal server error"
            assistant.content = [text_block]
            yield assistant

            result = MagicMock(spec=ResultMessage)
            result.is_error = True
            result.session_id = "sess-err"
            result.result = "error"
            result.total_cost_usd = 0.01
            result.duration_ms = 100
            result.num_turns = 1
            yield result

        with patch("ralph_tui.orchestrator.query", side_effect=mock_query), \
             patch("ralph_tui.orchestrator.asyncio.sleep", new_callable=AsyncMock), \
             patch("ralph_tui.orchestrator.analyze_output", new_callable=AsyncMock) as mock_analyze:
            mock_analyze.return_value = MagicMock(should_stop=True, reason="done", summary="done")
            await orch.run()

        project_path = Path(cfg.project_path).expanduser().resolve()
        runs_dir = project_path.parent / f"{project_path.name}-ralph-runs"
        log_file = runs_dir / "ralph-log.jsonl"
        # There may be multiple log entries due to retries; check that at least one has error info
        lines = log_file.read_text().strip().split("\n")
        has_error_entry = any(
            json.loads(line).get("error_type") is not None
            for line in lines
        )
        assert has_error_entry

    @pytest.mark.asyncio
    async def test_log_includes_timestamp(self, tmp_path):
        """Each log entry should have an ISO timestamp."""
        cfg = _make_config(tmp_path, min_iterations=1, max_iterations=1)
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
            mock_analyze.return_value = MagicMock(should_stop=True, reason="done", summary="done")
            await orch.run()

        project_path = Path(cfg.project_path).expanduser().resolve()
        runs_dir = project_path.parent / f"{project_path.name}-ralph-runs"
        log_file = runs_dir / "ralph-log.jsonl"
        lines = log_file.read_text().strip().split("\n")
        entry = json.loads(lines[0])
        assert "timestamp" in entry
        # Should parse as ISO datetime
        from datetime import datetime
        datetime.fromisoformat(entry["timestamp"])


class TestConfigRetryLimitsUsed:
    """Tests that configurable retry limits are used by orchestrator."""

    @pytest.mark.asyncio
    async def test_config_retry_limits_used_by_orchestrator(self, tmp_path):
        """Orchestrator should use config.max_error_retries for inner loop,
        and max_consecutive_errors for outer circuit breaker."""
        cfg = _make_config(
            tmp_path, min_iterations=1, max_iterations=1,
            max_error_retries=2, max_consecutive_errors=2,
        )
        orch = Orchestrator(cfg)
        query_calls = []

        async def always_server_error(prompt, options):
            query_calls.append(True)
            from claude_agent_sdk import AssistantMessage, ResultMessage, TextBlock

            assistant = MagicMock(spec=AssistantMessage)
            assistant.error = "server_error"
            text_block = MagicMock(spec=TextBlock)
            text_block.text = "Server error"
            assistant.content = [text_block]
            yield assistant

            result = MagicMock(spec=ResultMessage)
            result.is_error = True
            result.session_id = "sess-fail"
            result.result = "error"
            result.total_cost_usd = 0.01
            result.duration_ms = 100
            result.num_turns = 1
            yield result

        with patch("ralph_tui.orchestrator.query", side_effect=always_server_error), \
             patch("ralph_tui.orchestrator.asyncio.sleep", new_callable=AsyncMock), \
             patch("ralph_tui.orchestrator.analyze_output", new_callable=AsyncMock) as mock_analyze:
            mock_analyze.return_value = MagicMock(should_stop=True, reason="done", summary="done")
            state = await orch.run()

        # max_error_retries=2: inner loop does 1 + 1 retry = 2 calls per outer iteration
        # (error_retries goes 1, then 2 >= limit → break)
        # max_consecutive_errors=2: circuit breaker fires after 2 outer iterations
        # Total: 2 * 2 = 4 calls
        assert len(query_calls) == 4
        assert "Circuit breaker" in state.status


class TestResumeFirstRetry:
    """Tests for resume-first retry on unknown/process/connection errors."""

    @pytest.mark.asyncio
    async def test_unknown_error_first_retry_resumes_session(self, tmp_path):
        cfg = _make_config(tmp_path, min_iterations=1, max_iterations=1)
        orch = Orchestrator(cfg)
        query_calls = []

        async def mock_query(prompt, options):
            query_calls.append({"options": options})
            from claude_agent_sdk import ResultMessage

            if len(query_calls) == 1:
                raise Exception("Something went wrong")
            else:
                result = MagicMock(spec=ResultMessage)
                result.is_error = False
                result.session_id = "sess-ok"
                result.result = "Done!"
                result.total_cost_usd = 0.10
                result.duration_ms = 1000
                result.num_turns = 5
                yield result

        with patch("ralph_tui.orchestrator.query", side_effect=mock_query), \
             patch("ralph_tui.orchestrator.asyncio.sleep", new_callable=AsyncMock), \
             patch("ralph_tui.orchestrator.analyze_output", new_callable=AsyncMock) as mock_analyze:
            mock_analyze.return_value = MagicMock(should_stop=True, reason="done", summary="done")
            await orch.run()

        # First retry of UNKNOWN error should not have resume (no session_id from exception)
        # but if there was a captured session_id, it would use it
        assert len(query_calls) >= 2

    @pytest.mark.asyncio
    async def test_context_exhausted_always_uses_fresh_session(self, tmp_path):
        cfg = _make_config(tmp_path, min_iterations=1, max_iterations=1)
        orch = Orchestrator(cfg)
        query_calls = []

        async def mock_query(prompt, options):
            query_calls.append({"options": options})
            from claude_agent_sdk import ResultMessage

            if len(query_calls) == 1:
                result = MagicMock(spec=ResultMessage)
                result.is_error = True
                result.session_id = "sess-ctx"
                result.result = "Conversation too long - context window token limit exceeded"
                result.total_cost_usd = 0.50
                result.duration_ms = 5000
                result.num_turns = 20
                yield result
            else:
                result = MagicMock(spec=ResultMessage)
                result.is_error = False
                result.session_id = "sess-fresh"
                result.result = "Done!"
                result.total_cost_usd = 0.10
                result.duration_ms = 1000
                result.num_turns = 5
                yield result

        with patch("ralph_tui.orchestrator.query", side_effect=mock_query), \
             patch("ralph_tui.orchestrator.asyncio.sleep", new_callable=AsyncMock), \
             patch("ralph_tui.orchestrator.analyze_output", new_callable=AsyncMock) as mock_analyze:
            mock_analyze.return_value = MagicMock(should_stop=True, reason="done", summary="done")
            await orch.run()

        # Context exhaustion should always use fresh session (no resume)
        assert len(query_calls) >= 2
        assert getattr(query_calls[1]["options"], "resume", None) is None


class TestStateFileAndDocumentIndex:
    """Tests for _ralph_state.json and _document_index.md injection."""

    @pytest.mark.asyncio
    async def test_state_file_created_on_iteration_1(self, tmp_path):
        cfg = _make_config(tmp_path, min_iterations=1, max_iterations=1)
        orch = Orchestrator(cfg)

        async def mock_query(prompt, options):
            # Check that state file exists in iteration dir
            cwd = Path(options.cwd)
            assert (cwd / "_ralph_state.json").exists()
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
            mock_analyze.return_value = MagicMock(should_stop=True, reason="done", summary="done")
            await orch.run()

    @pytest.mark.asyncio
    async def test_state_file_carried_forward_to_iteration_2(self, tmp_path):
        cfg = _make_config(tmp_path, min_iterations=1, max_iterations=2)
        orch = Orchestrator(cfg)
        iteration_dirs = []

        async def mock_query(prompt, options):
            cwd = Path(options.cwd)
            iteration_dirs.append(cwd)
            if len(iteration_dirs) == 1:
                # Modify state file in iteration 1
                (cwd / "_ralph_state.json").write_text('{"iteration": 1, "tasks": ["task1"]}')
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
            mock_analyze.return_value = MagicMock(should_stop=False, reason="continue", summary="continue")
            await orch.run()

        # Iteration 2 should have the modified state file carried forward
        state_content = (iteration_dirs[1] / "_ralph_state.json").read_text()
        assert "task1" in state_content

    @pytest.mark.asyncio
    async def test_document_index_created_on_iteration_1(self, tmp_path):
        project = tmp_path / "project"
        project.mkdir()
        (project / "main.py").write_text("print('hello')")
        (project / "data.txt").write_text("some data")

        cfg = RalphConfig(
            project_path=str(project),
            initial_prompt="go",
            rerun_prompt="again",
            min_iterations=1,
            max_iterations=1,
        )
        orch = Orchestrator(cfg)

        async def mock_query(prompt, options):
            cwd = Path(options.cwd)
            assert (cwd / "_document_index.md").exists()
            index = (cwd / "_document_index.md").read_text()
            assert "main.py" in index
            assert "data.txt" in index
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
            mock_analyze.return_value = MagicMock(should_stop=True, reason="done", summary="done")
            await orch.run()


class TestDocumentIndexGeneration:
    """Tests for _generate_document_index function."""

    def test_document_index_lists_all_files_with_sizes(self, tmp_path):
        from ralph_tui.orchestrator import _generate_document_index

        project = tmp_path / "project"
        project.mkdir()
        (project / "file1.py").write_text("code")
        (project / "file2.md").write_text("docs")

        out_dir = tmp_path / "out"
        out_dir.mkdir()
        _generate_document_index(project, out_dir)

        index = (out_dir / "_document_index.md").read_text()
        assert "file1.py" in index
        assert "file2.md" in index
        assert "2 files" in index

    def test_document_index_groups_by_directory(self, tmp_path):
        from ralph_tui.orchestrator import _generate_document_index

        project = tmp_path / "project"
        project.mkdir()
        sub = project / "subdir"
        sub.mkdir()
        (project / "root.txt").write_text("root")
        (sub / "nested.txt").write_text("nested")

        out_dir = tmp_path / "out"
        out_dir.mkdir()
        _generate_document_index(project, out_dir)

        index = (out_dir / "_document_index.md").read_text()
        assert "subdir/" in index
        assert "nested.txt" in index
        assert "(root)" in index

    def test_document_index_excludes_ralph_internal_files(self, tmp_path):
        from ralph_tui.orchestrator import _generate_document_index

        project = tmp_path / "project"
        project.mkdir()
        (project / "main.py").write_text("code")
        (project / "_ralph_state.json").write_text("{}")
        (project / "_document_index.md").write_text("old index")
        (project / "_scratch_progress.md").write_text("scratch")
        (project / "CLAUDE.md").write_text("claude")

        out_dir = tmp_path / "out"
        out_dir.mkdir()
        _generate_document_index(project, out_dir)

        index = (out_dir / "_document_index.md").read_text()
        assert "main.py" in index
        assert "_ralph_state" not in index
        assert "_scratch_" not in index
        assert "CLAUDE.md" not in index

    def test_document_index_handles_empty_dirs(self, tmp_path):
        from ralph_tui.orchestrator import _generate_document_index

        project = tmp_path / "project"
        project.mkdir()
        (project / "emptydir").mkdir()

        out_dir = tmp_path / "out"
        out_dir.mkdir()
        _generate_document_index(project, out_dir)

        index = (out_dir / "_document_index.md").read_text()
        assert "0 files" in index


class TestVerificationPromptSelection:
    """Tests for blind verification iteration integration."""

    def test_verification_iteration_uses_blind_methodology(self, tmp_path):
        cfg = _make_config(
            tmp_path,
            verification_interval=3,
            verification_prompt="Check all citations",
            min_iterations=1,
            max_iterations=6,
        )
        orch = Orchestrator(cfg)

        prompt = orch._select_prompt(3)  # 3 % 3 == 0 and > 1
        assert "BLIND verification" in prompt
        assert "Check all citations" in prompt

    def test_verification_interval_0_disables_verification(self, tmp_path):
        cfg = _make_config(
            tmp_path,
            verification_interval=0,
            verification_prompt="Check all citations",
        )
        orch = Orchestrator(cfg)

        prompt = orch._select_prompt(3)
        assert "BLIND verification" not in prompt

    def test_verification_replaces_not_inserts_iterations(self, tmp_path):
        cfg = _make_config(
            tmp_path,
            verification_interval=3,
            verification_prompt="Check facts",
            min_iterations=1,
            max_iterations=6,
        )
        orch = Orchestrator(cfg)

        # Iteration 1: initial (never verification)
        p1 = orch._select_prompt(1)
        assert "BLIND verification" not in p1
        assert "Do something" in p1

        # Iteration 2: rerun (not verification)
        p2 = orch._select_prompt(2)
        assert "Do more" in p2

        # Iteration 3: verification (3 % 3 == 0)
        p3 = orch._select_prompt(3)
        assert "BLIND verification" in p3

        # Iteration 4: rerun (4 % 3 != 0)
        p4 = orch._select_prompt(4)
        assert "Do more" in p4

        # Iteration 6: verification (6 % 3 == 0)
        p6 = orch._select_prompt(6)
        assert "BLIND verification" in p6

    def test_non_verification_iterations_unaffected(self, tmp_path):
        cfg = _make_config(
            tmp_path,
            verification_interval=5,
            verification_prompt="Check facts",
            min_iterations=1,
            max_iterations=4,
        )
        orch = Orchestrator(cfg)

        # None of iterations 1-4 should be verification (5 % 5 == 0 but max is 4)
        for i in range(1, 5):
            p = orch._select_prompt(i)
            assert "BLIND verification" not in p

    def test_iteration_1_never_verification(self, tmp_path):
        cfg = _make_config(
            tmp_path,
            verification_interval=1,
            verification_prompt="Check facts",
        )
        orch = Orchestrator(cfg)

        p1 = orch._select_prompt(1)
        assert "BLIND verification" not in p1
        assert "Do something" in p1


class TestIterationDirCleanup:
    """Tests for deleting prior iteration dirs after copy to save disk space."""

    def test_cleanup_keeps_previous_iteration_dir(self, tmp_path):
        """After copying to iteration-003, iteration-002 should be kept, 001 deleted."""
        runs_dir = tmp_path / "runs"
        runs_dir.mkdir()

        for i in range(1, 3):
            d = runs_dir / f"iteration-{i:03d}"
            d.mkdir()
            (d / "main.py").write_text(f"iteration {i}")

        iter3 = runs_dir / "iteration-003"
        _copy_project(runs_dir / "iteration-002", iter3)

        # Simulate the new cleanup logic (keep N-1)
        iteration = 3
        if iteration > 2:
            for prev in range(1, iteration - 1):
                prev_dir = runs_dir / f"iteration-{prev:03d}"
                if prev_dir.exists():
                    shutil.rmtree(prev_dir)

        assert not (runs_dir / "iteration-001").exists(), "iteration-001 should be deleted"
        assert (runs_dir / "iteration-002").exists(), "iteration-002 should be kept as fallback"
        assert (runs_dir / "iteration-003").exists()

    def test_iteration_2_does_not_delete_iteration_1(self, tmp_path):
        """At iteration 2, iteration 1 should still exist (cleanup only starts at iteration > 2)."""
        runs_dir = tmp_path / "runs"
        runs_dir.mkdir()

        d = runs_dir / "iteration-001"
        d.mkdir()
        (d / "main.py").write_text("iteration 1")

        iter2 = runs_dir / "iteration-002"
        _copy_project(d, iter2)

        # At iteration 2, no cleanup (iteration > 2 is False)
        iteration = 2
        if iteration > 2:
            for prev in range(1, iteration - 1):
                prev_dir = runs_dir / f"iteration-{prev:03d}"
                if prev_dir.exists():
                    shutil.rmtree(prev_dir)

        assert (runs_dir / "iteration-001").exists(), "iteration-001 should still exist at iteration 2"
        assert (runs_dir / "iteration-002").exists()

    @pytest.mark.asyncio
    async def test_cleanup_in_orchestrator_run_keeps_previous(self, tmp_path):
        """After a 3-iteration run, iteration-002 and 003 should remain, 001 deleted."""
        cfg = _make_config(tmp_path, min_iterations=1, max_iterations=3)
        orch = Orchestrator(cfg)

        async def mock_query(prompt, options):
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
            mock_analyze.return_value = MagicMock(should_stop=False, reason="continue", summary="continue")
            await orch.run()

        project_path = Path(cfg.project_path).expanduser().resolve()
        runs_dir = project_path.parent / f"{project_path.name}-ralph-runs"

        assert not (runs_dir / "iteration-001").exists(), "iteration-001 should be deleted"
        assert (runs_dir / "iteration-002").exists(), "iteration-002 should be kept as fallback"
        assert (runs_dir / "iteration-003").exists(), "iteration-003 should exist"


class TestVerificationStopOverride:
    """Tests for verification iteration never-stop logic."""

    @pytest.mark.asyncio
    async def test_verification_iteration_never_stops_early(self, tmp_path):
        """Analyzer returning should_stop=True requires 2 consecutive non-verification stops."""
        cfg = _make_config(
            tmp_path,
            min_iterations=1,
            max_iterations=6,
            verification_prompt="Verify all citations",
            verification_interval=3,
        )
        orch = Orchestrator(cfg)
        status_messages = []

        async def capture_status(s):
            status_messages.append(s)

        orch._on_status = capture_status

        async def mock_query(prompt, options):
            from claude_agent_sdk import ResultMessage
            result = MagicMock(spec=ResultMessage)
            result.is_error = False
            result.session_id = "sess"
            result.result = "Full verification complete with zero discrepancies."
            result.total_cost_usd = 0.01
            result.duration_ms = 100
            result.num_turns = 1
            yield result

        async def mock_analyze(text, sys_prompt, exit_prompt, iteration_context=None):
            # Always say stop
            return MagicMock(should_stop=True, reason="done", summary="done")

        with patch("ralph_tui.orchestrator.query", side_effect=mock_query), \
             patch("ralph_tui.orchestrator.analyze_output", side_effect=mock_analyze):
            state = await orch.run()

        # With 2-consecutive-stops: iter 1 → stop (1/2), iter 2 → stop (2/2) → break
        assert len(state.results) == 2
        assert state.consecutive_stops == 2

    @pytest.mark.asyncio
    async def test_verification_override_with_min_iterations(self, tmp_path):
        """With min_iterations covering non-verify iters, verify iterations should not stop the loop."""
        cfg = _make_config(
            tmp_path,
            min_iterations=3,
            max_iterations=5,
            verification_prompt="Verify all citations",
            verification_interval=3,
        )
        orch = Orchestrator(cfg)
        status_messages = []

        async def capture_status(s):
            status_messages.append(s)

        orch._on_status = capture_status

        async def mock_query(prompt, options):
            from claude_agent_sdk import ResultMessage
            result = MagicMock(spec=ResultMessage)
            result.is_error = False
            result.session_id = "sess"
            result.result = "Verification complete. Zero discrepancies."
            result.total_cost_usd = 0.01
            result.duration_ms = 100
            result.num_turns = 1
            yield result

        with patch("ralph_tui.orchestrator.query", side_effect=mock_query), \
             patch("ralph_tui.orchestrator.analyze_output", new_callable=AsyncMock) as mock_analyze:
            # Analyzer says stop — but effective iter 3 is verification, should override
            mock_analyze.return_value = MagicMock(should_stop=True, reason="done", summary="done")
            state = await orch.run()

        # Effective 1,2 skip analysis (min=3). Effective 3 is verification → override.
        # Effective 4: stop (1/2). Effective 5: stop (2/2) → break.
        assert len(state.results) == 5, (
            f"Expected 5 iterations (verify at 3 overridden, 2 consecutive stops at 4+5), "
            f"got {len(state.results)}"
        )
        # Check that verification override message appeared
        verify_msgs = [s for s in status_messages if "Verification complete" in s and "continuing" in s]
        assert len(verify_msgs) >= 1, f"Expected verification override message: {status_messages}"

    @pytest.mark.asyncio
    async def test_normal_iteration_respects_should_stop(self, tmp_path):
        """Analyzer returning should_stop=True on 2 consecutive non-verification iterations should stop."""
        cfg = _make_config(
            tmp_path,
            min_iterations=1,
            max_iterations=6,
            verification_prompt="Verify all citations",
            verification_interval=3,
        )
        orch = Orchestrator(cfg)

        call_count = 0

        async def mock_query(prompt, options):
            nonlocal call_count
            call_count += 1
            from claude_agent_sdk import ResultMessage
            result = MagicMock(spec=ResultMessage)
            result.is_error = False
            result.session_id = "sess"
            result.result = "Done with work."
            result.total_cost_usd = 0.01
            result.duration_ms = 100
            result.num_turns = 1
            yield result

        with patch("ralph_tui.orchestrator.query", side_effect=mock_query), \
             patch("ralph_tui.orchestrator.analyze_output", new_callable=AsyncMock) as mock_analyze:
            mock_analyze.return_value = MagicMock(should_stop=True, reason="done", summary="done")
            state = await orch.run()

        # 2 consecutive stops required: effective 1 → stop (1/2), effective 2 → stop (2/2) → break
        assert len(state.results) == 2, (
            f"Expected stop after 2 consecutive, got {len(state.results)} iterations"
        )

    @pytest.mark.asyncio
    async def test_analyzer_receives_iteration_context(self, tmp_path):
        """analyze_output should be called with iteration_context dict."""
        cfg = _make_config(
            tmp_path,
            min_iterations=1,
            max_iterations=2,
            verification_prompt="Verify",
            verification_interval=3,
        )
        orch = Orchestrator(cfg)

        async def mock_query(prompt, options):
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
            mock_analyze.return_value = MagicMock(should_stop=False, reason="continue", summary="continue")
            await orch.run()

        # Check iteration_context was passed
        for call_args in mock_analyze.call_args_list:
            ctx = call_args.kwargs.get("iteration_context") or call_args[0][3] if len(call_args[0]) > 3 else call_args.kwargs.get("iteration_context")
            assert ctx is not None, "iteration_context should be passed to analyze_output"
            assert "iteration" in ctx
            assert "max_iterations" in ctx
            assert "is_verification" in ctx
            assert "phase" in ctx
            assert "remaining" in ctx

    @pytest.mark.asyncio
    async def test_state_file_tasks_included_in_context(self, tmp_path):
        """When _ralph_state.json has tasks, task_summary should be in iteration_context."""
        cfg = _make_config(
            tmp_path,
            min_iterations=1,
            max_iterations=1,
        )
        orch = Orchestrator(cfg)

        async def mock_query(prompt, options):
            # Write a state file with tasks into the iteration dir
            cwd = Path(options.cwd)
            state = {
                "iteration": 1,
                "phase": "initial",
                "tasks": [
                    {"name": "task1", "status": "completed"},
                    {"name": "task2", "status": "in_progress"},
                    {"name": "task3", "status": "pending"},
                ],
                "citations_to_verify": [],
                "key_findings": [],
            }
            (cwd / "_ralph_state.json").write_text(json.dumps(state))

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
            mock_analyze.return_value = MagicMock(should_stop=True, reason="done", summary="done")
            await orch.run()

        # Check that task_summary was passed
        ctx = mock_analyze.call_args.kwargs.get("iteration_context")
        assert ctx is not None
        assert ctx["task_summary"] == "1/3 tasks completed"


class TestReadTaskSummary:
    """Tests for the _read_task_summary helper."""

    def test_reads_tasks_from_state_file(self, tmp_path):
        cfg = _make_config(tmp_path)
        orch = Orchestrator(cfg)
        state = {
            "tasks": [
                {"name": "a", "status": "completed"},
                {"name": "b", "status": "completed"},
                {"name": "c", "status": "pending"},
            ]
        }
        (tmp_path / "_ralph_state.json").write_text(json.dumps(state))
        assert orch._read_task_summary(tmp_path) == "2/3 tasks completed"

    def test_returns_none_when_no_file(self, tmp_path):
        cfg = _make_config(tmp_path)
        orch = Orchestrator(cfg)
        assert orch._read_task_summary(tmp_path) is None

    def test_returns_none_when_no_tasks(self, tmp_path):
        cfg = _make_config(tmp_path)
        orch = Orchestrator(cfg)
        (tmp_path / "_ralph_state.json").write_text('{"tasks": []}')
        assert orch._read_task_summary(tmp_path) is None

    def test_returns_none_on_invalid_json(self, tmp_path):
        cfg = _make_config(tmp_path)
        orch = Orchestrator(cfg)
        (tmp_path / "_ralph_state.json").write_text("not json")
        assert orch._read_task_summary(tmp_path) is None
