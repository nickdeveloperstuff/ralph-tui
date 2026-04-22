"""Tests for the analyzer module with iteration context support."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ralph_tui.analyzer import analyze_output, AnalysisResult


def _mock_openrouter_response(should_stop: bool, reason: str = "test", summary: str = "test"):
    """Build a mock httpx response matching OpenRouter's format."""
    verdict = {"should_stop": should_stop, "reason": reason, "summary": summary}
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.json.return_value = {
        "choices": [{"message": {"content": json.dumps(verdict)}}]
    }
    return resp


class TestAnalyzeOutputIterationContext:
    """Tests for iteration_context parameter in analyze_output."""

    @pytest.mark.asyncio
    async def test_includes_iteration_context_in_prompt(self):
        """When iteration_context is provided, the user prompt should include it."""
        mock_resp = _mock_openrouter_response(False)
        captured_payloads = []

        async def capture_post(url, json=None, headers=None):
            captured_payloads.append(json)
            return mock_resp

        mock_client = AsyncMock()
        mock_client.post = capture_post
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        ctx = {
            "iteration": 3,
            "max_iterations": 6,
            "is_verification": True,
            "phase": "verification",
            "remaining": 3,
            "task_summary": "2/5 tasks completed",
        }

        with patch("ralph_tui.analyzer._get_api_key", return_value="test-key"), \
             patch("ralph_tui.analyzer.httpx.AsyncClient", return_value=mock_client):
            await analyze_output("Claude did stuff", "system prompt", "exit prompt", iteration_context=ctx)

        assert len(captured_payloads) == 1
        user_msg = captured_payloads[0]["messages"][1]["content"]
        assert "Iteration: 3 of 6" in user_msg
        assert "Phase: verification" in user_msg
        assert "Is verification iteration: True" in user_msg
        assert "Iterations remaining: 3" in user_msg
        assert "Task status: 2/5 tasks completed" in user_msg

    @pytest.mark.asyncio
    async def test_verification_framing_in_prompt(self):
        """Verification iterations should get the VERIFICATION checkpoint warning."""
        mock_resp = _mock_openrouter_response(False)
        captured_payloads = []

        async def capture_post(url, json=None, headers=None):
            captured_payloads.append(json)
            return mock_resp

        mock_client = AsyncMock()
        mock_client.post = capture_post
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        ctx = {
            "iteration": 3,
            "max_iterations": 6,
            "is_verification": True,
            "phase": "verification",
            "remaining": 3,
            "task_summary": None,
        }

        with patch("ralph_tui.analyzer._get_api_key", return_value="test-key"), \
             patch("ralph_tui.analyzer.httpx.AsyncClient", return_value=mock_client):
            await analyze_output("Done", "sys", "exit", iteration_context=ctx)

        user_msg = captured_payloads[0]["messages"][1]["content"]
        assert "VERIFICATION checkpoint iteration" in user_msg
        assert "does NOT mean the overall workflow is complete" in user_msg

    @pytest.mark.asyncio
    async def test_non_verification_no_checkpoint_warning(self):
        """Non-verification iterations should NOT get the checkpoint warning."""
        mock_resp = _mock_openrouter_response(False)
        captured_payloads = []

        async def capture_post(url, json=None, headers=None):
            captured_payloads.append(json)
            return mock_resp

        mock_client = AsyncMock()
        mock_client.post = capture_post
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        ctx = {
            "iteration": 2,
            "max_iterations": 6,
            "is_verification": False,
            "phase": "rerun",
            "remaining": 4,
            "task_summary": None,
        }

        with patch("ralph_tui.analyzer._get_api_key", return_value="test-key"), \
             patch("ralph_tui.analyzer.httpx.AsyncClient", return_value=mock_client):
            await analyze_output("Done", "sys", "exit", iteration_context=ctx)

        user_msg = captured_payloads[0]["messages"][1]["content"]
        assert "Iteration: 2 of 6" in user_msg
        assert "VERIFICATION checkpoint" not in user_msg

    @pytest.mark.asyncio
    async def test_backward_compat_no_context(self):
        """Calling without iteration_context should still work (backward compat)."""
        mock_resp = _mock_openrouter_response(True, reason="all done", summary="complete")
        captured_payloads = []

        async def capture_post(url, json=None, headers=None):
            captured_payloads.append(json)
            return mock_resp

        mock_client = AsyncMock()
        mock_client.post = capture_post
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("ralph_tui.analyzer._get_api_key", return_value="test-key"), \
             patch("ralph_tui.analyzer.httpx.AsyncClient", return_value=mock_client):
            result = await analyze_output("Claude output", "sys prompt", "exit prompt")

        assert result.should_stop is True
        assert result.summary == "complete"
        # No context block in prompt
        user_msg = captured_payloads[0]["messages"][1]["content"]
        assert "=== Iteration Context ===" not in user_msg
        assert user_msg.startswith("=== Claude Code Response ===")

    @pytest.mark.asyncio
    async def test_task_summary_omitted_when_none(self):
        """When task_summary is None, it should not appear in the prompt."""
        mock_resp = _mock_openrouter_response(False)
        captured_payloads = []

        async def capture_post(url, json=None, headers=None):
            captured_payloads.append(json)
            return mock_resp

        mock_client = AsyncMock()
        mock_client.post = capture_post
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        ctx = {
            "iteration": 1,
            "max_iterations": 6,
            "is_verification": False,
            "phase": "initial",
            "remaining": 5,
            "task_summary": None,
        }

        with patch("ralph_tui.analyzer._get_api_key", return_value="test-key"), \
             patch("ralph_tui.analyzer.httpx.AsyncClient", return_value=mock_client):
            await analyze_output("Done", "sys", "exit", iteration_context=ctx)

        user_msg = captured_payloads[0]["messages"][1]["content"]
        assert "Task status:" not in user_msg
