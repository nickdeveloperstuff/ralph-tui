"""OpenRouter Gemini API integration for semantic analysis of Claude Code output."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

import httpx


OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_MODEL = "google/gemini-3.1-pro-preview"
TIMEOUT_SECONDS = 120


@dataclass
class AnalysisResult:
    should_stop: bool
    reason: str
    summary: str


def _get_api_key() -> str:
    key = os.environ.get("OPENROUTER_API_KEY", "")
    if not key:
        raise RuntimeError(
            "OPENROUTER_API_KEY environment variable is not set. "
            "Set it to your OpenRouter API key."
        )
    return key


async def analyze_output(
    claude_response_text: str,
    analysis_prompt: str,
    exit_condition_prompt: str,
    iteration_context: dict | None = None,
) -> AnalysisResult:
    """Analyze Claude Code's natural language output using Gemini via OpenRouter.

    Args:
        claude_response_text: The result field from Claude Code's output — what Claude
            said it did, in natural language.
        analysis_prompt: System prompt telling Gemini how to evaluate Claude's output.
        exit_condition_prompt: User prompt asking Gemini for a JSON verdict.
        iteration_context: Optional dict with iteration, max_iterations, is_verification,
            phase, remaining, and task_summary keys.

    Returns:
        AnalysisResult with should_stop, reason, and summary.
        On API failure, returns should_stop=False to keep the loop running.
    """
    try:
        api_key = _get_api_key()
    except RuntimeError as e:
        return AnalysisResult(
            should_stop=False,
            reason=f"API key error: {e}",
            summary="Analysis skipped — no API key",
        )

    context_block = ""
    if iteration_context:
        ctx = iteration_context
        context_block = (
            f"=== Iteration Context ===\n"
            f"Iteration: {ctx['iteration']} of {ctx['max_iterations']}\n"
            f"Phase: {ctx['phase']}\n"
            f"Is verification iteration: {ctx['is_verification']}\n"
            f"Iterations remaining: {ctx['remaining']}\n"
        )
        if ctx.get("task_summary"):
            context_block += f"Task status: {ctx['task_summary']}\n"
        if ctx["is_verification"]:
            context_block += (
                "\nIMPORTANT: This was a VERIFICATION checkpoint iteration. "
                "Verification success means the document is accurate so far — "
                "it does NOT mean the overall workflow is complete. "
                "There are still drafting/review iterations remaining.\n"
            )
        context_block += "\n"

    user_content = (
        context_block
        + f"=== Claude Code Response ===\n{claude_response_text}\n\n"
        f"=== Your Task ===\n{exit_condition_prompt}"
    )

    payload = {
        "model": OPENROUTER_MODEL,
        "messages": [
            {"role": "system", "content": analysis_prompt},
            {"role": "user", "content": user_content},
        ],
        "reasoning": {"effort": "high"},
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://auteurlegal.com",
        "X-Title": "Ralph TUI",
    }

    try:
        async with httpx.AsyncClient(timeout=TIMEOUT_SECONDS) as client:
            resp = await client.post(OPENROUTER_BASE_URL, json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()

        content = data["choices"][0]["message"]["content"]
        # Extract JSON from response — Gemini may wrap it in markdown code blocks
        json_str = content
        if "```" in json_str:
            # Extract from code block
            for block in json_str.split("```"):
                stripped = block.strip()
                if stripped.startswith("json"):
                    stripped = stripped[4:].strip()
                if stripped.startswith("{"):
                    json_str = stripped
                    break

        verdict = json.loads(json_str)
        return AnalysisResult(
            should_stop=bool(verdict.get("should_stop", False)),
            reason=str(verdict.get("reason", "No reason provided")),
            summary=str(verdict.get("summary", "No summary")),
        )

    except (httpx.HTTPError, json.JSONDecodeError, KeyError, IndexError) as e:
        return AnalysisResult(
            should_stop=False,
            reason=f"Analysis API error: {type(e).__name__}: {e}",
            summary="Analysis failed — continuing",
        )


async def test_analyzer() -> None:
    """Standalone test for the analyzer."""
    from ralph_tui.config import DEFAULT_ANALYSIS_PROMPT, DEFAULT_EXIT_CONDITION_PROMPT

    sample_output = (
        "I've analyzed the codebase and found the authentication bug in auth.py:42. "
        "The issue was a missing token validation check. I've fixed it and all 15 tests "
        "now pass. The fix involved adding a `validate_token()` call before processing "
        "the request."
    )

    print("Testing analyzer with sample Claude output...")
    result = await analyze_output(
        sample_output, DEFAULT_ANALYSIS_PROMPT, DEFAULT_EXIT_CONDITION_PROMPT
    )
    print(f"  should_stop: {result.should_stop}")
    print(f"  reason: {result.reason}")
    print(f"  summary: {result.summary}")


if __name__ == "__main__":
    import asyncio

    asyncio.run(test_analyzer())
