"""Configuration dataclass with YAML save/load for Ralph TUI."""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields, asdict
from pathlib import Path

import yaml


DEFAULT_ANALYSIS_PROMPT = """\
You are evaluating the natural language output of an AI coding assistant \
(Claude Code). This is what Claude SAID about its work — not the source \
code itself. Analyze the response for: reported errors, signs of incomplete \
work, test failures mentioned, confidence level, and overall task completion \
signals. Claude's output is unpredictable natural language, so use your \
judgment to interpret it semantically."""

DEFAULT_EXIT_CONDITION_PROMPT = """\
Based on your analysis of Claude Code's response text above, should the \
orchestrator stop running (task appears complete) or continue with another \
iteration? Return JSON only: {"should_stop": true/false, "reason": "explanation", "summary": "one-line summary"}"""

CONTEXT_MANAGEMENT_SUFFIX = """

IMPORTANT - Context Window Management:
- Write things down: Create _scratch_progress.md as working memory. Save your plan, findings, and progress there.
- Break down large tasks: Write a plan first, then execute step by step.
- Context hygiene: Use grep/glob for targeted searches. Summarize large outputs in scratch files.
- Run /compact proactively when context approaches 40%.
- Subagents consume context budget. Use them sparingly and only for truly independent research tasks.
- Keep context under 50% capacity at all times.
"""

CLAUDE_MD_TEMPLATE = """\
# Ralph TUI - Iteration Guidelines

## Startup Sequence (DO THIS FIRST)
1. Read `_ralph_state.json` to see current tasks and progress
2. Read `_document_index.md` to understand available files
3. Read any `_scratch_*.md` files to recover prior context
4. Pick ONE task from the task list and work on it

## Context Management (CRITICAL)
- Target: keep context under 50%. Run /compact at 40%.
- Auto-compact is set but may fire late. Be proactive.
- Before /compact: save ALL progress to `_scratch_progress.md` and update `_ralph_state.json`

## Working Memory
Create scratch files to persist state across compactions:
- `_scratch_notes.md` - observations, questions, decisions
- `_scratch_progress.md` - what's done, what's next, blockers

## Citation Discipline
- When citing any source, immediately add it to `_ralph_state.json` `citations_to_verify`
- Format: {"citation": "source ref", "claim": "what you're claiming", "status": "unverified"}
- This enables automated verification in later iterations

## Efficiency
- Subagents consume context budget. Use them sparingly and only for truly independent research tasks.
- Read files directly. Use grep/glob for searching.
- Summarize large outputs immediately in scratch files.
- Prefer targeted edits over full file rewrites.

## Error Recovery
- If a tool fails, try an alternative approach before retrying.
- Save progress to _scratch_progress.md before risky operations.
"""

VERIFICATION_METHODOLOGY_TEMPLATE = """\
You are conducting BLIND verification. Your task: {verification_prompt}

## Methodology (FOLLOW EXACTLY)

Read `_ralph_state.json` and find all entries in `citations_to_verify` with status "unverified".

For EACH unverified item, do these steps IN ORDER:

### Step 1: Independent Research (DO NOT read the "claim" field yet)
- Read ONLY the "citation" field (case name, statute number, or document reference)
- Search the project files independently for that source
- Read the actual source material
- Write your independent finding in `_scratch_verification.md` under a heading for this citation

### Step 2: Compare (NOW read the "claim" field)
- Read the "claim" field from `_ralph_state.json`
- Compare it against your independent finding from Step 1
- Note any discrepancies

### Step 3: Update Status
In `_ralph_state.json`, update each citation's status:
- "verified" — your independent finding matches the claim
- "disputed" — your finding contradicts or significantly differs (add "discrepancy" field)
- "unable_to_verify" — source material not found in project files (add "reason" field)

Also verify factual claims in the main document by independently searching discovery/source files.
"""


@dataclass
class RalphConfig:
    project_path: str = ""
    initial_prompt: str = ""
    analysis_prompt: str = DEFAULT_ANALYSIS_PROMPT
    rerun_prompt: str = ""
    final_prompt: str = ""
    exit_condition_prompt: str = DEFAULT_EXIT_CONDITION_PROMPT
    min_iterations: int = 2
    max_iterations: int = 10
    transition_iteration: int = 0  # 0 = disabled (two-prompt mode)
    soft_timeout_sec: int = 120    # 2 min -> stall warning
    hard_timeout_sec: int = 300    # 5 min -> cancel + retry
    autocompact_pct: int = 60     # Auto-compact at this % of context window
    max_error_retries: int = 5     # was hardcoded 3
    max_rate_limit_retries: int = 10  # was hardcoded 5
    verification_prompt: str = ""   # User-specified what to verify (empty = disabled)
    verification_interval: int = 0  # 0 = disabled, N = every Nth iteration is verification

    def validate(self) -> list[str]:
        """Return list of validation errors, empty if valid."""
        errors: list[str] = []
        if not self.project_path:
            errors.append("Project path is required")
        elif not Path(self.project_path).expanduser().is_dir():
            errors.append(f"Project path does not exist: {self.project_path}")
        if not self.initial_prompt.strip():
            errors.append("Initial prompt is required")
        if not self.rerun_prompt.strip():
            errors.append("Re-run prompt is required")
        if not self.analysis_prompt.strip():
            errors.append("Analysis prompt is required")
        if not self.exit_condition_prompt.strip():
            errors.append("Exit condition prompt is required")
        if self.min_iterations < 1:
            errors.append("Min iterations must be >= 1")
        if self.max_iterations < 1:
            errors.append("Max iterations must be >= 1")
        if self.min_iterations > self.max_iterations:
            errors.append("Min iterations must be <= max iterations")
        if not (1 <= self.autocompact_pct <= 100):
            errors.append("Auto-compact percentage must be between 1 and 100")
        # Three-prompt validation
        if self.transition_iteration > 0:
            if not self.final_prompt.strip():
                errors.append("Final prompt is required when transition iteration is set")
            if self.transition_iteration <= 1:
                errors.append("Transition iteration must be > 1 (iteration 1 is always initial prompt)")
            if self.transition_iteration > self.max_iterations:
                errors.append("Transition iteration must be <= max iterations")
        # Verification validation
        if self.verification_interval > 0 and not self.verification_prompt.strip():
            errors.append("Verification prompt is required when verification interval is set")
        if self.verification_interval < 0:
            errors.append("Verification interval must be >= 0")
        return errors

    def save_yaml(self, path: str | Path) -> None:
        """Save config to a YAML file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(asdict(self), f, default_flow_style=False, sort_keys=False)

    @classmethod
    def load_yaml(cls, path: str | Path) -> RalphConfig:
        """Load config from a YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        valid_fields = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered)
