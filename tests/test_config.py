"""Behavioral tests for RalphConfig."""

import tempfile
from pathlib import Path

import pytest
import yaml

from ralph_tui.config import RalphConfig


def _valid_config(tmp_path: Path, **overrides) -> RalphConfig:
    """Return a RalphConfig with all required fields pointing at tmp_path."""
    defaults = dict(
        project_path=str(tmp_path),
        initial_prompt="Do something useful",
        rerun_prompt="Keep going",
        min_iterations=2,
        max_iterations=5,
    )
    defaults.update(overrides)
    return RalphConfig(**defaults)


class TestYamlRoundtrip:
    def test_yaml_roundtrip_preserves_all_fields(self, tmp_path):
        cfg = _valid_config(tmp_path)
        yaml_path = tmp_path / "config.yaml"
        cfg.save_yaml(yaml_path)
        loaded = RalphConfig.load_yaml(yaml_path)
        assert loaded.project_path == cfg.project_path
        assert loaded.initial_prompt == cfg.initial_prompt
        assert loaded.rerun_prompt == cfg.rerun_prompt
        assert loaded.analysis_prompt == cfg.analysis_prompt
        assert loaded.exit_condition_prompt == cfg.exit_condition_prompt
        assert loaded.min_iterations == cfg.min_iterations
        assert loaded.max_iterations == cfg.max_iterations


class TestValidation:
    def test_validates_missing_project_path(self):
        cfg = RalphConfig(
            project_path="",
            initial_prompt="go",
            rerun_prompt="again",
        )
        errors = cfg.validate()
        assert any("project path" in e.lower() for e in errors)

    def test_validates_nonexistent_path(self):
        cfg = RalphConfig(
            project_path="/no/such/directory/ever",
            initial_prompt="go",
            rerun_prompt="again",
        )
        errors = cfg.validate()
        assert any("does not exist" in e.lower() for e in errors)

    def test_validates_min_greater_than_max(self, tmp_path):
        cfg = _valid_config(tmp_path)
        cfg.min_iterations = 5
        cfg.max_iterations = 3
        errors = cfg.validate()
        assert any("min" in e.lower() and "max" in e.lower() for e in errors)

    def test_validates_empty_initial_prompt(self, tmp_path):
        cfg = _valid_config(tmp_path)
        cfg.initial_prompt = "   "
        errors = cfg.validate()
        assert any("initial prompt" in e.lower() for e in errors)


class TestThreePromptConfig:
    """Tests for the three-prompt system: initial, rerun, and final prompts."""

    def test_final_prompt_field_exists_with_default_empty(self, tmp_path):
        """RalphConfig should have a final_prompt field that defaults to empty."""
        cfg = _valid_config(tmp_path)
        assert hasattr(cfg, "final_prompt")
        assert cfg.final_prompt == ""

    def test_transition_iteration_field_exists_with_default_zero(self, tmp_path):
        """RalphConfig should have a transition_iteration field (0 = disabled)."""
        cfg = _valid_config(tmp_path)
        assert hasattr(cfg, "transition_iteration")
        assert cfg.transition_iteration == 0

    def test_yaml_roundtrip_preserves_three_prompt_fields(self, tmp_path):
        """YAML save/load should preserve final_prompt and transition_iteration."""
        cfg = _valid_config(
            tmp_path,
            final_prompt="Wrap it up",
            transition_iteration=4,
        )
        yaml_path = tmp_path / "config.yaml"
        cfg.save_yaml(yaml_path)
        loaded = RalphConfig.load_yaml(yaml_path)
        assert loaded.final_prompt == "Wrap it up"
        assert loaded.transition_iteration == 4

    def test_validates_final_prompt_required_when_transition_set(self, tmp_path):
        """If transition_iteration > 0, final_prompt must not be empty."""
        cfg = _valid_config(tmp_path, transition_iteration=3, final_prompt="")
        errors = cfg.validate()
        assert any("final prompt" in e.lower() for e in errors)

    def test_validates_transition_must_be_greater_than_1(self, tmp_path):
        """transition_iteration must be > 1 (iteration 1 is always initial_prompt)."""
        cfg = _valid_config(
            tmp_path,
            transition_iteration=1,
            final_prompt="Wrap up",
        )
        errors = cfg.validate()
        assert any("transition" in e.lower() for e in errors)

    def test_validates_transition_within_max_iterations(self, tmp_path):
        """transition_iteration must be <= max_iterations."""
        cfg = _valid_config(
            tmp_path,
            max_iterations=5,
            transition_iteration=6,
            final_prompt="Wrap up",
        )
        errors = cfg.validate()
        assert any("transition" in e.lower() for e in errors)

    def test_no_validation_error_when_three_prompt_disabled(self, tmp_path):
        """When transition_iteration=0 and final_prompt='', no errors related to three-prompt."""
        cfg = _valid_config(tmp_path)
        errors = cfg.validate()
        # Should have no errors at all (it's a valid config)
        assert len(errors) == 0

    def test_valid_three_prompt_config_passes_validation(self, tmp_path):
        """A properly configured three-prompt config should validate without errors."""
        cfg = _valid_config(
            tmp_path,
            final_prompt="Final phase: clean up and summarize",
            transition_iteration=4,
            max_iterations=5,
        )
        errors = cfg.validate()
        assert len(errors) == 0


class TestRetryLimitConfig:
    """Tests for configurable retry limits."""

    def test_default_retry_limits(self, tmp_path):
        cfg = _valid_config(tmp_path)
        assert cfg.max_error_retries == 5
        assert cfg.max_rate_limit_retries == 10

    def test_custom_retry_limits_roundtrip(self, tmp_path):
        cfg = _valid_config(tmp_path, max_error_retries=8, max_rate_limit_retries=15)
        yaml_path = tmp_path / "config.yaml"
        cfg.save_yaml(yaml_path)
        loaded = RalphConfig.load_yaml(yaml_path)
        assert loaded.max_error_retries == 8
        assert loaded.max_rate_limit_retries == 15


class TestVerificationConfig:
    """Tests for blind verification system config fields."""

    def test_verification_config_fields_default_disabled(self, tmp_path):
        cfg = _valid_config(tmp_path)
        assert cfg.verification_prompt == ""
        assert cfg.verification_interval == 0

    def test_verification_config_requires_prompt_when_interval_set(self, tmp_path):
        cfg = _valid_config(tmp_path, verification_interval=3, verification_prompt="")
        errors = cfg.validate()
        assert any("verification prompt" in e.lower() for e in errors)

    def test_verification_config_valid_when_both_set(self, tmp_path):
        cfg = _valid_config(tmp_path, verification_interval=3, verification_prompt="Check citations")
        errors = cfg.validate()
        assert not any("verification" in e.lower() for e in errors)

    def test_verification_config_no_error_when_disabled(self, tmp_path):
        cfg = _valid_config(tmp_path, verification_interval=0, verification_prompt="")
        errors = cfg.validate()
        assert not any("verification" in e.lower() for e in errors)

    def test_verification_roundtrip_yaml(self, tmp_path):
        cfg = _valid_config(tmp_path, verification_interval=3, verification_prompt="Check facts")
        yaml_path = tmp_path / "config.yaml"
        cfg.save_yaml(yaml_path)
        loaded = RalphConfig.load_yaml(yaml_path)
        assert loaded.verification_interval == 3
        assert loaded.verification_prompt == "Check facts"


class TestClaudeMdTemplate:
    """Tests for enhanced CLAUDE.md template content."""

    def test_claude_md_template_contains_startup_sequence(self):
        from ralph_tui.config import CLAUDE_MD_TEMPLATE
        assert "Startup Sequence" in CLAUDE_MD_TEMPLATE
        assert "_ralph_state.json" in CLAUDE_MD_TEMPLATE

    def test_claude_md_template_references_state_file(self):
        from ralph_tui.config import CLAUDE_MD_TEMPLATE
        assert "_ralph_state.json" in CLAUDE_MD_TEMPLATE

    def test_claude_md_template_references_document_index(self):
        from ralph_tui.config import CLAUDE_MD_TEMPLATE
        assert "_document_index.md" in CLAUDE_MD_TEMPLATE

    def test_claude_md_template_no_hard_tool_ban(self):
        from ralph_tui.config import CLAUDE_MD_TEMPLATE
        assert "Do NOT use the Task tool" not in CLAUDE_MD_TEMPLATE

    def test_claude_md_template_has_citation_discipline(self):
        from ralph_tui.config import CLAUDE_MD_TEMPLATE
        assert "Citation Discipline" in CLAUDE_MD_TEMPLATE


class TestLegacyYamlLoading:
    def test_loads_yaml_ignoring_removed_fields(self, tmp_path):
        """YAML files with old fields (claude_model, allowed_tools, etc.)
        should load without error — unknown keys are silently dropped."""
        yaml_path = tmp_path / "legacy.yaml"
        data = {
            "project_path": str(tmp_path),
            "initial_prompt": "go",
            "rerun_prompt": "again",
            "min_iterations": 2,
            "max_iterations": 5,
            # These fields no longer exist on RalphConfig:
            "claude_model": "claude-opus-4-6",
            "allowed_tools": "Read,Edit,Bash",
            "max_turns": 50,
            "permission_mode": "bypassPermissions",
        }
        with open(yaml_path, "w") as f:
            yaml.dump(data, f)

        cfg = RalphConfig.load_yaml(yaml_path)
        assert cfg.project_path == str(tmp_path)
        assert cfg.initial_prompt == "go"
        # Confirm the removed fields didn't sneak onto the object
        assert not hasattr(cfg, "claude_model")
        assert not hasattr(cfg, "allowed_tools")
        assert not hasattr(cfg, "max_turns")
        assert not hasattr(cfg, "permission_mode")
