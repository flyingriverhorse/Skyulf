"""Scoring rules for the fare model: every input row is scored; no extra columns.

SCORING_MODE="pre_split" reuses the saved config/pre_split.yml recipe, which is
empty here. See the generated SCORING.md for custom eligibility and outputs.
"""

SCORING_MODE = "pre_split"  # Options: pre_split, custom, combined
SKIP_TARGET_PRE_SPLIT_STEPS = False


def build_model_rules():
    """Reuse the training pre-split recipe for scoring rows."""
    return {"reuse_pre_split": True, "skip_target_steps": SKIP_TARGET_PRE_SPLIT_STEPS}


build_scoring = build_model_rules


def build_combined_rules():
    """A single model has no cross-model results."""
    return []
