"""Reuse distance eligibility on Spark and endpoints; requests need no actual fare."""

SKIP_TARGET_PRE_SPLIT_STEPS = True


def build_model_rules():
    """Skip the fare target filter and preserve failed distance checks as excluded rows."""
    return {"reuse_pre_split": True, "skip_target_steps": SKIP_TARGET_PRE_SPLIT_STEPS}


def build_combined_rules():
    """A single model has no cross-model outputs."""
    return []
