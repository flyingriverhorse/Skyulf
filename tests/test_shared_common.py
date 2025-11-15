from core.feature_engineering.modeling.shared import _resolve_problem_type_hint


def test_resolve_problem_type_hint_handles_user_input():
    assert _resolve_problem_type_hint("Classification", "regression") == "classification"
    assert _resolve_problem_type_hint(" regression ", "classification") == "regression"
    assert _resolve_problem_type_hint(None, "regression") == "regression"
