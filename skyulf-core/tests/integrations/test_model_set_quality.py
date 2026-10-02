"""Set activation requires every saved component quality policy to pass."""

from dataclasses import replace
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import Mock

import pytest

from skyulf.integrations.mlflow.validation import ModelComparisonReport


def _report(**overrides):
    """Provide a pinned absolute-quality comparison without a previous champion."""
    values: dict[str, Any] = {
        "dataset_id": "data",
        "row_count": 4,
        "code_version": "0.9.1",
        "model_name": "a.b.model",
        "candidate_version": "2",
        "candidate_digest": "a" * 64,
        "champion_version": None,
        "champion_digest": None,
        "metric": "heldout_rmse",
        "metric_direction": "minimize",
        "min_improvement": 0.0,
        "quality_threshold": 2.0,
        "candidate_metrics": {"heldout_rmse": 1.0},
        "champion_metrics": None,
        "improvement": None,
        "eligible": False,
        "reason": "no_champion",
    }
    return ModelComparisonReport(**(values | overrides))


@pytest.mark.parametrize(
    "score,threshold,passed", [(1.0, 2.0, True), (3.0, 2.0, False), (1.0, None, False)]
)
def test_initial_set_requires_absolute_quality(score, threshold, passed):
    """A measured first model cannot become champion without an explicit passing floor."""
    from skyulf.integrations.databricks.model_set_quality import component_quality

    report = _report(candidate_metrics={"heldout_rmse": score}, quality_threshold=threshold)
    result = component_quality(report)
    assert result["passed"] is passed


@pytest.mark.parametrize(
    "improvement,eligible,passed",
    [
        (0.5, True, True),
        (0.01, False, True),
        (0.0, False, True),
        (-0.01, False, False),
        (None, False, False),
        (float("nan"), False, False),
        (float("inf"), False, False),
    ],
)
def test_replacement_requires_no_regression_and_additional_gates(improvement, eligible, passed):
    """Tied peers may join an improving set, but regression or invalid evidence cannot."""
    from skyulf.integrations.databricks.model_set_quality import component_quality

    report = _report(
        champion_version="1",
        champion_digest="b" * 64,
        eligible=eligible,
        improvement=improvement,
        reason="candidate_improved",
    )
    assert component_quality(report)["passed"] is passed
    assert component_quality(report)["improved"] is (passed and eligible)
    failing = replace(
        report,
        quality_gates={"heldout_r2": 0.9},
        candidate_metrics={"heldout_rmse": 1.0, "heldout_r2": 0.5},
    )
    assert not component_quality(failing)["passed"]


def test_quality_uses_set_components_and_all_failures(monkeypatch):
    """Mutable component champion aliases must never decide set replacement."""
    from skyulf.integrations.databricks import model_set_quality as module

    components = [SimpleNamespace(branch=name) for name in ("amount", "risk")]
    artifact = SimpleNamespace(
        manifest=SimpleNamespace(
            components=components,
            quality_evidence={
                "expected_champion_version": "7",
                "comparisons": {"amount": "a", "risk": "b"},
            },
        )
    )
    champion = SimpleNamespace(manifest=SimpleNamespace(components=components))
    evaluate = Mock(
        side_effect=[{"passed": True, "improved": True}, {"passed": False, "improved": False}]
    )
    monkeypatch.setattr(module, "_evaluate_component", evaluate)
    result = module.evaluate_model_set_quality(
        None,
        cast(Any, artifact),
        cast(Any, champion),
        expected_champion_version="7",
        max_rows=100,
        max_bytes=10000,
        tracking_uri="tracking",
        registry_uri="registry",
    )
    assert result["passed"] is False
    assert result["failed_components"] == ["risk"]
    assert evaluate.call_count == 2
    assert evaluate.call_args.args[3] is components[1]


@pytest.mark.parametrize(
    "revenue_gain,risk_gain,minimum,passed,reason",
    [
        (0.4, 0.0, 0.0, True, "set_improved"),
        (0.0, 0.1, 0.0, True, "set_improved"),
        (0.4, -0.01, 0.0, False, "component_quality_failed"),
        (-0.01, 0.1, 0.0, False, "component_quality_failed"),
        (0.0, 0.0, 0.0, False, "no_component_improved"),
        (0.1, 0.0, 0.2, False, "no_component_improved"),
        (0.25, 0.0, 0.25, True, "set_improved"),
    ],
)
def test_set_requires_one_meaningful_improvement_without_regression(
    monkeypatch, revenue_gain, risk_gain, minimum, passed, reason
):
    """Whole-set activation needs one eligible improvement and no worsening peer."""
    from skyulf.integrations.databricks import model_set_quality as module
    from skyulf.integrations.mlflow.validation import _comparison_decision, _metric_improvement

    results = []
    for metric, candidate, baseline, direction in (
        ("heldout_rmse", 1.0 - revenue_gain, 1.0, "minimize"),
        ("heldout_accuracy", 0.8 + risk_gain, 0.8, "maximize"),
    ):
        candidate_metrics = {metric: candidate}
        champion_metrics = {metric: baseline}
        gain = _metric_improvement(candidate_metrics, champion_metrics, metric)
        eligible, comparison_reason = _comparison_decision(gain, True, minimum)
        results.append(
            module.component_quality(
                _report(
                    champion_version="1",
                    champion_digest="b" * 64,
                    metric=metric,
                    metric_direction=direction,
                    quality_threshold=2.0 if direction == "minimize" else 0.5,
                    candidate_metrics=candidate_metrics,
                    champion_metrics=champion_metrics,
                    improvement=gain,
                    min_improvement=minimum,
                    eligible=eligible,
                    reason=comparison_reason,
                )
            )
        )
    components = [SimpleNamespace(branch=name) for name in ("revenue", "risk")]
    artifact = SimpleNamespace(
        manifest=SimpleNamespace(
            components=components,
            quality_evidence={
                "expected_champion_version": "1",
                "comparisons": {"revenue": "a", "risk": "b"},
            },
        )
    )
    champion = SimpleNamespace(manifest=SimpleNamespace(components=components))
    monkeypatch.setattr(module, "_evaluate_component", Mock(side_effect=results))
    decision = module.evaluate_model_set_quality(
        None,
        cast(Any, artifact),
        cast(Any, champion),
        expected_champion_version="1",
        max_rows=100,
        max_bytes=10000,
    )
    assert decision["passed"] is passed
    assert decision["reason"] == reason
    assert decision["improved_components"] == [
        name
        for name, result in zip(("revenue", "risk"), results, strict=True)
        if result["improved"]
    ]


def test_quality_rejects_stale_baseline_before_loading_models():
    """An approval against a newer set must require a fresh candidate comparison."""
    from skyulf.integrations.databricks.model_set_quality import evaluate_model_set_quality

    artifact = SimpleNamespace(
        manifest=SimpleNamespace(
            quality_evidence={"expected_champion_version": "7", "comparisons": {}}
        )
    )
    with pytest.raises(ValueError, match="baseline"):
        evaluate_model_set_quality(
            None,
            cast(Any, artifact),
            None,
            expected_champion_version="8",
            max_rows=10,
            max_bytes=1000,
        )


def test_quality_evidence_is_part_of_frozen_set_identity(tmp_path):
    """Changing a saved policy pin must change package identity and survive source removal."""
    from tests.integrations.test_model_set_batch import _saved_set

    from skyulf.inference.model_set import load_model_set, save_model_set

    original, _, _ = _saved_set(tmp_path / "original")
    components = {
        c.branch: (c.reference, original.directory / "components" / c.branch)
        for c in original.manifest.components
    }
    evidence = {
        "expected_champion_version": None,
        "comparisons": dict.fromkeys(components, "a" * 64),
    }
    saved = save_model_set(
        tmp_path / "quality",
        components,
        record_key_schema=original.manifest.record_key_schema,
        quality_evidence=evidence,
    )
    evidence["comparisons"].clear()
    restored = load_model_set(saved.directory)
    assert restored.manifest.quality_evidence is not None
    assert restored.manifest.quality_evidence["comparisons"]
    assert restored.manifest.set_sha256 != original.manifest.set_sha256


def test_automatic_failure_preserves_candidate_without_alias_change(monkeypatch):
    """A failed quality decision must be returned for inspection without an activation."""
    pytest.importorskip("mlflow")
    from skyulf.integrations.databricks import model_set_release as module

    decision = {"passed": False, "failed_components": ["risk"]}
    approve = Mock(side_effect=module.ModelSetQualityError(decision))
    monkeypatch.setattr(module, "approve_project_model_set", approve)
    nominate = Mock()
    monkeypatch.setattr(module, "nominate_model_set", nominate)
    result = module.automatic_model_set_release(
        None,
        cast(Any, "candidate"),
        {"promotion_policy": "automatic", "expected_champion_version": "3"},
        {},
    )
    assert result["alias_change"] is None and result["quality"] == decision
    assert approve.call_args.kwargs["expected_champion_version"] == "3"
    nominate.assert_called_once()


def test_automatic_policy_requires_all_thresholds_before_registry(monkeypatch):
    """A missing branch threshold must fail preflight rather than waste training work."""
    pytest.importorskip("mlflow")
    from skyulf.integrations.databricks import model_set_release as module

    read = Mock()
    monkeypatch.setattr(module, "controlled_champion_version", read)
    with pytest.raises(ValueError, match="quality_threshold.*risk"):
        module.pin_model_set_baseline({"promotion_policy": "automatic"}, {"risk": {}}, {})
    read.assert_not_called()


def test_set_baseline_overrides_independent_component_aliases(monkeypatch):
    """Set comparisons must carry the release's exact versions into branch training."""
    pytest.importorskip("mlflow")
    from skyulf.integrations.databricks import model_set_release as module

    champion = SimpleNamespace(
        manifest=SimpleNamespace(
            components=[
                SimpleNamespace(
                    branch="risk", reference=SimpleNamespace(name="a.b.risk", version="4")
                )
            ]
        )
    )
    monkeypatch.setattr(module, "controlled_champion_version", Mock(return_value="9"))
    monkeypatch.setattr(module, "champion_artifact", Mock(return_value=champion))
    settings, pins = module.pin_model_set_baseline(
        {"model_name": "a.b.set"}, {"risk": {"model_name": "a.b.risk"}}, {}
    )
    assert pins == {"risk": "4"}
    assert settings["expected_champion_version"] == "9"


def test_quality_bound_set_cannot_use_functional_only_approval():
    """Low-level approval must not bypass a newly packaged set's required quality proof."""
    pytest.importorskip("mlflow")
    from skyulf.integrations.mlflow.model_set_lifecycle import _quality_validation

    artifact = SimpleNamespace(manifest=SimpleNamespace(quality_evidence={"comparisons": {}}))
    with pytest.raises(ValueError, match="requires a quality validator"):
        _quality_validation(cast(Any, artifact), None, None)


@pytest.mark.parametrize("layout,visible", [("single_model", False), ("multi_target", True)])
def test_set_promotion_prompt_is_independent_of_branch_policies(layout, visible):
    """Only set layouts should offer whole-set automatic activation in Bundle setup."""
    import json
    from pathlib import Path

    from jsonschema import Draft7Validator

    root = Path(__file__).resolve().parents[2]
    schema = json.loads((root / "templates/databricks/databricks_template_schema.json").read_text())
    field = schema["properties"]["model_set_promotion_policy"]
    assert (
        not Draft7Validator(field["skip_prompt_if"]).is_valid({"training_layout": layout})
    ) is visible
    assert field["default"] == "manual_approval"


@pytest.mark.parametrize(
    "result",
    [
        None,
        {"passed": False},
        {"passed": True, "expected_champion_version": "9", "components": {}},
        {"passed": True, "expected_champion_version": None, "components": {}},
        {
            "passed": True,
            "expected_champion_version": None,
            "components": {"risk": {"passed": False}},
        },
    ],
)
def test_quality_proof_cannot_omit_or_contradict_a_component(result):
    """Approval and rollback must reject incomplete or contradictory saved decisions."""
    pytest.importorskip("mlflow")
    from skyulf.integrations.mlflow.model_set_lifecycle import _validate_quality_result

    artifact = SimpleNamespace(
        manifest=SimpleNamespace(components=[SimpleNamespace(branch="risk")])
    )
    with pytest.raises(ValueError, match="quality validation"):
        _validate_quality_result(result, cast(Any, artifact), None)


def test_failed_release_next_actions_require_new_training():
    """A failed candidate must not be presented as manually approvable without new evidence."""
    from skyulf.integrations.databricks.branch_notebook import set_next_actions

    actions = set_next_actions({"quality": {"passed": False}, "alias_change": None})
    assert any("train a new candidate" in action for action in actions)
    assert not any("Run approve" in action for action in actions)
