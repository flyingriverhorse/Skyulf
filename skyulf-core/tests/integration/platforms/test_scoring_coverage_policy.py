"""Coverage guards compare common predictions without hiding rejected inputs."""

from dataclasses import replace
from typing import Any

import numpy as np
import pandas as pd
import polars as pl
import pytest

pytest.importorskip("mlflow")

from skyulf.data.dataset import SplitDataset
from skyulf.inference.fitted_pipeline import load_pipeline, save_pipeline
from skyulf.integrations.mlflow.lifecycle import validation
from skyulf.integrations.mlflow.registration.registry import ResolvedModel
from skyulf.pipeline import SkyulfPipeline


def _model(path, lower=None, upper=None, *, offset=0.0, engine="pandas"):
    """Fit a real deterministic model with an independently selectable scoring population."""
    x = np.arange(100, dtype=float)
    frame = pd.DataFrame({"x": x, "target": 2 * x + offset})
    steps = (
        []
        if lower is None and upper is None
        else [
            {
                "name": "eligible",
                "transformer": "ManualBounds",
                "params": {"bounds": {"x": {"lower": lower, "upper": upper}}},
            }
        ]
    )
    pipeline = SkyulfPipeline({"preprocessing": steps, "modeling": {"type": "linear_regression"}})
    if engine == "polars":
        frame = pl.from_pandas(frame)
    pipeline.fit(SplitDataset(train=frame, test=frame.head(0)), target_column="target")
    save_pipeline(pipeline, path)
    return load_pipeline(path)


def _compare(monkeypatch, candidate, champion=None, **options):
    """Use actual fitted artifacts while replacing only the registry download boundary."""
    artifacts = {"2": candidate, "1": champion}
    monkeypatch.setattr(
        validation,
        "load_registered_pipeline",
        lambda resolved, **kwargs: artifacts[resolved.version],
    )

    def reference(artifact, version):
        """Bind concrete bytes to the same registered model name."""
        return ResolvedModel(
            name="workspace.test.coverage",
            version=version,
            model_uri=f"models:/workspace.test.coverage/{version}",
            signature=None,
            digest=artifact.manifest.pipeline_sha256,
        )

    frame = pd.DataFrame(
        {"x": np.arange(100, dtype=float), "target": 2 * np.arange(100, dtype=float)},
        index=[7] * 100,
    )
    if candidate.manifest.fitted_engine == "polars":
        frame = pl.from_pandas(frame)
    return validation.compare_registered_pipeline_models(
        reference(candidate, "2"),
        reference(champion, "1") if champion else None,
        frame,
        target_column="target",
        dataset_id="holdout@1",
        metric="heldout_rmse",
        min_improvement=0.0,
        max_rows=100,
        max_bytes=100_000,
        quality_threshold=1.0,
        **options,
    )


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_different_masks_compare_common_rows_but_keep_own_coverage(tmp_path, monkeypatch, engine):
    """A changed fitted filter can be compared fairly without using its easier subset alone."""
    candidate = _model(tmp_path / "candidate", 5, 99, engine=engine)
    champion = _model(tmp_path / "champion", 0, 94, offset=2, engine=engine)
    report = _compare(monkeypatch, candidate, champion)
    assert report.eligible
    assert report.candidate_metrics["heldout_scored_rows"] == 95
    assert report.champion_metrics["heldout_scored_rows"] == 95
    assert report.candidate_metrics["heldout_scored_fraction"] == 0.95
    assert report.candidate_metrics["heldout_comparison_rows"] == 90
    assert report.champion_metrics["heldout_comparison_rows"] == 90
    assert report.improvement == pytest.approx(2.0)


@pytest.mark.parametrize("rows,passed", [(89, False), (90, True), (100, True)])
def test_first_candidate_coverage_default_is_ninety_percent(tmp_path, monkeypatch, rows, passed):
    """The bootstrap quality path must reject a high-quality model that scores too little."""
    candidate = _model(tmp_path / "candidate", 0, rows - 1)
    report = _compare(monkeypatch, candidate)
    assert validation.quality_gates_pass(report) is passed
    gate = next(
        item
        for item in validation.quality_gate_results(report)
        if item["metric"] == "heldout_scored_fraction"
    )
    assert gate["threshold"] == 0.90
    assert gate["value"] == rows / 100
    assert report.reason == ("no_champion" if passed else "quality_gate_failed")


def test_coverage_override_and_explicit_quality_gate(tmp_path, monkeypatch):
    """The per-model minimum and explicit quality guard resolve to one strongest bound."""
    candidate = _model(tmp_path / "candidate", 0, 84)
    report = _compare(
        monkeypatch,
        candidate,
        min_scored_fraction=0.85,
        quality_gates={"heldout_scored_fraction": 0.85},
    )
    assert validation.quality_gates_pass(report)
    assert report.quality_gates == {"heldout_scored_fraction": 0.85}


@pytest.mark.parametrize("rows,passed", [(90, True), (89, False)])
def test_maximum_coverage_drop_is_five_percentage_points(tmp_path, monkeypatch, rows, passed):
    """Equal-to-limit coverage passes while a larger loss cannot be hidden by better RMSE."""
    candidate = _model(tmp_path / "candidate", 0, rows - 1)
    champion = _model(tmp_path / "champion", 0, 94, offset=2)
    report = _compare(monkeypatch, candidate, champion, min_scored_fraction=0.8)
    assert report.eligible is passed
    assert validation.quality_gates_pass(report) is passed
    assert report.max_coverage_drop == 0.05
    assert report.reason == ("candidate_improved" if passed else "coverage_drop_exceeded")


@pytest.mark.parametrize(
    "field,value",
    [
        ("min_scored_fraction", -0.01),
        ("min_scored_fraction", 1.01),
        ("min_scored_fraction", True),
        ("min_scored_fraction", float("nan")),
        ("max_coverage_drop", -0.01),
        ("max_coverage_drop", 1.01),
        ("max_coverage_drop", True),
        ("max_coverage_drop", float("nan")),
        ("heldout_scored_fraction", -0.01),
        ("heldout_scored_fraction", 1.01),
        ("heldout_scored_fraction", True),
        ("heldout_scored_fraction", float("inf")),
    ],
)
def test_coverage_policy_rejects_invalid_values(tmp_path, monkeypatch, field, value):
    """Policy values cannot weaken a saved decision through coercion or non-finite numbers."""
    candidate = _model(tmp_path / "candidate", 0, 99)
    options = (
        {field: value}
        if field in {"min_scored_fraction", "max_coverage_drop"}
        else {"quality_gates": {field: value}}
    )
    with pytest.raises(ValueError, match="coverage|scored_fraction"):
        _compare(monkeypatch, candidate, **options)


@pytest.mark.parametrize("upper", [49, 50])
def test_zero_or_one_common_prediction_cannot_produce_comparison(tmp_path, monkeypatch, upper):
    """A comparison requires at least two common predicted rows even with relaxed coverage."""
    candidate = _model(tmp_path / "candidate", 0, upper)
    champion = _model(tmp_path / "champion", 50, 99, offset=2)
    with pytest.raises(ValueError, match="at least two.*common|common.*at least two"):
        _compare(
            monkeypatch,
            candidate,
            champion,
            quality_gates={"heldout_scored_fraction": 0.0},
            max_coverage_drop=1.0,
        )


def test_comparison_digest_binds_nondefault_coverage_drop(tmp_path, monkeypatch):
    """Fresh promotion replay cannot substitute a more permissive policy unnoticed."""
    candidate = _model(tmp_path / "candidate", 0, 99)
    report = _compare(monkeypatch, candidate, max_coverage_drop=0.02)
    modified = replace(report, max_coverage_drop=0.5)
    assert validation.comparison_digest(report) != validation.comparison_digest(modified)
    assert validation.comparison_payload(report)["max_coverage_drop"] == 0.02


def test_predictions_are_reused_after_intersection(tmp_path, monkeypatch):
    """A scoring callback must see the raw holdout once, never a reduced replay."""
    from skyulf.inference import pipeline_evaluation

    candidate = _model(tmp_path / "candidate", 5, 99)
    champion = _model(tmp_path / "champion", 0, 94, offset=2)
    original = pipeline_evaluation.predict_pipeline
    calls = []

    def observe(frame, artifact):
        """Record actual full-frame prediction without changing its result."""
        calls.append((artifact.manifest.pipeline_sha256, len(frame)))
        return original(frame, artifact)

    monkeypatch.setattr(pipeline_evaluation, "predict_pipeline", observe)
    report = _compare(monkeypatch, candidate, champion)
    assert calls == [
        (candidate.manifest.pipeline_sha256, 100),
        (champion.manifest.pipeline_sha256, 100),
    ]
    assert report.candidate_metrics["heldout_comparison_rows"] == 90


@pytest.mark.parametrize("rows", [89, 90])
def test_first_champion_replays_coverage_before_alias_write(tmp_path, monkeypatch, rows):
    """The real registry alias can initialize only after the fresh coverage check passes."""
    import mlflow

    from skyulf.integrations.mlflow.lifecycle import promotion

    artifact = _model(tmp_path / "candidate", 0, rows - 1)
    report = _compare(monkeypatch, artifact, max_coverage_drop=0.02)
    uri = f"sqlite:///{(tmp_path / 'registry.db').as_posix()}"
    client = mlflow.MlflowClient(tracking_uri=uri, registry_uri=uri)
    client.create_registered_model(report.model_name)
    for _ in range(2):
        client.create_model_version(report.model_name, (tmp_path / "candidate").as_uri())
    reference = ResolvedModel(
        name=report.model_name,
        version="2",
        model_uri=f"models:/{report.model_name}/2",
        signature=None,
        digest=artifact.manifest.pipeline_sha256,
    )
    monkeypatch.setattr(promotion, "resolve_model", lambda *a, **k: reference)
    frame = pd.DataFrame(
        {"x": np.arange(100, dtype=float), "target": 2 * np.arange(100, dtype=float)},
        index=[7] * 100,
    )
    options: dict[str, Any] = {
        "target_column": "target",
        "admission": promotion.LocalAliasAdmission(tmp_path / "locks"),
        "max_rows": 100,
        "max_bytes": 100_000,
        "tracking_uri": uri,
        "registry_uri": uri,
    }
    if rows == 89:
        with pytest.raises(ValueError, match="absolute quality"):
            promotion.initialize_champion(report, frame, **options)
        assert "champion" not in client.get_registered_model(report.model_name).aliases
    else:
        receipt = promotion.initialize_champion(report, frame, **options)
        assert receipt.comparison_sha256 == validation.comparison_digest(report)
        assert str(client.get_model_version_by_alias(report.model_name, "champion").version) == "2"


def test_manual_replay_preserves_coverage_policy_and_rejects_tampering(tmp_path, monkeypatch):
    """Manual approval must recompute the pinned policy and all own coverage counts."""
    from skyulf.integrations.mlflow.lifecycle import promotion

    candidate = _model(tmp_path / "candidate", 0, 89)
    champion = _model(tmp_path / "champion", 0, 94, offset=2)
    report = _compare(monkeypatch, candidate, champion, max_coverage_drop=0.07)
    artifacts = {"2": candidate, "1": champion}
    monkeypatch.setattr(
        promotion,
        "resolve_model",
        lambda name, version, **k: ResolvedModel(
            name=name,
            version=version,
            model_uri=f"models:/{name}/{version}",
            signature=None,
            digest=artifacts[version].manifest.pipeline_sha256,
        ),
    )
    frame = pd.DataFrame(
        {"x": np.arange(100, dtype=float), "target": 2 * np.arange(100, dtype=float)},
        index=[7] * 100,
    )
    options: dict[str, Any] = {
        "target_column": "target",
        "expected_champion_version": "1",
        "max_rows": 100,
        "max_bytes": 100_000,
        "tracking_uri": None,
        "registry_uri": None,
    }
    assert promotion._validated_report(report, frame, **options) == validation.comparison_digest(
        report
    )
    stale = replace(
        report, candidate_metrics={**report.candidate_metrics, "heldout_scored_rows": 100}
    )
    with pytest.raises(ValueError, match="no longer matches"):
        promotion._validated_report(stale, frame, **options)
    with pytest.raises(ValueError, match="no longer matches"):
        promotion._validated_report(replace(report, max_coverage_drop=0.04), frame, **options)


def test_approval_and_rollout_bind_coverage_policy(tmp_path, monkeypatch):
    """A later operator configuration cannot silently relax the saved coverage margin."""
    from skyulf.integrations.databricks.lifecycle.approval import _validate_approval_policy
    from skyulf.integrations.databricks.serving.rollout_promotion import approval_config_digest

    report = _compare(monkeypatch, _model(tmp_path / "candidate", 0, 99), max_coverage_drop=0.02)
    config = {
        "metric": report.metric,
        "min_improvement": report.min_improvement,
        "quality_threshold": report.quality_threshold,
        "max_coverage_drop": 0.02,
    }
    _validate_approval_policy(config, report, None)
    changed = {**config, "max_coverage_drop": 0.5}
    assert approval_config_digest(config) != approval_config_digest(changed)
    with pytest.raises(ValueError, match="Approval policy"):
        _validate_approval_policy(changed, report, None)


def test_job_report_shows_own_coverage_and_drop_gate(tmp_path, monkeypatch):
    """Operator output shows why a good RMSE did not justify losing coverage."""
    from skyulf.integrations.databricks.jobs.shared.job_output import _comparison_summary

    candidate = _model(tmp_path / "candidate", 0, 89)
    champion = _model(tmp_path / "champion", 0, 99, offset=2)
    report = _compare(monkeypatch, candidate, champion)
    output = "".join(_comparison_summary({"comparison": validation.comparison_payload(report)}))
    assert "heldout_scored_fraction" in output
    assert "heldout_comparison_rows" in output
    assert "heldout_coverage_drop" in output
    assert "Failed: threshold not met" in output


@pytest.mark.parametrize(
    "minimum,maximum,passed",
    [(None, None, True), (None, 0.05, False), (0.90, None, False), (0.80, 0.20, True)],
)
def test_optional_coverage_rules_are_independently_configurable(
    tmp_path, monkeypatch, minimum, maximum, passed
):
    """Null disables only its own rule; saved reports retain both decisions for replay."""
    candidate = _model(tmp_path / "candidate", 0, 84)
    champion = _model(tmp_path / "champion", 0, 99, offset=2)
    report = _compare(
        monkeypatch, candidate, champion, min_scored_fraction=minimum, max_coverage_drop=maximum
    )
    assert report.eligible is passed
    assert validation.quality_gates_pass(report) is passed
    payload = validation.comparison_payload(report)
    assert payload["min_scored_fraction"] == minimum
    assert payload["max_coverage_drop"] == maximum
    assert validation.ModelComparisonReport(**payload) == report


@pytest.mark.parametrize("minimum", [None, 0.80, 0.95])
def test_explicit_coverage_quality_gate_remains_independent(tmp_path, monkeypatch, minimum):
    """Disabling the automatic minimum cannot disable a user supplied quality gate."""
    report = _compare(
        monkeypatch,
        _model(tmp_path / "candidate", 0, 89),
        min_scored_fraction=minimum,
        quality_gates={"heldout_scored_fraction": 0.92},
    )
    gates = [
        gate
        for gate in validation.quality_gate_results(report)
        if gate["metric"] == "heldout_scored_fraction"
    ]
    assert len(gates) == 1
    assert gates[0]["threshold"] == max(minimum or 0, 0.92)
    assert not gates[0]["passed"]


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_project_eligibility_coverage_is_measured_on_full_holdout(tmp_path, monkeypatch, engine):
    """Saved project rules share the same coverage policy as fitted exclusion steps."""
    frame = pd.DataFrame(
        {"x": np.arange(100, dtype=float), "target": 2 * np.arange(100, dtype=float)}
    )
    pipeline = SkyulfPipeline(
        {
            "preprocessing": [],
            "modeling": {"type": "linear_regression"},
            "project_python_source": "import pandas as pd\ndef eligible(frame, params):\n    return pd.Series(['policy' if x > 88 else None for x in frame.x], index=frame.index)\n",
            "project_scoring": {
                "eligibility": [
                    {"name": "allowed", "version": "1", "function": "eligible", "params": {}}
                ],
                "outputs": [],
            },
        }
    )
    if engine == "polars":
        frame = pl.from_pandas(frame)
    pipeline.fit(SplitDataset(train=frame, test=frame.head(0)), target_column="target")
    save_pipeline(pipeline, tmp_path / "artifact")
    artifact = load_pipeline(tmp_path / "artifact")
    report = _compare(monkeypatch, artifact)
    assert report.candidate_metrics["heldout_scored_rows"] == 89
    assert report.candidate_metrics["heldout_scored_fraction"] == 0.89
    assert not validation.quality_gates_pass(report)


@pytest.mark.parametrize("candidate_filters", [True, False])
def test_filtered_and_unfiltered_models_share_comparable_coverage(
    tmp_path, monkeypatch, candidate_filters
):
    """Introducing or removing a fitted exclusion step remains compatible with old plain models."""
    candidate = _model(
        tmp_path / "candidate", 0 if candidate_filters else None, 94 if candidate_filters else None
    )
    champion = _model(
        tmp_path / "champion",
        None if candidate_filters else 0,
        None if candidate_filters else 94,
        offset=2,
    )
    report = _compare(monkeypatch, candidate, champion)
    assert report.eligible
    assert report.candidate_metrics["heldout_scored_rows"] == (95 if candidate_filters else 100)
    assert report.champion_metrics["heldout_scored_rows"] == (100 if candidate_filters else 95)
    assert report.candidate_metrics["heldout_comparison_rows"] == 95


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_power_invalids_count_toward_minimum_coverage(tmp_path, engine):
    """Invalid transform inputs cannot disappear from the model's coverage denominator."""
    from skyulf.inference.pipeline_evaluation import evaluate_holdout

    x = np.arange(1, 101, dtype=float)
    frame = pd.DataFrame({"x": x, "target": 2 * x})
    pipeline = SkyulfPipeline(
        {
            "preprocessing": [
                {
                    "name": "power",
                    "transformer": "PowerTransformer",
                    "params": {"columns": ["x"], "method": "box-cox"},
                }
            ],
            "modeling": {"type": "linear_regression"},
        }
    )
    if engine == "polars":
        frame = pl.from_pandas(frame)
    pipeline.fit(SplitDataset(train=frame, test=frame.head(0)), target_column="target")
    save_pipeline(pipeline, tmp_path / "artifact")
    artifact = load_pipeline(tmp_path / "artifact")
    raw = np.arange(-10, 90, dtype=float)
    holdout = pd.DataFrame({"x": raw, "target": 2 * raw}, index=[7] * 100)
    if engine == "polars":
        holdout = pl.from_pandas(holdout)
    metrics = evaluate_holdout(artifact, holdout, target_column="target")
    gates = validation.evaluate_quality_gates(metrics, "heldout_rmse", None)
    assert metrics["heldout_total_rows"] == 100
    assert metrics["heldout_scored_rows"] == 89
    assert metrics["heldout_scored_fraction"] == 0.89
    assert gates[0]["metric"] == "heldout_scored_fraction"
    assert not gates[0]["passed"]
