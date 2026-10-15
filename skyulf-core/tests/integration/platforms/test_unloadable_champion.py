"""A champion from another Core release must hold promotion until an explicit, gated override."""

import json
from dataclasses import replace

import pytest
from test_mlflow_promotion import (  # noqa: F401 - shared real-registry fixture
    _promote,
    _stage,
    case,
)

from skyulf.inference.fitted_pipeline import ArtifactRuntimeMismatchError
from skyulf.integrations.databricks.jobs.shared.job_runtime import operator_options
from skyulf.integrations.mlflow.lifecycle import validation
from skyulf.integrations.mlflow.lifecycle.validation import (
    compare_registered_pipeline_models,
    comparison_payload,
)
from skyulf.integrations.mlflow.registration.registry import resolve_model


@pytest.fixture
def old_champion(monkeypatch):
    """Make version 1 behave like a model pickled by a previous Skyulf release."""
    real = validation._checked_artifact

    def checked(reference, **kwargs):
        if reference.version == "1":
            raise ArtifactRuntimeMismatchError(
                "Local artifact runtime mismatch for skyulf-core: requires 0.9.3, found 0.9.4."
            )
        return real(reference, **kwargs)

    monkeypatch.setattr(validation, "_checked_artifact", checked)


def _unloadable_report(case, quality_threshold: float | None = 1.0):  # noqa: F811 - pytest fixture argument
    """Compare v2 with the unloadable v1 champion on the shared holdout."""
    _, uri, name, heldout, report, _ = case
    return compare_registered_pipeline_models(
        resolve_model(name, version="2", tracking_uri=uri, registry_uri=uri),
        resolve_model(name, version="1", tracking_uri=uri, registry_uri=uri),
        heldout,
        target_column="target",
        dataset_id=report.dataset_id,
        metric="heldout_mse",
        min_improvement=1.0,
        max_rows=10,
        max_bytes=10_000,
        tracking_uri=uri,
        registry_uri=uri,
        quality_threshold=quality_threshold,
    )


def test_comparison_reports_the_champion_without_measuring_it(case, old_champion):  # noqa: F811
    """Training must continue with an explicit reason instead of failing on the old pickle."""
    report = _unloadable_report(case)
    assert (report.eligible, report.reason) == (False, "champion_runtime_unavailable")
    assert report.champion_version == "1" and report.champion_metrics is None
    assert "requires 0.9.3" in report.champion_unavailable
    assert "champion_unavailable" in comparison_payload(report)


def test_ordinary_comparisons_keep_their_historical_digest(case):  # noqa: F811
    """Evidence written before this field existed must hash exactly as before."""
    _, _, _, _, report, _ = case
    assert report.champion_unavailable is None
    assert "champion_unavailable" not in comparison_payload(report)


def test_unloadable_candidate_still_fails(case, monkeypatch):  # noqa: F811
    """Only the champion may be reported as unloadable; a broken candidate stops the run."""
    real = validation._checked_artifact

    def checked(reference, **kwargs):
        if reference.version == "2":
            raise ArtifactRuntimeMismatchError("candidate runtime differs")
        return real(reference, **kwargs)

    monkeypatch.setattr(validation, "_checked_artifact", checked)
    with pytest.raises(ArtifactRuntimeMismatchError):
        _unloadable_report(case)


def test_promotion_needs_the_explicit_override(case, old_champion):  # noqa: F811
    """Without the flag the operator gets the fix; with it the candidate becomes champion."""
    report = _unloadable_report(case)
    with pytest.raises(ValueError, match="allow_unloadable_champion=true"):
        _promote(case, report=report)
    _stage(case, report=report)
    receipt = _promote(case, report=report, allow_unloadable_champion=True)
    assert (receipt.kind, receipt.prior_version, receipt.new_version) == ("promotion", "1", "2")


def test_override_requires_an_absolute_quality_bound(case, old_champion):  # noqa: F811
    """Skipping the champion comparison must never skip the candidate's own gate."""
    report = _unloadable_report(case, quality_threshold=None)
    with pytest.raises(ValueError, match="quality_threshold"):
        _promote(case, report=report, allow_unloadable_champion=True)


def test_override_is_rejected_when_the_champion_loads(case):  # noqa: F811
    """A loadable champion is compared normally; the flag cannot bypass that comparison."""
    _, _, _, _, report, _ = case
    with pytest.raises(ValueError, match="only when the champion cannot be loaded"):
        _promote(case, report=report, allow_unloadable_champion=True)
    stale = replace(report, eligible=False, reason="champion_runtime_unavailable")
    with pytest.raises(ValueError):
        _promote(case, report=stale, allow_unloadable_champion=True)


@pytest.mark.parametrize(
    "action, values, message",
    [
        ("approve", {"allow_unloadable_champion": "yes"}, "exactly true"),
        ("reject", {"allow_unloadable_champion": "true", "rejection_reason": "x"}, "Unexpected"),
        ("rollback", {"allow_unloadable_champion": "true"}, "Unexpected"),
    ],
)
def test_job_parameter_accepts_only_true_for_approve(action, values, message):
    """The override is an approve-only literal so a typo cannot silently enable it."""
    base = {"candidate_version": "2", "expected_champion_version": "1"}
    with pytest.raises(ValueError, match=message):
        operator_options(action, {**base, **values})
    assert operator_options("approve", {**base, "allow_unloadable_champion": "true"})[
        "allow_unloadable_champion"
    ]


@pytest.mark.parametrize("policy", ["manual_approval", "automatic"])
def test_report_offers_the_override_under_either_policy(case, old_champion, policy):  # noqa: F811
    """Automatic promotion holds, so the operator needs the override block in both policies."""
    from skyulf.integrations.databricks.lifecycle.workflow import next_actions
    from skyulf.integrations.databricks.training.fitting.candidate import CandidateResult

    report = _unloadable_report(case)
    candidate = CandidateResult(
        "run",
        report.model_name,
        "2",
        report.candidate_digest,
        report.dataset_id,
        10,
        5,
        0,
        "pandas",
        report,
        "0" * 64,
        "0" * 64,
    )
    actions = next_actions(candidate, policy)
    assert actions["approve"] == {
        "lifecycle_action": "approve",
        "candidate_version": "2",
        "expected_champion_version": "1",
        "allow_unloadable_champion": "true",
    }
    assert ("reject" in actions) == (policy == "manual_approval")


def _saved_proof(monkeypatch):
    """Serve one committed challenger receipt so only the policy guard decides."""
    from types import SimpleNamespace
    from unittest.mock import Mock

    from skyulf.integrations.databricks.lifecycle import approval

    event = {"k": "challenger", "s": "committed", "h": "b" * 64}
    client = Mock()
    client.get_model_version.return_value = SimpleNamespace(
        tags={"promotion_event": json.dumps(event)}
    )
    monkeypatch.setattr(approval, "controlled_champion_version", lambda *a, **kw: "1")
    monkeypatch.setattr(approval, "active_marker", lambda *a, **kw: "event")
    monkeypatch.setattr(approval, "require_mlflow", lambda: object())
    monkeypatch.setattr(approval, "make_registry_client", lambda *a: client)
    return approval


def test_override_finds_saved_proof_under_automatic_policy(monkeypatch):
    """The job's approve form must work without a digest when automatic promotion held."""
    approval = _saved_proof(monkeypatch)
    config = {"promotion_policy": "automatic", "model_name": "workspace.test.model"}
    with pytest.raises(ValueError, match="manual_approval"):
        approval.resolve_candidate_comparison_digest(config, "2", action="approve")
    with pytest.raises(ValueError, match="manual_approval"):
        approval.resolve_candidate_comparison_digest(
            config, "2", action="reject", allow_unloadable_champion=True
        )
    digest = approval.resolve_candidate_comparison_digest(
        config, "2", action="approve", allow_unloadable_champion=True
    )
    assert digest == "b" * 64


def test_job_passes_the_override_to_the_proof_lookup(monkeypatch):
    """Without the flag the automatic-policy lookup refuses before approval can run."""
    from unittest.mock import Mock

    from skyulf.integrations.databricks.jobs.shared import job_runtime

    resolver = Mock(return_value="b" * 64)
    monkeypatch.setattr(job_runtime, "resolve_candidate_comparison_digest", resolver)
    monkeypatch.setattr(job_runtime, "run_action", Mock())
    config = {
        "score_model_selection": "pinned_version",
        "promotion_policy": "automatic",
        "score_handoff": "after_alias_change",
        "model_name": "workspace.test.model",
        "model_version": "1",
    }
    params = {
        "lifecycle_action": "approve",
        "candidate_version": "3",
        "expected_champion_version": "1",
        "allow_unloadable_champion": "true",
    }
    job_runtime.run_bundle_action(None, config, params, task_role="lifecycle")
    resolver.assert_called_once_with(config, "3", action="approve", allow_unloadable_champion=True)
