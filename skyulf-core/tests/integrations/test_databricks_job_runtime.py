"""Bundle job parameters must preserve Core lifecycle and independent score ownership."""

import json
from dataclasses import asdict

import pytest

from skyulf.integrations.mlflow.promotion import AliasChangeReceipt


def _config(**changes):
    """Use explicit independent policies rather than inheriting a legacy default."""
    return {
        "score_model_selection": "pinned_version",
        "promotion_policy": "manual_approval",
        "score_handoff": "after_alias_change",
        "model_name": "workspace.test.model",
        "model_version": "1",
        **changes,
    }


def _receipt(kind="promotion"):
    """Represent a checked Core result independently of the Bundle wrapper."""
    return AliasChangeReceipt(
        event_id="a" * 32,
        kind=kind,
        model_name="workspace.test.model",
        alias="champion",
        prior_version="1",
        new_version="2",
        comparison_sha256="b" * 64,
        parent_event_id=None,
    )


@pytest.mark.parametrize("version", ["", "2", "123"])
def test_score_version_override_is_run_scoped(monkeypatch, version):
    """An operator can select a version without changing the saved policy or champion."""
    from unittest.mock import Mock

    from skyulf.integrations.databricks import job_runtime

    execute = Mock(return_value={})
    monkeypatch.setattr(job_runtime, "run_action", execute)
    config = _config(score_model_selection="champion")
    job_runtime.run_bundle_action(None, config, {"score_model_version": version}, task_role="score")
    used = execute.call_args.args[1]
    assert used["score_model_selection"] == ("pinned_version" if version else "champion")
    assert used["model_version"] == (version or "1")
    assert config == _config(score_model_selection="champion")


@pytest.mark.parametrize("version", ["latest", "0", "-1", "1.0", " 2", 2, None])
def test_invalid_score_override_never_reaches_core(monkeypatch, version):
    """Only explicit positive versions can bypass the configured scoring selector."""
    from unittest.mock import Mock

    from skyulf.integrations.databricks import job_runtime

    execute = Mock()
    monkeypatch.setattr(job_runtime, "run_action", execute)
    with pytest.raises(ValueError):
        job_runtime.run_bundle_action(
            None, _config(), {"score_model_version": version}, task_role="score"
        )
    execute.assert_not_called()


def test_lifecycle_refuses_scoring_override(monkeypatch):
    """A lifecycle run cannot silently pass an accidental model pin to child scoring."""
    from unittest.mock import Mock

    from skyulf.integrations.databricks import job_runtime

    execute = Mock()
    monkeypatch.setattr(job_runtime, "run_action", execute)
    with pytest.raises(ValueError, match="score_model_version"):
        job_runtime.run_bundle_action(
            None,
            _config(),
            {"lifecycle_action": "train", "score_model_version": "2"},
            task_role="lifecycle",
        )
    execute.assert_not_called()


@pytest.mark.parametrize("action", ["approve", "reject", "rollback"])
def test_score_role_refuses_lifecycle_inputs_before_calling_core(action):
    """The score notebook cannot serve as an independent concurrent alias writer."""
    from skyulf.integrations.databricks.job_runtime import run_bundle_action

    with pytest.raises(ValueError, match="score|Score"):
        run_bundle_action(None, _config(), {"lifecycle_action": action}, task_role="score")


@pytest.mark.parametrize(
    "parameters",
    [
        {"task_role": "lifecycle", "lifecycle_action": "approve"},
        {"task_role": "lifecycle", "lifecycle_action": "rollback", "promotion_receipt_json": "[]"},
        {"task_role": "lifecycle", "lifecycle_action": "score"},
        {"task_role": "lifecycle", "lifecycle_action": "train", "candidate_version": "2"},
        {"task_role": "other"},
    ],
)
def test_bad_operator_parameters_fail_without_spark_or_registry(parameters):
    """Incomplete or stale form inputs must not fall through to training or alias mutation."""
    from skyulf.integrations.databricks.job_runtime import run_bundle_action

    with pytest.raises(ValueError):
        run_bundle_action(None, _config(), parameters, task_role=parameters.pop("task_role"))


@pytest.mark.parametrize(
    "config",
    [
        {"model_selection_mode": "auto_champion"},
        {"score_model_selection": "champion"},
        _config(score_handoff="always"),
    ],
)
def test_bundle_rejects_partial_policy_and_handoff_migration(config):
    """A new job graph must not silently inherit the old coupled policy semantics."""
    from skyulf.integrations.databricks.job_runtime import run_bundle_action

    with pytest.raises(ValueError):
        run_bundle_action(None, config, {"lifecycle_action": "train"}, task_role="lifecycle")


@pytest.mark.parametrize("field", ["task_role", "action"])
def test_job_parameters_cannot_override_deployed_notebook_role(field):
    """User parameters cannot turn the score entrypoint into an alias writer."""
    from skyulf.integrations.databricks.job_runtime import run_bundle_action

    with pytest.raises(ValueError, match="cannot be overridden"):
        run_bundle_action(None, _config(), {field: "approve"}, task_role="score")


@pytest.mark.parametrize(
    "payload",
    [
        "{",
        "[]",
        "null",
        json.dumps({**asdict(_receipt()), "kind": "initial"}),
        json.dumps({**asdict(_receipt()), "alias": "challenger"}),
        json.dumps({**asdict(_receipt()), "new_version": "3"}),
        json.dumps({**asdict(_receipt()), "prior_version": None}),
        json.dumps(
            {key: value for key, value in asdict(_receipt()).items() if key != "prior_version"}
        ),
    ],
)
def test_invalid_rollback_receipt_fails_after_expected_version_validation(payload):
    """A valid expected version must not let malformed transition evidence reach Core."""
    from skyulf.integrations.databricks.job_runtime import run_bundle_action

    with pytest.raises(ValueError, match="complete promotion_receipt_json"):
        run_bundle_action(
            None,
            _config(),
            {
                "lifecycle_action": "rollback",
                "expected_champion_version": "2",
                "promotion_receipt_json": payload,
            },
            task_role="lifecycle",
        )


@pytest.mark.parametrize("action", ["approve", "reject", "rollback"])
@pytest.mark.parametrize("selection", ["pinned_version", "champion"])
@pytest.mark.parametrize("handoff", ["disabled", "after_alias_change"])
def test_operator_handoff_depends_on_committed_transition_not_score_selector(
    monkeypatch, action, selection, handoff
):
    """Rejected candidates never score; approved or reversed transitions respect explicit handoff."""
    from skyulf.integrations.databricks import job_runtime

    receipt = _receipt(
        "rejection" if action == "reject" else "rollback" if action == "rollback" else "promotion"
    )
    observed = []

    def execute(spark, config, selected_action, **kwargs):
        """Capture the external lifecycle boundary while returning its real receipt type."""
        observed.append((selected_action, kwargs))
        return receipt

    monkeypatch.setattr(job_runtime, "run_action", execute)
    parameters = {
        "task_role": "lifecycle",
        "lifecycle_action": action,
        "expected_champion_version": "1",
    }
    if action == "rollback":
        parameters.update(
            promotion_receipt_json=json.dumps(asdict(_receipt())), expected_champion_version="2"
        )
    else:
        parameters.update(candidate_version="2", comparison_sha256="b" * 64)
        if action == "reject":
            parameters["rejection_reason"] = "Business review declined"
    config = _config(score_model_selection=selection, score_handoff=handoff)
    outcome = job_runtime.run_bundle_action(
        None, config, parameters, task_role=parameters.pop("task_role")
    )
    assert outcome.result == receipt
    assert outcome.score_requested == (handoff == "after_alias_change" and action != "reject")
    assert observed[0][0] == action
    assert observed[0][1]["expected_champion_version"] == parameters["expected_champion_version"]
    assert config["model_version"] == "1"


def test_unknown_alias_outcome_never_yields_score_handoff(monkeypatch):
    """A failed alias action cannot produce a success envelope that triggers score."""
    from skyulf.integrations.databricks import job_runtime
    from skyulf.integrations.mlflow.promotion import AliasOutcomeUnknownError

    def uncertain(*args, **kwargs):
        """Represent a real remote write whose final state is unknown."""
        raise AliasOutcomeUnknownError("pending event")

    monkeypatch.setattr(job_runtime, "run_action", uncertain)
    with pytest.raises(AliasOutcomeUnknownError):
        job_runtime.run_bundle_action(
            None,
            _config(),
            {
                "lifecycle_action": "approve",
                "candidate_version": "2",
                "comparison_sha256": "b" * 64,
                "expected_champion_version": "1",
            },
            task_role="lifecycle",
        )


@pytest.mark.parametrize(
    "event",
    [
        None,
        "{",
        {"k": "nomination", "s": "committed"},
        {"k": "challenger", "s": "prepared", "h": "b" * 64},
        {"k": "challenger", "s": "committed", "h": "bad"},
    ],
)
def test_saved_proof_lookup_refuses_missing_uncommitted_or_malformed_receipts(monkeypatch, event):
    """Omitting the digest must never derive authority from an unchecked comparison artifact."""
    from types import SimpleNamespace
    from unittest.mock import Mock

    from skyulf.integrations.databricks import local_approval
    from skyulf.integrations.mlflow.promotion import AliasConflictError

    client = Mock()
    raw = event if isinstance(event, str) else json.dumps(event)
    client.get_model_version.return_value = SimpleNamespace(tags={"promotion_event": raw})
    monkeypatch.setattr(local_approval, "controlled_champion_version", lambda *a, **kw: "1")
    monkeypatch.setattr(local_approval, "_active_marker", lambda *a, **kw: "event")
    monkeypatch.setattr(local_approval, "_require_mlflow", lambda: object())
    monkeypatch.setattr(local_approval, "_make_client", lambda *a: client)
    with pytest.raises(AliasConflictError):
        local_approval.resolve_candidate_comparison_digest(_config(), "2", action="approve")


@pytest.mark.parametrize("action", ["approve", "reject"])
def test_simple_operator_form_passes_full_saved_digest_to_strict_service(monkeypatch, action):
    """The UI convenience must preserve the full evidence pin at the mutation boundary."""
    from unittest.mock import Mock

    from skyulf.integrations.databricks import job_runtime

    resolver = Mock(return_value="b" * 64)
    execute = Mock(return_value=_receipt("rejection" if action == "reject" else "promotion"))
    monkeypatch.setattr(job_runtime, "resolve_candidate_comparison_digest", resolver)
    monkeypatch.setattr(job_runtime, "run_action", execute)
    params = {
        "lifecycle_action": action,
        "candidate_version": "2",
        "expected_champion_version": "1",
    }
    if action == "reject":
        params["rejection_reason"] = "Business review"
    job_runtime.run_bundle_action(None, _config(), params, task_role="lifecycle")
    resolver.assert_called_once_with(_config(), "2", action=action)
    assert execute.call_args.kwargs["comparison_sha256"] == "b" * 64
