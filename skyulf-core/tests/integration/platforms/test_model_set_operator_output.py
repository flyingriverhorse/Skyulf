"""Model-set reports must show operator values that the train job accepts unchanged."""

import json
import re
from dataclasses import asdict
from html import unescape

import pytest

from skyulf.integrations.databricks.jobs.shared.job_runtime import operator_options
from skyulf.integrations.databricks.model_sets.model_set_project import (
    render_model_set_result,
    set_operator_actions,
)
from skyulf.integrations.mlflow.lifecycle.promotion import AliasChangeReceipt


def _promotion(prior="2", new="3"):
    """Build the receipt a completed set promotion stores in its output."""
    return asdict(
        AliasChangeReceipt(
            event_id="evt-1",
            kind="promotion",
            model_name="main.models.claims_set",
            alias="champion",
            prior_version=prior,
            new_version=new,
            comparison_sha256=None,
            parent_event_id=None,
            previous_champion_version=prior,
        )
    )


def _manual_train(passed=True, champion="2"):
    """Return a manual-approval model_decision output for candidate set v3."""
    return {
        "model_set_candidate": {"name": "main.models.claims_set", "version": "3", "digest": "d"},
        "quality": {"passed": passed, "expected_champion_version": champion},
        "promotion_policy": "manual_approval",
        "alias_change": None,
        "next_actions": ["Run approve with candidate_version and expected_champion_version."],
    }


def _table_values(html: str) -> dict[str, str]:
    """Read the Parameter/Value rows rendered after one action heading."""
    rows = re.findall(r"<tr><td>([a-z_]+)</td><td>(.*?)</td></tr>", html)
    return {key: unescape(value) for key, value in rows}


def test_manual_train_offers_approve_and_reject_with_accepted_values():
    """Copied approve/reject values must pass the train job's operator validation."""
    actions = set_operator_actions(_manual_train())
    assert set(actions) == {"approve", "reject"}
    approve = {k: v for k, v in actions["approve"].items() if k != "lifecycle_action"}
    assert operator_options("approve", approve)["candidate_version"] == "3"
    reject = {k: v for k, v in actions["reject"].items() if k != "lifecycle_action"}
    assert operator_options("reject", {**reject, "rejection_reason": "Needs review"})
    html = render_model_set_result(_manual_train())
    assert "Available action: approve" in html and "Available action: reject" in html


def test_first_set_uses_none_and_failed_quality_hides_approve():
    """The bootstrap sentinel is explicit and a failed set cannot be offered for approval."""
    first = set_operator_actions(_manual_train(champion=None))
    assert first["approve"]["expected_champion_version"] == "none"
    assert operator_options(
        "approve", {"candidate_version": "3", "expected_champion_version": "none"}
    )
    failed = set_operator_actions(_manual_train(passed=False))
    assert set(failed) == {"reject"}


@pytest.mark.parametrize("key", ["alias_change", "receipt"])
def test_promotion_offers_rollback_that_round_trips(key):
    """The rendered receipt JSON must be accepted by the rollback parser as-is."""
    payload = {"action": "approve", key: _promotion()}
    if key == "alias_change":
        payload.update(_manual_train(), alias_change=_promotion(), promotion_policy="automatic")
    actions = set_operator_actions(payload)
    assert set(actions) == {"rollback"}
    html = render_model_set_result(payload)
    assert "If rollback is needed" in html and "Restore version" in html
    values = _table_values(html.split("If rollback is needed", 1)[1])
    options = operator_options("rollback", values)
    assert options["promotion_receipt"].new_version == "3"
    assert json.loads(values["promotion_receipt_json"])["prior_version"] == "2"


def test_initial_activation_and_rollback_results_offer_no_rollback():
    """Only a completed promotion can be reversed; initial and rollback receipts cannot."""
    initial = {**_promotion(prior=None), "kind": "initial"}
    assert set_operator_actions({"action": "approve", "receipt": initial}) == {}
    reversed_ = {**_promotion(), "kind": "rollback"}
    assert set_operator_actions({"action": "rollback", "receipt": reversed_}) == {}
    html = render_model_set_result({"action": "rollback", "receipt": reversed_})
    assert "Available action" not in html and "If rollback is needed" not in html
