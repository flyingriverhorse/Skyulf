"""Versioned MLflow invocation and receipt storage for serialized lifecycle tasks."""

import json
import re
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from skyulf.integrations.mlflow.shared._client import make_registry_client, require_mlflow

from ...mlflow.runs.tracking import TrackingRun
from ..training.shared.local_training_evidence import evidence_digest

PHASE_PREDECESSORS = {
    "load_data": "prepare",
    "prepare_dataset": "load_data",
    "select_best_model": "train",
    "train": "prepare",
    "evaluate_register": "train",
    "compare": "evaluate_register",
    "generate_charts": "compare",
    "decide": "compare",
    "operator": "prepare",
    "finalize": "prepare",
    "result": "prepare",
}


@dataclass(frozen=True, slots=True)
class LifecycleContext:
    """Bind each fixed notebook task to one unrepaired Databricks job invocation."""

    job_id: str
    job_run_id: str
    repair_count: int = 0
    execution_count: int = 1

    def validate(self) -> None:
        """Reject unresolved references and repeated attempts before external work."""
        if any(
            not isinstance(value, str) or not re.fullmatch(r"[1-9][0-9]*", value)
            for value in (self.job_id, self.job_run_id)
        ):
            raise ValueError("Lifecycle invocation requires concrete numeric job and run IDs.")
        if (
            type(self.repair_count) is not int
            or self.repair_count != 0
            or type(self.execution_count) is not int
            or self.execution_count != 1
        ):
            raise ValueError(
                "Lifecycle repair/retry is unsupported; inspect evidence and start a fresh run."
            )

    def identity(self) -> dict[str, str]:
        """Exclude per-task attempt counters from the shared invocation identity."""
        return {"job_id": self.job_id, "job_run_id": self.job_run_id}


@dataclass(frozen=True, slots=True)
class LifecyclePhaseResult:
    """Separate a small task reference from human-readable phase evidence."""

    reference: dict[str, str]
    output: dict[str, Any]


class PhaseStore:
    """Read and write explicit client artifacts without a fluent active MLflow run."""

    def __init__(self, tracking_uri: str, context: LifecycleContext) -> None:
        """Keep the client store and validated invocation fixed across each operation."""
        self.context = context
        self.client = make_registry_client(require_mlflow(), tracking_uri, None)
        self.run_id = ""
        self.request_digest = ""
        self.request: dict[str, Any] = {}

    @property
    def run(self) -> TrackingRun:
        """Provide the same logging surface as ordinary sequential candidate training."""
        return TrackingRun(client=self.client, run_id=self.run_id, enabled=True)

    def log(self, path: str, payload: dict[str, Any]) -> None:
        """Persist JSON under this explicit run without retaining local paths."""
        self.client.log_dict(self.run_id, payload, path)

    def read(self, path: str) -> dict[str, Any]:
        """Download each document into a disposable directory and require an object."""
        with TemporaryDirectory(prefix="skyulf-phase-evidence-") as directory:
            downloaded = self.client.download_artifacts(self.run_id, path, directory)
            value = json.loads(Path(downloaded).read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError("Lifecycle evidence must be a JSON object.")
        return value

    def tags(self) -> dict[str, str]:
        """Refresh durable status instead of trusting an in-process cached outcome."""
        return self.client.get_run(self.run_id).data.tags

    def bind(self, reference: dict[str, str]) -> None:
        """Verify request integrity and invocation before accepting a phase reference."""
        if not isinstance(reference, dict) or set(reference) != {
            "run_id",
            "request_sha256",
            "phase",
            "receipt_sha256",
        }:
            raise ValueError("Lifecycle requires a complete predecessor reference.")
        if any(not isinstance(value, str) or not value for value in reference.values()):
            raise ValueError("Lifecycle predecessor reference values must be nonempty strings.")
        self.run_id = reference["run_id"]
        self.request_digest = reference["request_sha256"]
        self.request = self.read("lifecycle/request.json")
        self._validate_bound_request()
        receipt = self.receipt(reference["phase"])
        if evidence_digest(receipt) != reference["receipt_sha256"]:
            raise ValueError("Lifecycle predecessor receipt digest differs from its reference.")

    def _validate_bound_request(self) -> None:
        """Verify the loaded request identity and saved integrity tag."""
        if (
            self.request.get("version") != 1
            or self.request.get("context") != self.context.identity()
            or self.request.get("run_id") != self.run_id
            or evidence_digest(self.request) != self.request_digest
            or self.tags().get("skyulf.lifecycle.request") != self.request_digest
        ):
            raise ValueError("Lifecycle invocation or pinned request differs from its reference.")

    def receipt(self, phase: str) -> dict[str, Any]:
        """Verify the saved receipt and its entire predecessor chain before reuse."""
        digest = self.tags().get(f"skyulf.lifecycle.{phase}.receipt")
        if not digest:
            raise ValueError(f"Lifecycle predecessor {phase} has no completed receipt.")
        value = self.read(f"lifecycle/{phase}.json")
        if (
            evidence_digest(value) != digest
            or value.get("version") != 1
            or value.get("phase") != phase
            or value.get("request_sha256") != self.request_digest
            or value.get("run_id") != self.run_id
        ):
            raise ValueError(
                "Lifecycle predecessor identity or digest differs from saved evidence."
            )
        self._validate_predecessor_chain(phase, value)
        return value

    def _validate_predecessor_chain(self, phase: str, value: dict[str, Any]) -> None:
        """Recursively verify the expected predecessor before reusing a receipt."""
        previous = value.get("predecessor")
        expected = self.predecessor(phase)
        if expected is None:
            if phase != "prepare" or previous is not None:
                raise ValueError("Invalid lifecycle predecessor chain.")
        elif (
            not isinstance(previous, dict)
            or previous.get("phase") != expected
            or previous != self.reference(self.receipt(expected))
        ):
            raise ValueError("Lifecycle predecessor identity differs from saved evidence.")

    def predecessor(self, phase: str) -> str | None:
        """Use the graph pinned at initialization when validating the receipt chain."""
        if "branch_plan" in self.request:
            branches = {
                f"branch_{item['name']}" for item in self.request["branch_plan"]["branches"]
            }
            if phase in branches | {"register_model_set"}:
                return "prepare"
            if phase in {"evaluate_model_set", "model_decision", "generate_charts"}:
                return {
                    "evaluate_model_set": "register_model_set",
                    "model_decision": "evaluate_model_set",
                    "generate_charts": "evaluate_model_set",
                }[phase]
        if phase.startswith("candidate_") and phase.removeprefix("candidate_") in self.request.get(
            "competition", {}
        ).get("candidates", {}):
            return "prepare_dataset"
        if self.request.get("graph_version", 2) == 3:
            overrides = {"train": "prepare_dataset", "evaluate_register": "select_best_model"}
            if phase in overrides:
                return overrides[phase]
        return PHASE_PREDECESSORS.get(phase)

    def reference(self, receipt: dict[str, Any]) -> dict[str, str]:
        """Expose only durable identity and digest strings to task values."""
        return {
            "run_id": self.run_id,
            "request_sha256": self.request_digest,
            "phase": receipt["phase"],
            "receipt_sha256": evidence_digest(receipt),
        }

    def begin(self, phase: str) -> None:
        """Write attempt intent first and reject ambiguous or completed repeated attempts."""
        key = f"skyulf.lifecycle.{phase}.attempt"
        if self.tags().get(key):
            raise ValueError(
                "Lifecycle phase already attempted; inspect evidence and start a fresh run."
            )
        self.client.set_tag(self.run_id, key, "started")

    def complete(
        self, phase: str, output: dict[str, Any], predecessor: dict[str, str] | None
    ) -> LifecyclePhaseResult:
        """Persist completion before publishing the reference to the next notebook."""
        receipt = {
            "version": 1,
            "run_id": self.run_id,
            "request_sha256": self.request_digest,
            "phase": phase,
            "predecessor": predecessor,
            "output": output,
        }
        self.log(f"lifecycle/{phase}.json", receipt)
        self.client.set_tag(
            self.run_id, f"skyulf.lifecycle.{phase}.receipt", evidence_digest(receipt)
        )
        return LifecyclePhaseResult(self.reference(receipt), output)
