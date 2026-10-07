# Integration helper boundaries

Integration orchestration consumes named helpers from their owning modules.
An underscore-prefixed **module** denotes package-internal implementation;
underscore-prefixed **symbols** inside a module remain that module's details.
These helpers are not new top-level public SDK exports.

Shared primitives live in small modules with no workflow orchestration:

| Module | Responsibility |
| --- | --- |
| `databricks/shared/_local_frames.py` | Local frame memory accounting and finite scalar normalization |
| `databricks/shared/_batch_manifest.py` | Identical request/evidence manifests for local and Spark publication |
| `mlflow/shared/_client.py` | Lazy tracking/registry clients and experiment resolution |
| `mlflow/shared/_model_metadata.py` | MLflow primitive signature types and portable artifact metadata |

Tracking and registry clients retain separate URI contracts. Importing the
client module does not import MLflow, create a client, or change global URIs.
Typed registry errors remain owned by `mlflow/registration/registry.py`.

Workflow, lifecycle, training, prediction and notebook operations stay in their
existing owner modules. Shared helpers are defined directly with the names their
callers use. There are no private/public compatibility aliases or forwarding
wrappers. Internal-only helpers retain their leading underscore.

For example, another module imports `training_spec` from
`databricks/lifecycle/local_workflow.py` or uses `candidate_config` from
`databricks/training/fitting/local_retraining.py`; it does not reach into those modules'
`_training_spec` or `_candidate_config`. The definitions are `def training_spec`
and `def candidate_config`, and same-module calls use those names too.

Monitoring and retraining follow the same rule. Local and Spark implementations
share declared helpers such as `monitoring_metric`, `validate_monitoring_inputs`,
`validate_record_keys`, `row_identity_counts` and `partition_training_rows` from
their domain owners. Core categorical drift evidence is shared through
`profiling/_drift_evidence.py`'s `categorical_drift_test`.

Compatibility modules may import a canonical implementation **module** as
`_implementation` to preserve module identity. This does not permit importing a
function under a second private name or reaching into that module's private
symbols. The boundary checks resolve the imported module file or package before
allowing the module binding; domain membership alone grants no exemption.

Tests patch the name resolved by the code under test: the consumer's binding
for an eager import, or the owner's function name for module-qualified and
same-module calls. Ordinary Python `from module import helper` captures a binding;
renaming does not change that rule. Negative tests must verify the actual lookup
with a failing sentinel, and a valid-path control proves that sentinel is active.

`test_internal_api_boundary.py` rejects borrowed private-symbol imports and
module-qualified private access through relative and absolute imports. It also checks
fresh-process import ordering without the optional MLflow package and rejects
private/public helper aliases. Positive and negative fixture cases distinguish
module compatibility imports from private function access. Existing
integration tests own behavioral contracts, including validation ordering,
exact errors, saved digests, alias transactions and prediction publication.

Core integration-facing helpers (`validate_bundle_contract`,
`read_bounded_artifact`, `validate_class_membership`) are declared at their
existing owners. Their behavior is shared, not reimplemented in integrations.
