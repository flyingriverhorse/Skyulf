# Delivery 182: native feature model packaging evidence

Status: implementation and focused local verification complete. Native
Databricks execution is user-deferred; no cloud run, deployment, commit or push
was performed by this workstream.

## Verified SDK boundary

Inspected the official `databricks-feature-engineering` 0.18.1 wheel without
installing or importing it. PyPI SHA256:
`e79c1d66a32d2c0cd0781947214256e8991d5d33b5d94333055b584addbb45d2`.
Source: [official API reference](https://api-docs.databricks.com/python/feature-engineering/latest/feature_engineering.client.html)
and [official distribution metadata](https://pypi.org/pypi/databricks-feature-engineering/0.18.1/json).

- `log_model(model=..., flavor=mlflow.pyfunc)` forwards the model through the
  pyfunc `python_model` save argument. Additional signature, metadata,
  artifacts, requirements and code paths belong to the nested raw model.
- The SDK logs an outer `databricks.feature_store.mlflow_model` wrapper with
  its own feature signature. It does not propagate the raw metadata to that
  wrapper. It returns model information in the inspected implementation;
  MLflow 3 can use a logged-model URI rather than the requested run URI.
- `TrainingSet.feature_spec` and `get_output_columns()` expose enough public
  metadata to validate exact named model inputs, types, exclusions and lookup
  semantics before logging. `load_df()` drops excluded metadata. Lifecycle
  code therefore creates separate materialization and logging TrainingSets.
- `score_batch` forwards `params` and the explicit `result_type` to the raw
  model UDF. Its internal local-package option is unavailable on serverless.
  There is no public hook to encode nullable integers/booleans after lookup
  and before Arrow. The native bridge rejects artifacts requiring that codec.
- SDK parameter inference can classify Python integer defaults as `integer`;
  passing the Skyulf batch-row default as `numpy.int64` preserves its declared
  `long` parameter type.

## Packaging contract

`log_local_feature_model` and `log_feature_model_set` use the same save-option
builders as the ordinary loggers. Fitted artifacts, exact pyfunc classes,
named input/output signatures, optional partition certificates, exact source
snapshots and runtime pins remain shared. Native feature logging does not
grant partition safety to uncertified artifacts.

The SDK copies the raw environment into its outer wrapper, including relative
`code/<wheel>.whl` pins. The bridge copies that identical worker wheel into
the outer `code` directory and verifies equality when loading. This makes
both raw and outer environment references resolve without rewriting pins.

The original native TrainingSet is passed directly to the injected or lazily
created FeatureEngineeringClient. The saved SDK `feature_spec.yaml` is also
checked against the declared lookup contract and raw named input types. Its
SHA256 is stored in the outer metadata. SDK column order may differ; the raw
signature retains the fitted input order and MLflow enforces named columns.
Unused excluded metadata may be absent from the SDK's saved spec; undeclared
exclusions, mismatched included types and changed temporal key types fail.

The final complete envelope is uploaded through an explicit tracking client
to `runs:/<run>/<path>`. It records the canonical three-key binding under
`skyulf_feature_store` and its contained raw path separately under
`skyulf_feature_store_raw_model_path`. Run and registry loaders validate the
envelope before exposing raw fitted assets, then attach canonical immutable
`feature_lookup_json`. Local artifact serialization remains unchanged.

The SDK uses a distinct internal logged-model name. MLflow 3 can merge a
same-named logged model over a run artifact while downloading a `runs:` URI;
a real regression reproduced loss of the enriched envelope through that path.
Separating those names preserves both explicit and default URI resolution.
Callers publish the returned completed `runs:` URI, not the SDK intermediate
logged-model URI. The bridge does not register the intermediate model.

`FeatureTrainingSpec` rejects different lookback windows for the same table,
including case variants, before any SDK operation. Native FeatureSpec stores
one lookback window per table, so it cannot represent that conflicting request.

`copy_feature_package` validates and copies a complete package for competition
adoption without reconstructing or relabeling its training lineage.

Matching active fluent runs remain active. When no fluent run exists, the
bridge temporarily selects the explicit run and restores the caller's tracking
URI, environment run selector and run status. MLflow's public `end_run` writes an end timestamp even when
restoring `RUNNING`; the outer lifecycle owns final termination. Conflicting
active runs are rejected before the SDK call. No private active-run stack is
modified.

## Local verification boundary

Tests use genuine pandas fitting, real local MLflow 3 logged-model serialization, raw pyfunc
reload/prediction and a SQLite registry. Only Feature Engineering itself is an
injected simulation matching the inspected public call and package layout.
These tests are not native SDK, Unity Catalog, Spark-worker or serverless
acceptance.

Final focused command, against frozen production source:

```powershell
.venv/Scripts/python.exe -m pytest skyulf-core/tests/integration/platforms/test_local_feature_model.py skyulf-core/tests/integration/platforms/test_databricks_feature_store.py -q --show-capture=no --basetemp=tmp_repro_artifacts/task182/pytest-native-package-verified
```

Result: **67 passed** in 70.20 seconds: 20 new package/contract tests and 47
existing feature-store helper tests. Eight existing MLflow warnings concern
the wrapper's absent optional input example; no test was skipped.

The affected consumer batch also ran these explicit files:

- `test_local_mlflow_model.py`
- `test_mlflow_model_set.py`
- `test_nullable_pyfunc_review_batch17.py`
- `test_registry_payload_transport.py`
- `test_registry_metadata_transport.py`
- `test_local_feature_model.py`

That batch finished **296 passed, one failed** in 366.66 seconds. The one
failure was the new MLflow 3 same-name adoption regression described above;
it was corrected and passed in the final 67-test run. The 278 ordinary
logger/model-set/nullable/registry consumer cases passed, and their relevant
implementation did not change afterward. The final narrow repairs also
reproduced lost `MLFLOW_RUN_ID` and accepted conflicting windows before fixing
them; both regression tests passed afterward.

Scoped Ruff, Ruff format, Ty and Lizard CCN <= 10 passed for the changed
packaging, registry and feature configuration sources and new test file.
`git diff --check` passed for the tracked workstream edits. Root owns the
complete CI Ty scope, full static gates, wheel build and final combined report.
The scoring owner independently verifies the real envelope through the
existing raw Spark admission gate; this remains a local test without a JVM.
