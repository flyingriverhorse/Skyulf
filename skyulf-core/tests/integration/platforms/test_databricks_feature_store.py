"""Offline boundaries for the optional Unity Catalog Feature Engineering adapter."""

import importlib
import subprocess
import sys
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


def _api():
    """Import the optional API inside tests so missing functionality fails clearly."""
    return importlib.import_module("skyulf.integrations.databricks.feature_store")


def _lookup(**changes):
    """Describe one explicitly named point-in-time feature lookup."""
    return _api().FeatureLookupSpec(
        **{
            "table_name": "catalog.features.customer_history",
            "lookup_key": ("customer_id",),
            "feature_names": ("spend_30d",),
            "timestamp_lookup_key": "event_time",
            **changes,
        }
    )


def _spec(**changes):
    """Keep identifier and timestamp columns out of model inputs by declaration."""
    return _api().FeatureTrainingSpec(
        **{
            "lookups": (_lookup(),),
            "label": "target",
            "exclude_columns": ("customer_id", "event_time"),
            **changes,
        }
    )


def _frame(**changes):
    """Represent only the Spark schema boundary without starting a JVM."""
    dtypes = {"customer_id": "bigint", "event_time": "timestamp", "target": "double"}
    dtypes.update(changes)
    return SimpleNamespace(columns=list(dtypes), dtypes=list(dtypes.items()))


def test_training_set_preserves_point_in_time_lookup_and_native_identity():
    """Dropping the time key or replacing the native set would lose feature lineage."""
    api = _api()
    client, factory = Mock(), Mock()
    frame = _frame()
    spec = _spec(lookups=(_lookup(lookback_window=timedelta(days=7)),))
    result = api.create_feature_training_set(frame, spec, client=client, lookup_factory=factory)
    factory.assert_called_once_with(
        table_name="catalog.features.customer_history",
        lookup_key=["customer_id"],
        feature_names=["spend_30d"],
        timestamp_lookup_key="event_time",
        lookback_window=timedelta(days=7),
    )
    client.create_training_set.assert_called_once_with(
        df=frame,
        feature_lookups=[factory.return_value],
        label="target",
        exclude_columns=["customer_id", "event_time"],
    )
    assert result is client.create_training_set.return_value


@pytest.mark.parametrize(
    "changes",
    [
        {"table_name": "features.customer_history"},
        {"lookup_key": "customer_id"},
        {"lookup_key": ()},
        {"lookup_key": ("customer_id", "customer_id")},
        {"lookup_key": ("customer id",)},
        {"feature_names": None},
        {"feature_names": ()},
        {"feature_names": ("spend_30d", "spend_30d")},
        {"feature_names": ("customer_id",)},
        {"timestamp_lookup_key": "customer_id"},
        {"timestamp_lookup_key": "spend_30d"},
        {"timestamp_type": "string"},
        {"lookback_window": timedelta(seconds=-1)},
        {"lookback_window": True},
        {"timestamp_lookup_key": None, "lookback_window": timedelta(days=1)},
    ],
)
def test_lookup_rejects_ambiguous_or_leaking_configuration(changes):
    """Invalid identity, wildcard features or reversed time bounds must fail locally."""
    with pytest.raises((TypeError, ValueError)):
        _lookup(**changes)


@pytest.mark.parametrize(
    "changes",
    [
        {"lookups": ()},
        {"lookups": "lookup"},
        {"lookups": (object(),)},
        {"label": "customer_id"},
        {"label": "event_time"},
        {"label": "spend_30d"},
        {"exclude_columns": ("target",)},
        {"exclude_columns": ("customer_id", "customer_id")},
    ],
)
def test_training_spec_rejects_label_leakage_and_invalid_contracts(changes):
    """The target cannot become a lookup input or disappear from the training set."""
    with pytest.raises((TypeError, ValueError)):
        _spec(**changes)


def test_training_spec_rejects_colliding_lookup_outputs():
    """Joining two equally named features must never choose an arbitrary source."""
    with pytest.raises(ValueError, match="feature"):
        _spec(lookups=(_lookup(), _lookup(table_name="catalog.other.history")))


@pytest.mark.parametrize("column", ["customer_id", "event_time", "target"])
def test_training_rejects_missing_columns_before_sdk_call(column):
    """Malformed source schemas must fail before contacting Databricks."""
    api = _api()
    client = Mock()
    frame = _frame()
    frame.columns.remove(column)
    frame.dtypes = [(name, dtype) for name, dtype in frame.dtypes if name != column]
    with pytest.raises(ValueError, match=column):
        api.create_feature_training_set(frame, _spec(), client=client, lookup_factory=Mock())
    client.create_training_set.assert_not_called()


@pytest.mark.parametrize("dtype", ["date", "string", "timestamp_ntz"])
def test_training_rejects_timestamp_dtype_drift(dtype):
    """Training and scoring must share the explicitly declared temporal dtype."""
    api = _api()
    with pytest.raises(ValueError, match="timestamp"):
        api.create_feature_training_set(
            _frame(event_time=dtype), _spec(), client=Mock(), lookup_factory=Mock()
        )


def test_date_lookup_and_static_lookup_are_supported():
    """Date-based history and ordinary keyed features retain their declared behavior."""
    api = _api()
    spec = _spec(
        lookups=(
            _lookup(timestamp_type="date"),
            _lookup(
                table_name="catalog.features.customer_profile",
                feature_names=("age",),
                timestamp_lookup_key=None,
            ),
        )
    )
    client, factory = Mock(), Mock()
    result = api.create_feature_training_set(
        _frame(event_time="date"), spec, client=client, lookup_factory=factory
    )
    assert factory.call_args_list[1].kwargs == {
        "table_name": "catalog.features.customer_profile",
        "lookup_key": ["customer_id"],
        "feature_names": ["age"],
    }
    assert result is client.create_training_set.return_value


def test_feature_values_cannot_silently_override_training_lookup():
    """A materialized input feature must not accidentally replace its history lookup."""
    api = _api()
    with pytest.raises(ValueError, match="override"):
        api.create_feature_training_set(
            _frame(spend_30d="double"), _spec(), client=Mock(), lookup_factory=Mock()
        )


def test_feature_override_detection_matches_spark_case_insensitive_names():
    """Changing capitalization must not bypass the source feature override guard."""
    api = _api()
    with pytest.raises(ValueError, match="override"):
        api.create_feature_training_set(
            _frame(SPEND_30D="double"), _spec(), client=Mock(), lookup_factory=Mock()
        )


def test_sdk_is_loaded_only_when_an_operation_needs_default_components(monkeypatch):
    """The unmocked adapter must resolve both default SDK factories at call time."""
    api = _api()
    runtime = importlib.import_module("skyulf.integrations.databricks.feature_store.runtime")
    sdk = SimpleNamespace(FeatureLookup=Mock(), FeatureEngineeringClient=Mock())
    loader = Mock(return_value=sdk)
    monkeypatch.setattr(runtime.importlib, "import_module", loader)
    spec = api.FeatureTrainingSpec(
        lookups=(
            api.FeatureLookupSpec(
                table_name="cat.schema.features",
                lookup_key=("customer_id",),
                feature_names=("spend_30d",),
            ),
        ),
        label="target",
    )
    result = api.create_feature_training_set(_frame(), spec)
    assert result is sdk.FeatureEngineeringClient.return_value.create_training_set.return_value
    assert {call.args for call in loader.call_args_list} == {("databricks.feature_engineering",)}


@pytest.mark.parametrize("missing", ["databricks", "databricks.feature_engineering"])
def test_missing_sdk_reports_optional_install_without_masking_input_validation(
    monkeypatch, missing
):
    """Runtime-only installation guidance must not affect base package imports."""
    api, spec = _api(), _spec()
    runtime = importlib.import_module("skyulf.integrations.databricks.feature_store.runtime")
    monkeypatch.setattr(
        runtime.importlib, "import_module", Mock(side_effect=ModuleNotFoundError(name=missing))
    )
    with pytest.raises(ImportError, match=r"skyulf-core\[feature-store\]"):
        api.create_feature_training_set(_frame(), spec)


def test_nested_sdk_dependency_error_is_not_misreported(monkeypatch):
    """An incomplete installed SDK must expose its actual missing dependency."""
    api, spec = _api(), _spec()
    runtime = importlib.import_module("skyulf.integrations.databricks.feature_store.runtime")
    failure = ModuleNotFoundError("missing transitive dependency", name="mlflow")
    monkeypatch.setattr(runtime.importlib, "import_module", Mock(side_effect=failure))
    with pytest.raises(ModuleNotFoundError) as error:
        api.create_feature_training_set(_frame(), spec)
    assert error.value is failure


def test_logging_preserves_training_set_and_named_signature():
    """Model packaging must pass the native feature lineage and output names intact."""
    api = _api()
    client, model, training_set, signature = Mock(), object(), Mock(), object()
    flavor = SimpleNamespace(save_model=lambda **kwargs: None)
    result = api.log_feature_model(
        model,
        training_set=training_set,
        flavor=flavor,
        artifact_path="model",
        client=client,
        signature=signature,
        registered_model_name="catalog.models.churn",
    )
    client.log_model.assert_called_once_with(
        model=model,
        training_set=training_set,
        flavor=flavor,
        artifact_path="model",
        signature=signature,
        registered_model_name="catalog.models.churn",
    )
    assert result is client.log_model.return_value


@pytest.mark.parametrize("artifact_path", ["", "/model", "../model", "a/../model"])
def test_logging_rejects_nonrelative_artifact_paths(artifact_path):
    """Feature artifacts must remain under the selected MLflow run."""
    api = _api()
    with pytest.raises(ValueError, match="artifact_path"):
        api.log_feature_model(
            object(),
            training_set=Mock(),
            flavor=Mock(),
            artifact_path=artifact_path,
            client=Mock(),
        )


def test_scoring_keeps_named_schema_and_requires_no_training_label():
    """Batch lookup must preserve structured outputs while accepting unlabeled rows."""
    api = _api()
    client = Mock()
    frame = SimpleNamespace(
        columns=["customer_id", "event_time"],
        dtypes=[("customer_id", "bigint"), ("event_time", "timestamp")],
    )
    schema = "struct<prediction:double,probability:double>"
    result = api.score_feature_model(
        "models:/catalog.models.churn/3", frame, _spec(), result_type=schema, client=client
    )
    client.score_batch.assert_called_once_with(
        model_uri="models:/catalog.models.churn/3", df=frame, result_type=schema
    )
    assert result is client.score_batch.return_value


def test_scoring_feature_overrides_require_explicit_opt_in():
    """Databricks-supplied feature overrides must be a visible caller decision."""
    api = _api()
    frame, client = _frame(spend_30d="double"), Mock()
    with pytest.raises(ValueError, match="override"):
        api.score_feature_model(
            "runs:/abc/model", frame, _spec(), result_type="double", client=client
        )
    result = api.score_feature_model(
        "runs:/abc/model",
        frame,
        _spec(),
        result_type="double",
        client=client,
        allow_feature_overrides=True,
    )
    assert result is client.score_batch.return_value


def test_scoring_rejects_temporal_dtype_drift_before_sdk_call():
    """A DATE cannot silently replace the timestamp used when the model was trained."""
    api = _api()
    client = Mock()
    with pytest.raises(ValueError, match="timestamp"):
        api.score_feature_model(
            "runs:/abc/model",
            _frame(event_time="date"),
            _spec(),
            result_type="double",
            client=client,
        )
    client.score_batch.assert_not_called()


def test_import_does_not_require_databricks_sdk_spark_or_mlflow():
    """Base installs must retain usable configuration without optional runtimes."""
    script = """
import sys
class BlockOptionalImports:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'databricks', 'pyspark', 'mlflow'}:
            raise AssertionError('Unexpected optional import: ' + fullname)
sys.meta_path.insert(0, BlockOptionalImports())
from skyulf.integrations.databricks.feature_store import FeatureLookupSpec
assert FeatureLookupSpec.__name__ == 'FeatureLookupSpec'
"""
    child = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=60
    )
    assert child.returncode == 0, child.stderr
