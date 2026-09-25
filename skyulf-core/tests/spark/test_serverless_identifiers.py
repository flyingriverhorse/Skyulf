"""Identifier safety when serverless compute hides Spark configuration."""

from types import SimpleNamespace

import pytest

from skyulf.core.execution import ExecutionOptions, FrameSpec
from skyulf.inference import build_bundle, predict_spark
from skyulf.preprocessing._spark import _resolved_names


class UnavailableConfiguration(Exception):
    """Mirror the structured condition returned by Databricks Spark Connect."""

    def getCondition(self):
        """Identify unavailable configuration without parsing error messages."""
        return "CONFIG_NOT_AVAILABLE.WITHOUT_SUGGESTION"


@pytest.mark.parametrize("error", [UnavailableConfiguration(), PermissionError("denied")])
def test_unavailable_config_uses_conservative_identifier_rules(error):
    """Hidden configuration must retain collision detection without hiding access failures."""

    def get(key):
        """Reproduce the live failure at the RuntimeConfig boundary."""
        assert key == "spark.sql.caseSensitive"
        raise error

    frame = SimpleNamespace(
        columns=["Feature", "feature"], sparkSession=SimpleNamespace(conf=SimpleNamespace(get=get))
    )
    if isinstance(error, UnavailableConfiguration):
        assert _resolved_names(frame) == ["feature", "feature"]
    else:
        with pytest.raises(PermissionError, match="denied"):
            _resolved_names(frame)


@pytest.mark.parametrize("mode", ["native_features", "python_pipeline"])
def test_inference_with_hidden_identifier_config(
    spark, fitted_regression_pipeline, monkeypatch, mode
):
    """Both inference paths retain keyed predictions and reject ambiguous input names."""
    bundle = build_bundle(fitted_regression_pipeline, input_stage="raw", feature_order=("x", "z"))
    original = type(spark.conf).get

    def hidden(self, key, *args, **kwargs):
        """Hide only the setting rejected by real Databricks serverless."""
        if key == "spark.sql.caseSensitive":
            raise UnavailableConfiguration()
        return original(self, key, *args, **kwargs)

    monkeypatch.setattr(type(spark.conf), "get", hidden)
    frame = spark.createDataFrame([(7, 150.0, 9.0)], "id long, x double, z double")
    rows = predict_spark(
        frame,
        bundle,
        frame_spec=FrameSpec(record_key_columns=("id",)),
        options=ExecutionOptions("spark"),
        mode=mode,
    ).collect()
    assert rows[0].id == 7 and rows[0].prediction == pytest.approx(280.0)
    ambiguous = frame.selectExpr("id", "x", "z", "x AS X")
    with pytest.raises(ValueError, match="Duplicate Spark column"):
        predict_spark(
            ambiguous,
            bundle,
            frame_spec=FrameSpec(record_key_columns=("id",)),
            options=ExecutionOptions("spark"),
            mode=mode,
        )
