"""Tests for skyulf.preprocessing.base (StatefulTransformer fit_transform/transform paths)."""

import gc
import tracemalloc
import typing

import pandas as pd
import pytest
from tests.utils.test_case_loader import TestCaseLoader

from skyulf.core.schema import SkyulfSchema
from skyulf.data.dataset import SplitDataset
from skyulf.preprocessing.base import BaseApplier, BaseCalculator, StatefulTransformer

_split_dataset_guard_cases = TestCaseLoader("preprocessing/preprocessing_base").load()


class _AddOneCalculator(BaseCalculator):
    """Calculator that just remembers a constant increment."""

    def fit(self, df, config):
        """Return a static params dict (no data-dependent fitting needed)."""
        return {"increment": config.get("increment", 1)}


class _AddOneApplier(BaseApplier):
    """Applier that adds `increment` to every value in column 'a'."""

    def apply(self, df, params):
        """Add params['increment'] to column 'a' and return the modified frame."""
        if isinstance(df, tuple):
            X, y = df
            X = X.copy()
            X["a"] = X["a"] + params["increment"]
            return (X, y)
        df = df.copy()
        df["a"] = df["a"] + params["increment"]
        return df


class _FailingCalculator(BaseCalculator):
    """Calculator used to exercise fit-time cleanup paths."""

    def fit(self, df, config):
        """Raise a deliberate error before producing fitted params."""
        raise ValueError("intentional fit failure")


class _FailingApplier(BaseApplier):
    """Applier used to exercise apply-time cleanup paths."""

    def apply(self, df, params):
        """Raise a deliberate error while applying fitted params."""
        raise RuntimeError("intentional apply failure")


class _TrainTransformCalculator(BaseCalculator):
    """Calculator with an explicit cross-fitted-style training output."""

    def fit(self, df, config):
        """Fail if the generic fallback path is selected."""
        raise AssertionError("fit_transform_train should be used")

    def fit_transform_train(self, df, config):
        """Return fitted parameters and a special training representation."""
        out = df.copy()
        out["a"] = out["a"] + 100
        return {"increment": 10}, out


class _ConfigEchoCalculator(BaseCalculator):
    """Calculator that passes test-only flags through to the applier."""

    def fit(self, df, config):
        """Return the increment plus any apply-failure flag from config."""
        return {
            "increment": config.get("increment", 1),
            "fail_apply": config.get("fail_apply", False),
        }


class _MaybeFailApplier(BaseApplier):
    """Applier that can fail on demand for transformer reuse tests."""

    def apply(self, df, params):
        """Raise when requested; otherwise delegate to the normal add-one applier."""
        if params.get("fail_apply"):
            raise RuntimeError("intentional configurable apply failure")
        return _AddOneApplier().apply(df, params)


class _TracingControlCalculator(BaseCalculator):
    """Calculator that forwards tracing-control flags to the applier."""

    def fit(self, df, config):
        """Return the increment plus tracemalloc control flags from config."""
        return {
            "increment": config.get("increment", 1),
            "stop_tracing": config.get("stop_tracing", False),
            "fail_after_stop": config.get("fail_after_stop", False),
        }


class _TracingControlApplier(BaseApplier):
    """Applier that can stop tracemalloc before succeeding or failing."""

    def apply(self, df, params):
        """Apply normally, then optionally stop tracing and raise the original error."""
        result = _AddOneApplier().apply(df, params)
        if params.get("stop_tracing"):
            tracemalloc.stop()
        if params.get("fail_after_stop"):
            raise RuntimeError("intentional post-stop apply failure")
        return result


class _CountingSplitReturningApplier(BaseApplier):
    """Returns the input frame unchanged for the first `trigger_after` calls, then
    illegally returns a SplitDataset — used to selectively trigger the test/validation
    guard (as opposed to the train guard, which fires on the very first call)."""

    def __init__(self, trigger_after: int):
        self.calls = 0
        self.trigger_after = trigger_after

    def apply(self, df, params):
        """Pass df through until `trigger_after` calls, then return an illegal SplitDataset."""
        self.calls += 1
        if self.calls > self.trigger_after:
            return SplitDataset(train=df, test=df)
        return df


def _transformer():
    """Build a StatefulTransformer using the AddOne calculator/applier pair."""
    return StatefulTransformer(_AddOneCalculator(), _AddOneApplier(), node_id="add_one")


def _fit_failure_transformer() -> StatefulTransformer:
    """Build a transformer that fails during calculator.fit."""
    return StatefulTransformer(_FailingCalculator(), _AddOneApplier(), node_id="fit_failure")


def _apply_failure_transformer() -> StatefulTransformer:
    """Build a transformer that fails during applier.apply."""
    return StatefulTransformer(_AddOneCalculator(), _FailingApplier(), node_id="apply_failure")


def _configurable_apply_failure_transformer() -> StatefulTransformer:
    """Build a transformer whose applier can fail on a later reuse run."""
    return StatefulTransformer(
        _ConfigEchoCalculator(),
        _MaybeFailApplier(),
        node_id="configurable_apply_failure",
    )


def _tracing_control_transformer() -> StatefulTransformer:
    """Build a transformer whose applier can stop tracemalloc mid-run."""
    return StatefulTransformer(
        _TracingControlCalculator(),
        _TracingControlApplier(),
        node_id="tracing_control",
    )


def _establish_historical_peak() -> int:
    """Create a large caller-owned historical peak that later work should not claim."""
    peak_marker = bytearray(8 * 1024 * 1024)
    _, peak_with_marker = tracemalloc.get_traced_memory()
    del peak_marker
    gc.collect()
    current, stable_peak = tracemalloc.get_traced_memory()
    assert stable_peak >= peak_with_marker
    assert stable_peak - current > 4 * 1024 * 1024
    return stable_peak


@pytest.fixture
def _isolated_tracemalloc_state() -> typing.Iterator[None]:
    """Keep tracemalloc ownership tests isolated from process-global state."""
    if tracemalloc.is_tracing():
        tracemalloc.stop()
    yield
    if tracemalloc.is_tracing():
        tracemalloc.stop()


_TRACEMALLOC_FAILURE_CASES = [
    pytest.param(_fit_failure_transformer, ValueError, "intentional fit failure", id="fit"),
    pytest.param(_apply_failure_transformer, RuntimeError, "intentional apply failure", id="apply"),
]


def test_fit_transform_on_plain_dataframe():
    """fit_transform on a bare DataFrame should fit + apply directly (no splits)."""
    df = pd.DataFrame({"a": [1, 2, 3]})
    transformer = _transformer()
    result = transformer.fit_transform(df, {"increment": 5})
    assert list(result["a"]) == [6, 7, 8]
    assert transformer.params == {"increment": 5}


def test_fit_transform_on_tuple_input():
    """fit_transform on an (X, y) tuple should pass the tuple through untouched in shape."""
    X = pd.DataFrame({"a": [1, 2, 3]})
    y = pd.Series([0, 1, 0])
    transformer = _transformer()
    result = transformer.fit_transform((X, y), {"increment": 2})
    assert isinstance(result, tuple)
    assert list(result[0]["a"]) == [3, 4, 5]


def test_fit_transform_uses_train_transform_hook_when_available():
    """fit_transform should prefer fit_transform_train for training rows."""
    result = StatefulTransformer(
        _TrainTransformCalculator(), _AddOneApplier(), node_id="train_hook"
    ).fit_transform(pd.DataFrame({"a": [1, 2]}), {})
    assert isinstance(result, pd.DataFrame)
    assert list(result["a"]) == [101, 102]


def test_fit_transform_on_split_dataset_applies_to_all_splits():
    """fit_transform on a SplitDataset should fit on train and apply to all three splits."""
    train = pd.DataFrame({"a": [1, 2]})
    test = pd.DataFrame({"a": [10, 20]})
    val = pd.DataFrame({"a": [100, 200]})
    dataset = SplitDataset(train=train, test=test, validation=val)
    transformer = _transformer()
    result = transformer.fit_transform(dataset, {"increment": 1})
    assert list(result.train["a"]) == [2, 3]
    assert list(result.test["a"]) == [11, 21]
    assert list(result.validation["a"]) == [101, 201]


def test_fit_transform_skips_test_apply_when_disabled():
    """apply_on_test=False should leave the test split untouched by the applier."""
    train = pd.DataFrame({"a": [1, 2]})
    test = pd.DataFrame({"a": [10, 20]})
    dataset = SplitDataset(train=train, test=test, validation=None)
    transformer = StatefulTransformer(
        _AddOneCalculator(), _AddOneApplier(), node_id="add_one", apply_on_test=False
    )
    result = transformer.fit_transform(dataset, {"increment": 1})
    assert isinstance(result, SplitDataset)
    assert isinstance(result.test, pd.DataFrame)
    assert isinstance(result.train, pd.DataFrame)
    assert list(result.test["a"]) == [10, 20]  # unchanged
    assert list(result.train["a"]) == [2, 3]


def test_fit_transform_skips_validation_when_none():
    """A dataset with validation=None should not invoke the applier on validation."""
    train = pd.DataFrame({"a": [1, 2]})
    test = pd.DataFrame({"a": [10, 20]})
    dataset = SplitDataset(train=train, test=test, validation=None)
    transformer = _transformer()
    result = transformer.fit_transform(dataset, {"increment": 1})
    assert result.validation is None


def test_fit_transform_records_profiling_metrics():
    """fit_transform should record fit_time, rows_in, and rows_out."""
    df = pd.DataFrame({"a": [1, 2, 3, 4]})
    transformer = _transformer()
    transformer.fit_transform(df, {"increment": 1})
    assert transformer.fit_time >= 0.0
    assert transformer.rows_in == 4
    assert transformer.rows_out == 4


def test_fit_transform_stops_tracing_it_started_after_success(
    _isolated_tracemalloc_state: None,
) -> None:
    """A successful transformer must stop only the tracing session it created."""
    assert not tracemalloc.is_tracing()

    transformer = _transformer()
    result = transformer.fit_transform(pd.DataFrame({"a": [1, 2]}), {"increment": 1})

    assert list(result["a"]) == [2, 3]
    assert transformer.rows_out == 2
    assert transformer.peak_memory_bytes >= 0
    assert not tracemalloc.is_tracing()


@pytest.mark.parametrize(
    ("config", "expected_rows_out", "expected_error", "message"),
    [
        pytest.param(
            {"increment": 1, "stop_tracing": True},
            2,
            None,
            None,
            id="success-after-stop",
        ),
        pytest.param(
            {"increment": 1, "stop_tracing": True, "fail_after_stop": True},
            0,
            RuntimeError,
            "intentional post-stop apply failure",
            id="failure-after-stop",
        ),
    ],
)
def test_fit_transform_handles_tracing_becoming_inactive_mid_run(
    _isolated_tracemalloc_state: None,
    config: dict[str, typing.Any],
    expected_rows_out: int,
    expected_error: type[Exception] | None,
    message: str | None,
) -> None:
    """Stopping tracemalloc mid-run must preserve transform/error semantics and zero fallback."""
    transformer = _tracing_control_transformer()
    df = pd.DataFrame({"a": [1, 2]})

    if expected_error is None:
        result = transformer.fit_transform(df, config)
        assert isinstance(result, pd.DataFrame)
        assert list(result["a"]) == [2, 3]
    else:
        with pytest.raises(expected_error, match=message):
            transformer.fit_transform(df, config)

    assert transformer.rows_out == expected_rows_out
    assert transformer.peak_memory_bytes == 0
    assert not tracemalloc.is_tracing()


@pytest.mark.parametrize(
    ("build_transformer", "expected_error", "message"),
    _TRACEMALLOC_FAILURE_CASES,
)
def test_fit_transform_stops_tracing_it_started_after_failure(
    _isolated_tracemalloc_state: None,
    build_transformer: typing.Callable[[], StatefulTransformer],
    expected_error: type[Exception],
    message: str,
) -> None:
    """A failing transformer must stop only the tracing session it created."""
    assert not tracemalloc.is_tracing()

    with pytest.raises(expected_error, match=message):
        build_transformer().fit_transform(pd.DataFrame({"a": [1, 2]}), {})

    assert not tracemalloc.is_tracing()


@pytest.mark.parametrize(
    ("build_transformer", "expected_error", "message"),
    _TRACEMALLOC_FAILURE_CASES,
)
def test_fit_transform_preserves_caller_owned_tracing_after_failure(
    _isolated_tracemalloc_state: None,
    build_transformer: typing.Callable[[], StatefulTransformer],
    expected_error: type[Exception],
    message: str,
) -> None:
    """A failing transformer must leave caller-owned tracing and peak state intact."""
    df = pd.DataFrame({"a": [1, 2]})
    transformer = build_transformer()
    tracemalloc.start()
    historical_peak = _establish_historical_peak()

    with pytest.raises(expected_error, match=message):
        transformer.fit_transform(df, {})

    assert tracemalloc.is_tracing()
    _, after_peak = tracemalloc.get_traced_memory()
    assert after_peak == historical_peak
    assert transformer.peak_memory_bytes == 0


def test_fit_transform_does_not_inherit_caller_peak_history_on_success(
    _isolated_tracemalloc_state: None,
) -> None:
    """Caller-owned tracing should report only new global peak growth since entry."""
    df = pd.DataFrame({"a": [1, 2]})
    tracemalloc.start()
    historical_peak = _establish_historical_peak()

    transformer = _transformer()
    result = transformer.fit_transform(df, {"increment": 1})

    assert list(result["a"]) == [2, 3]
    assert tracemalloc.is_tracing()
    _, after_peak = tracemalloc.get_traced_memory()
    assert after_peak == historical_peak
    assert transformer.peak_memory_bytes == 0


def test_fit_transform_clears_rows_out_before_reuse_failure() -> None:
    """A failed reuse run must not leak rows_out from a prior successful run."""
    transformer = _configurable_apply_failure_transformer()
    transformer.fit_transform(pd.DataFrame({"a": [1, 2, 3]}), {"increment": 1})
    assert transformer.rows_out == 3

    with pytest.raises(RuntimeError, match="intentional configurable apply failure"):
        transformer.fit_transform(
            pd.DataFrame({"a": [10, 20]}),
            {"increment": 1, "fail_apply": True},
        )

    assert transformer.rows_in == 2
    assert transformer.rows_out == 0


def test_transform_on_plain_dataframe_reuses_stored_params():
    """transform() on a bare DataFrame should reuse previously fitted params."""
    train_df = pd.DataFrame({"a": [1, 2, 3]})
    transformer = _transformer()
    transformer.fit_transform(train_df, {"increment": 10})

    new_df = pd.DataFrame({"a": [0, 0]})
    result = transformer.transform(new_df)
    assert list(result["a"]) == [10, 10]


def test_transform_on_raw_polars_dataframe_reuses_stored_params():
    """Regression test: `transform()` previously only checked
    `isinstance(dataset, pd.DataFrame)` before falling through to the
    tuple/SplitDataset branches, so a raw (unwrapped) `pl.DataFrame` passed
    directly crashed with `AttributeError: 'DataFrame' object has no
    attribute 'train'`. `StatefulTransformer` is a public API class
    (exported from `skyulf.preprocessing.__init__`), so this is a directly
    reachable call path, not just an internal implementation detail. Uses a
    polars-native applier since `_AddOneApplier` above is pandas-only.
    """
    import polars as pl

    class _PolarsAddOneApplier(BaseApplier):
        def apply(self, df, params):
            """Add params['increment'] to column 'a' via Polars' expression API."""
            return df.with_columns(pl.col("a") + params["increment"])

    train_df = pl.DataFrame({"a": [1, 2, 3]})
    transformer = StatefulTransformer(
        calculator=_AddOneCalculator(), applier=_PolarsAddOneApplier(), node_id="poly1"
    )
    transformer.fit_transform(train_df, {"increment": 10})

    new_df = pl.DataFrame({"a": [0, 0]})
    result = typing.cast(pl.DataFrame, transformer.transform(new_df))
    assert list(result["a"]) == [10, 10]


def test_transform_on_tuple_reuses_stored_params():
    """transform() on an (X, y) tuple should reuse the stored params."""
    train_df = pd.DataFrame({"a": [1, 2, 3]})
    transformer = _transformer()
    transformer.fit_transform(train_df, {"increment": 3})

    X = pd.DataFrame({"a": [0, 0]})
    y = pd.Series([1, 1])
    result = transformer.transform((X, y))
    assert list(result[0]["a"]) == [3, 3]


def test_transform_on_split_dataset_applies_to_all_splits():
    """transform() on a SplitDataset should apply the stored params to every split."""
    train = pd.DataFrame({"a": [1, 2]})
    test = pd.DataFrame({"a": [10, 20]})
    val = pd.DataFrame({"a": [100, 200]})
    dataset = SplitDataset(train=train, test=test, validation=val)
    transformer = _transformer()
    transformer.fit_transform(dataset, {"increment": 1})

    new_dataset = SplitDataset(train=train.copy(), test=test.copy(), validation=val.copy())
    result = transformer.transform(new_dataset)
    assert list(result.train["a"]) == [2, 3]
    assert list(result.test["a"]) == [11, 21]
    assert list(result.validation["a"]) == [101, 201]


def test_transform_skips_test_when_disabled():
    """transform() with apply_on_test=False should leave the test split untouched."""
    train = pd.DataFrame({"a": [1, 2]})
    test = pd.DataFrame({"a": [10, 20]})
    dataset = SplitDataset(train=train, test=test, validation=None)
    transformer = StatefulTransformer(
        _AddOneCalculator(), _AddOneApplier(), node_id="add_one", apply_on_test=False
    )
    transformer.fit_transform(dataset, {"increment": 1})

    new_dataset = SplitDataset(train=train.copy(), test=test.copy(), validation=None)
    result = transformer.transform(new_dataset)
    assert isinstance(result, SplitDataset)
    assert isinstance(result.test, pd.DataFrame)
    assert list(result.test["a"]) == [10, 20]


def test_base_calculator_infer_output_schema_defaults_to_none():
    """The default infer_output_schema on BaseCalculator should return None."""
    calc = _AddOneCalculator()
    assert calc.infer_output_schema(typing.cast(SkyulfSchema, None), {}) is None


class TestSplitDatasetGuard:
    """fit_transform() and transform() must both guard against an Applier illegally
    returning a SplitDataset for any split (train/test/validation) — scenarios loaded
    from ``tests/test_cases/preprocessing/preprocessing_base.json``.
    """

    @pytest.mark.parametrize(*_split_dataset_guard_cases)
    def test_raises_type_error(
        self, method: str, trigger_after: int, include_validation: bool
    ) -> None:
        train = pd.DataFrame({"a": [1, 2]})
        test = pd.DataFrame({"a": [10, 20]})
        validation = pd.DataFrame({"a": [100, 200]}) if include_validation else None
        dataset = SplitDataset(train=train, test=test, validation=validation)
        applier = _CountingSplitReturningApplier(trigger_after=trigger_after)
        transformer = StatefulTransformer(_AddOneCalculator(), applier, node_id="bad")

        if method == "transform":
            transformer.params = {}
            with pytest.raises(TypeError, match="not supported"):
                transformer.transform(dataset)
        else:
            with pytest.raises(TypeError, match="not supported"):
                transformer.fit_transform(dataset, {})
