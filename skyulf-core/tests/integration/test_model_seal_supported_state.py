"""Supported fitted estimators must retain semantic identity through persistence."""

import pickle
import subprocess
import sys

import numpy as np
import pandas as pd
import polars as pl
import pytest
from sklearn._loss import _loss  # ty: ignore[unresolved-import] - compiled sklearn module
from sklearn.datasets import make_classification
from sklearn.linear_model import (
    _sgd_fast,  # ty: ignore[unresolved-import] - compiled sklearn module
)

from skyulf.pipeline import SkyulfPipeline
from skyulf.pipeline.seal import artifact_digest


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize(
    "model_type, params",
    [
        ("gradient_boosting_classifier", {"n_estimators": 5}),
        ("hist_gradient_boosting_classifier", {"max_iter": 5}),
        ("sgd_classifier", {"max_iter": 50, "tol": None}),
        ("gradient_boosting_regressor", {"n_estimators": 5, "loss": "quantile", "alpha": 0.2}),
        ("hist_gradient_boosting_regressor", {"max_iter": 5}),
    ],
)
def test_fitted_model_card_and_seal_survive_reload(tmp_path, engine, model_type, params):
    """Boosting and SGD pipelines must export cards and preserve identity after save/load."""
    features, target = make_classification(n_samples=100, n_features=4, random_state=3)
    frame = pd.DataFrame(features, columns=["a", "b", "c", "d"])
    frame["target"] = target
    data = pl.from_pandas(frame) if engine == "polars" else frame
    pipeline = SkyulfPipeline(
        {
            "preprocessing": [],
            "modeling": {"type": model_type, "params": {**params, "random_state": 4}},
        }
    )
    metrics = pipeline.fit(data, target_column="target")
    assert "modeling_error" not in metrics
    assert pipeline.is_fitted()

    fingerprint = pipeline.fingerprint()
    predictions = pipeline.predict(frame.drop(columns="target"))
    path = tmp_path / "model.pkl"
    pipeline.save(str(path))
    restored = SkyulfPipeline.load(str(path))

    assert restored.export_model_card()["fingerprint"] == fingerprint
    assert restored.fingerprint() == pipeline.fingerprint()
    np.testing.assert_array_equal(restored.predict(frame.drop(columns="target")), predictions)


@pytest.mark.parametrize(
    "generator_type",
    [np.random.PCG64, np.random.PCG64DXSM, np.random.MT19937, np.random.Philox, np.random.SFC64],
)
def test_generator_digest_tracks_state_without_advancing_it(generator_type):
    """Sealing RNG state must preserve future draws while distinguishing advanced streams."""
    generator = np.random.Generator(generator_type(12))
    restored = pickle.loads(pickle.dumps(generator))

    before = artifact_digest(generator)

    assert artifact_digest(restored) == before
    np.testing.assert_array_equal(generator.integers(100, size=10), restored.integers(100, size=10))
    assert artifact_digest(generator) != before
    assert artifact_digest(restored) == artifact_digest(generator)


def test_generator_subclass_state_is_not_silently_ignored():
    """A customized RNG must fail closed rather than discard its extra behavior."""

    class ScaledGenerator(np.random.Generator):
        """Keep behavior-changing state outside the underlying bit generator."""

        multiplier: int = 1

        def integers(self, *args, **kwargs):
            """Scale draws using state the built-in RNG adapter cannot describe."""
            return super().integers(*args, **kwargs) * self.multiplier

    first = ScaledGenerator(np.random.PCG64(12))
    second = ScaledGenerator(np.random.PCG64(12))
    first.multiplier, second.multiplier = 1, 10
    assert not np.array_equal(first.integers(100, size=3), second.integers(100, size=3))
    with pytest.raises(TypeError, match="Unsupported generator"):
        artifact_digest(first)


def test_custom_bit_generator_is_not_silently_accepted():
    """Supporting NumPy's built-ins must not claim to understand custom RNG engines."""

    class CustomPCG(np.random.PCG64):
        """Represent a bit generator outside the supported exact library types."""

    with pytest.raises(TypeError, match="Unsupported generator"):
        artifact_digest(np.random.Generator(CustomPCG(12)))


@pytest.mark.parametrize(
    "constructor",
    [
        _loss.CyPinballLoss,
        _loss.CyHuberLoss,
        _loss.CyHalfTweedieLoss,
        _loss.CyHalfTweedieLossIdentity,
        _sgd_fast.Hinge,
        _sgd_fast.SquaredHinge,
        _sgd_fast.EpsilonInsensitive,
        _sgd_fast.SquaredEpsilonInsensitive,
    ],
)
def test_compiled_loss_digest_preserves_numeric_parameters(constructor):
    """Losses with different hidden constructor parameters must never share an identity."""
    first, second = constructor(0.2), constructor(0.7)

    assert artifact_digest(first) != artifact_digest(second)
    assert artifact_digest(first) == artifact_digest(pickle.loads(pickle.dumps(first)))


def test_multiclass_compiled_loss_is_distinct_from_binary_loss():
    """Stateless compiled losses must still encode their exact algorithm type."""
    assert artifact_digest(_loss.CyHalfMultinomialLoss()) != artifact_digest(
        _loss.CyHalfBinomialLoss()
    )


@pytest.mark.parametrize(
    "loss, expected",
    [
        (
            _loss.CyPinballLoss(0.2),
            "612e5f0eaabfc3bd34e373ae4591a42019ba6643ca8aaa8a4cc4a745364ecf79",
        ),
        (
            _loss.CyHalfMultinomialLoss(),
            "72786a618e3f701dc7c23e1265209e354bf5d5a2265ce754d335c46f5a5ba9b0",
        ),
    ],
)
def test_compiled_loss_digest_retains_its_cross_version_format(loss, expected):
    """Cython module names and reduction layouts must not change saved semantic identities."""
    assert artifact_digest(loss).hex() == expected


def test_unknown_reducible_object_is_not_silently_accepted():
    """Supporting known sklearn losses must not enable arbitrary pickle-based fingerprints."""

    class UnknownState:
        """Expose a reduction that the semantic seal must never invoke."""

        __slots__ = ()

        def __reduce__(self):
            """Fail if the seal falls back to arbitrary object serialization."""
            raise AssertionError("Unknown reductions must not run")

    with pytest.raises(TypeError, match="no canonical representation"):
        artifact_digest(UnknownState())


def test_compiled_state_digest_is_stable_in_a_fresh_process(tmp_path):
    """Compiled object addresses and Cython reconstruction helpers must not enter the digest."""
    artifact = {
        "generator": np.random.default_rng(19),
        "quantile": _loss.CyPinballLoss(0.2),
        "multiclass": _loss.CyHalfMultinomialLoss(),
        "hinge": _sgd_fast.Hinge(0.7),
    }
    path = tmp_path / "compiled-state.pkl"
    path.write_bytes(pickle.dumps(artifact))
    script = (
        "import pickle,sys; from skyulf.pipeline.seal import artifact_digest; "
        "from pathlib import Path; "
        "print(artifact_digest(pickle.loads(Path(sys.argv[1]).read_bytes())).hex())"
    )

    restored_digest = subprocess.check_output(
        [sys.executable, "-c", script, str(path)], text=True, timeout=30
    ).strip()

    assert restored_digest == artifact_digest(artifact).hex()
