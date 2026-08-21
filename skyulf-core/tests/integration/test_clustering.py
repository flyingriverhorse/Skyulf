"""Integration tests for the clustering nodes (kmeans, minibatch_kmeans, gaussian_mixture, birch).

All four calculators share the numeric-only mixin: they drop an optional
``reference_column``, auto-detect numeric features, fit the sklearn model, and
return the fitted model object (not a dict). Appliers are plain
:class:`SklearnApplier` subclasses, so clustering models expose
``predict`` (labels); only ``gaussian_mixture`` additionally exposes
``predict_proba`` (normalized responsibilities) because the other three
sklearn models lack the attribute. These tests drive the public
Calculator/Applier API end to end and verify hand-computed 3-blob cluster
assignments.
"""

import numpy as np
import pandas as pd
import pytest

from skyulf.modeling.clustering import (
    BirchApplier,
    BirchCalculator,
    GaussianMixtureApplier,
    GaussianMixtureCalculator,
    KMeansApplier,
    KMeansCalculator,
    MiniBatchKMeansApplier,
    MiniBatchKMeansCalculator,
)
from skyulf.registry import NodeRegistry

MODELS = {
    "kmeans": (KMeansCalculator, KMeansApplier),
    "minibatch_kmeans": (MiniBatchKMeansCalculator, MiniBatchKMeansApplier),
    "gaussian_mixture": (GaussianMixtureCalculator, GaussianMixtureApplier),
    "birch": (BirchCalculator, BirchApplier),
}


def _three_blobs(n_per_blob: int = 40, seed: int = 42) -> tuple[pd.DataFrame, pd.Series]:
    """Three tight Gaussian blobs with known true labels.

    Centers (0, 0), (5, 5), (10, 0) with sigma=0.05 are far apart relative to
    the noise, so every correct clustering algorithm must find 3 clusters
    aligned with the true labels.
    """
    rng = np.random.RandomState(seed)
    centers = [(0.0, 0.0), (5.0, 5.0), (10.0, 0.0)]
    x = np.concatenate([rng.normal(c[0], 0.05, n_per_blob) for c in centers])
    y = np.concatenate([rng.normal(c[1], 0.05, n_per_blob) for c in centers])
    true_labels = np.repeat([0, 1, 2], n_per_blob)
    return pd.DataFrame({"x": x, "y": y}), pd.Series(true_labels)


# ---- Registry ----------------------------------------------------------------


class TestRegistry:
    def test_all_four_models_registered(self) -> None:
        """Each registry name resolves to its Calculator/Applier pair."""
        assert NodeRegistry.get_calculator("kmeans") is KMeansCalculator
        assert NodeRegistry.get_applier("kmeans") is KMeansApplier
        assert NodeRegistry.get_calculator("minibatch_kmeans") is MiniBatchKMeansCalculator
        assert NodeRegistry.get_applier("minibatch_kmeans") is MiniBatchKMeansApplier
        assert NodeRegistry.get_calculator("gaussian_mixture") is GaussianMixtureCalculator
        assert NodeRegistry.get_applier("gaussian_mixture") is GaussianMixtureApplier
        assert NodeRegistry.get_calculator("birch") is BirchCalculator
        assert NodeRegistry.get_applier("birch") is BirchApplier


# ---- fit -> predict round trip -----------------------------------------------


@pytest.mark.parametrize("name", sorted(MODELS))
class TestFitPredict:
    def test_fit_returns_sklearn_model_and_predicts_labels(self, name: str) -> None:
        """fit() returns the fitted sklearn model; predict() returns a label Series."""
        calc_cls, appl_cls = MODELS[name]
        df, _ = _three_blobs()
        model = calc_cls().fit(df, None, {})

        # The modeling wrapper returns the fitted sklearn model, not a dict.
        assert not isinstance(model, dict)
        labels = appl_cls().predict(df, model)
        assert isinstance(labels, pd.Series)
        assert len(labels) == len(df)
        assert labels.nunique() == 3

    def test_predict_out_of_sample(self, name: str) -> None:
        """Predict works on fresh data that the model has never seen."""
        calc_cls, appl_cls = MODELS[name]
        df, _ = _three_blobs(seed=7)
        model = calc_cls().fit(df, None, {})

        # A single point at blob 1's center must land in blob 1's cluster.
        probe = pd.DataFrame({"x": [5.0], "y": [5.0]})
        (probe_label,) = appl_cls().predict(probe, model).tolist()

        # The probe must match the majority cluster of blob 1's rows.
        all_labels = appl_cls().predict(df, model)
        majority = all_labels.iloc[40:80].mode().iloc[0]
        assert probe_label == majority

    def test_predict_proba_behavior(self, name: str) -> None:
        """kmeans/minibatch/birch expose no probabilities; GMM returns responsibilities.

        GaussianMixture has a native ``predict_proba`` (normalized
        responsibilities), so the wrapper returns a DataFrame of shape
        (n, n_components) whose rows sum to 1; the other models lack the
        attribute entirely and the wrapper returns None.
        """
        calc_cls, appl_cls = MODELS[name]
        df, _ = _three_blobs()
        model = calc_cls().fit(df, None, {})
        proba = appl_cls().predict_proba(df, model)

        if name == "gaussian_mixture":
            assert isinstance(proba, pd.DataFrame)
            assert proba.shape == (len(df), 3)
            assert np.allclose(proba.sum(axis=1).to_numpy(), np.ones(len(df)), atol=1e-6)
            # The argmax responsibility must agree with predict()'s labels.
            labels = appl_cls().predict(df, model)
            assert (proba.to_numpy().argmax(axis=1) == labels.to_numpy()).all()
        else:
            assert proba is None


# ---- Hand-verified 3-blob clustering ------------------------------------------


@pytest.mark.parametrize("name", sorted(MODELS))
class TestThreeBlobRecovery:
    def test_recovers_three_well_separated_blobs(self, name: str) -> None:
        """All four models must recover 3 clusters on well-separated blobs."""
        calc_cls, appl_cls = MODELS[name]
        df, true_labels = _three_blobs()
        model = calc_cls().fit(df, None, {})
        labels = appl_cls().predict(df, model)

        assert labels.nunique() == 3

        # Purity: each true blob must map to exactly one cluster, and the
        # three blobs must land in three different clusters.
        for blob in (0, 1, 2):
            cluster_ids = labels.iloc[blob * 40 : (blob + 1) * 40].unique()
            assert len(cluster_ids) == 1, f"blob {blob} split across clusters {cluster_ids}"
        assert len({labels.iloc[0], labels.iloc[40], labels.iloc[80]}) == 3

    def test_blob_purity_is_perfect(self, name: str) -> None:
        """Every true blob maps to a single cluster with zero misassignments."""
        calc_cls, appl_cls = MODELS[name]
        df, true_labels = _three_blobs()
        model = calc_cls().fit(df, None, {})
        labels = appl_cls().predict(df, model)

        for blob in (0, 1, 2):
            assert labels.iloc[blob * 40 : (blob + 1) * 40].nunique() == 1
        # Each of the 3 clusters corresponds to exactly one true blob.
        for cluster in labels.unique():
            mask = labels == cluster
            assert true_labels[mask].nunique() == 1


# ---- reference_column --------------------------------------------------------


class TestReferenceColumn:
    def test_reference_column_dropped_at_fit(self) -> None:
        """A reference_column set in config is stashed on the model, not fitted."""
        df, _ = _three_blobs()
        df = df.assign(ref=["r00", "r01"] * 60)
        model = KMeansCalculator().fit(df, None, {"reference_column": "ref"})

        assert getattr(model, "reference_column_", None) == "ref"
        # The fitted model only saw the 2 numeric features.
        assert model.n_features_in_ == 2

    def test_predict_drops_reference_column_again(self) -> None:
        """The applier re-drops the reference column before predicting."""
        df, _ = _three_blobs()
        df = df.assign(ref=[f"ref-{i}" for i in range(len(df))])
        model = KMeansCalculator().fit(df, None, {"reference_column": "ref"})

        # Predict on a frame where the reference column has different dtype values.
        probe = pd.DataFrame({"x": [5.0], "y": [5.0], "ref": ["zzz"]})
        labels = KMeansApplier().predict(probe, model)
        assert len(labels) == 1

    def test_no_reference_column_means_no_attribute(self) -> None:
        """Without reference_column in config, no reference_column_ is set."""
        df, _ = _three_blobs()
        model = KMeansCalculator().fit(df, None, {})
        assert not hasattr(model, "reference_column_")

    def test_reference_column_string_dtype_is_harmless(self) -> None:
        """A string reference column is dropped even though it is non-numeric."""
        df, _ = _three_blobs(n_per_blob=20)
        df = df.assign(id=df.index.astype(str) + "_id")
        model = KMeansCalculator().fit(df, None, {"reference_column": "id"})
        assert model.n_features_in_ == 2


# ---- Non-numeric auto-drop ---------------------------------------------------


class TestNonNumericAutoDrop:
    def test_string_columns_are_auto_dropped(self) -> None:
        """Non-numeric columns (without explicit reference_column) are dropped."""
        df, _ = _three_blobs(n_per_blob=30)
        df["city"] = ["a", "b", "c"] * 30
        model = KMeansCalculator().fit(df, None, {})
        assert model.n_features_in_ == 2

        labels = KMeansApplier().predict(df, model)
        assert len(labels) == len(df)
        assert labels.nunique() == 3

    def test_mixed_numeric_and_text_still_clusters_numeric(self) -> None:
        """Clustering uses only the numeric subset; text is ignored."""
        df, _ = _three_blobs(n_per_blob=30)
        df["note"] = "n" * 3
        df["category"] = ["x", "y"] * 45
        model = BirchCalculator().fit(df, None, {})
        assert model.n_features_in_ == 2
        labels = BirchApplier().predict(df, model)
        assert labels.nunique() == 3


# ---- Out-of-sample sanity ----------------------------------------------------


class TestOutOfSample:
    def test_new_point_near_blob_center_gets_that_blobs_cluster(self) -> None:
        """A probe at a blob center predicts that blob's majority cluster."""
        calc, appl = MODELS["kmeans"]
        df, _ = _three_blobs(seed=11)
        model = calc().fit(df, None, {})
        labels = appl().predict(df, model)

        # Majority cluster per blob (blobs are index-aligned in the builder).
        majority = {
            blob: labels.iloc[blob * 40 : (blob + 1) * 40].mode().iloc[0] for blob in (0, 1, 2)
        }
        centers = {0: (0.0, 0.0), 1: (5.0, 5.0), 2: (10.0, 0.0)}
        for blob, (cx, cy) in centers.items():
            probe = pd.DataFrame({"x": [cx], "y": [cy]})
            (probe_label,) = appl().predict(probe, model).tolist()
            assert probe_label == majority[blob]

    def test_prediction_deterministic_given_random_state(self) -> None:
        """Refitting with the same data gives the same label assignment."""
        calc, appl = MODELS["kmeans"]
        df, _ = _three_blobs(seed=3)
        m1 = calc().fit(df, None, {})
        m2 = calc().fit(df, None, {})
        l1 = appl().predict(df, m1)
        l2 = appl().predict(df, m2)

        # Labels may be permuted between runs, so compare as a mapping.
        # With a fixed random_state the assignment should be identical.
        assert l1.tolist() == l2.tolist()
