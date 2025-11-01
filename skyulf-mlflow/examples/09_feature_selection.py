"""
Example: Feature Selection with the Skyulf-MLFlow FeatureSelector

Demonstrates multiple selection strategies (SelectKBest, SelectPercentile,
SelectFromModel, VarianceThreshold) and shows how metadata and column
lists are exposed.
"""

import numpy as np
import pandas as pd

from skyulf_mlflow_library.features.selection import FeatureSelector

np.set_printoptions(precision=3, suppress=True)

print("=" * 80)
print("Skyulf-MLFlow: Feature Selection Showcase")
print("=" * 80)
print()

# ---------------------------------------------------------------------------
# Create synthetic classification dataset with informative and noisy features
# ---------------------------------------------------------------------------
RNG = np.random.default_rng(42)
N_SAMPLES = 200

signal_a = RNG.normal(size=N_SAMPLES)
signal_b = RNG.normal(size=N_SAMPLES)
noise_c = RNG.normal(size=N_SAMPLES)
noise_d = RNG.normal(size=N_SAMPLES)
weak_signal = signal_a * 0.25 + RNG.normal(scale=0.5, size=N_SAMPLES)

linear_combination = 1.2 * signal_a + 0.8 * signal_b - 0.3 * noise_c + RNG.normal(scale=0.75, size=N_SAMPLES)
target = (linear_combination > 0.5).astype(int)

frame = pd.DataFrame(
    {
        "signal_a": signal_a,
        "signal_b": signal_b,
        "noise_c": noise_c,
        "noise_d": noise_d,
        "weak_signal": weak_signal,
        "target": target,
        "customer_id": np.arange(N_SAMPLES),
    }
)

CANDIDATE_COLUMNS = ["signal_a", "signal_b", "noise_c", "noise_d", "weak_signal"]

print("Base dataset preview:")
print(frame.head())
print()


def preview_selector(name: str, selector: FeatureSelector, dataframe: pd.DataFrame) -> pd.DataFrame:
    """Fit, transform, and summarize the selector."""

    print(f"--- {name} ---")
    transformed = selector.fit_transform(dataframe, dataframe["target"])

    print("Summary:", selector.metadata.get("summary"))
    print("Selected columns:", selector.get_selected_columns())
    print("Dropped columns:", selector.get_dropped_columns())
    print("Output columns:", transformed.columns.tolist())

    summaries = selector.get_feature_summaries()
    if summaries:
        print("Top feature scores:")
        for summary in summaries[:3]:
            print(
                f"  - {summary.column}: selected={summary.selected} "
                f"score={summary.score if summary.score is not None else 'na'} "
                f"p={summary.p_value if summary.p_value is not None else 'na'}"
            )
    print()
    return transformed


def demo_select_k_best():
    selector = FeatureSelector(
        columns=CANDIDATE_COLUMNS,
        target_column="target",
        method="select_k_best",
        score_func="f_classif",
        k=3,
        drop_unselected=True,
        auto_detect=False,
    )
    preview_selector("SelectKBest (classification, k=3)", selector, frame)


def demo_select_percentile():
    selector = FeatureSelector(
        columns=CANDIDATE_COLUMNS,
        target_column="target",
        method="select_percentile",
        score_func="f_classif",
        percentile=40,
        drop_unselected=False,
        auto_detect=False,
    )
    transformed = preview_selector("SelectPercentile (keep all columns, mark support)", selector, frame)
    print("Transformed preview (first rows):")
    print(transformed.head())
    print()


def demo_select_from_model():
    selector = FeatureSelector(
        columns=CANDIDATE_COLUMNS,
        target_column="target",
        method="select_from_model",
        estimator="random_forest",
        drop_unselected=True,
        auto_detect=False,
    )
    preview_selector("SelectFromModel (RandomForest)", selector, frame)


def demo_variance_threshold():
    # Add a near-constant column to highlight variance filtering
    augmented = frame.copy()
    augmented["constant_like"] = 1.0 + RNG.normal(scale=1e-4, size=N_SAMPLES)

    selector = FeatureSelector(
        columns=CANDIDATE_COLUMNS + ["constant_like"],
        method="variance_threshold",
        threshold=1e-3,
        drop_unselected=True,
        auto_detect=False,
    )
    preview_selector("VarianceThreshold (drop near-constant)", selector, augmented)


if __name__ == "__main__":
    demo_select_k_best()
    demo_select_percentile()
    demo_select_from_model()
    demo_variance_threshold()

    print("=" * 80)
    print("Feature selection demo complete!")
    print("=" * 80)
