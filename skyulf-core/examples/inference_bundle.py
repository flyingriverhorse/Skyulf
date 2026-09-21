"""Save and restore a standalone inference bundle, with no Spark or MLflow dependency."""

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from skyulf.data.dataset import SplitDataset
from skyulf.inference import build_bundle, load_bundle, predict_local, save_bundle
from skyulf.pipeline import SkyulfPipeline


def run_example(destination: Path) -> None:
    """Compare raw and prepared-feature predictions before and after persistence."""
    training = pd.DataFrame({"amount": [1.0, 2.0, 3.0], "target": [10.0, 20.0, 30.0]})
    pipeline = SkyulfPipeline(
        {
            "preprocessing": [
                {"name": "fill", "transformer": "SimpleImputer", "params": {"columns": ["amount"]}},
                {
                    "name": "scale",
                    "transformer": "StandardScaler",
                    "params": {"columns": ["amount"]},
                },
            ],
            "modeling": {"type": "linear_regression"},
        }
    )
    pipeline.fit(SplitDataset(train=training, test=training.head(0)), target_column="target")
    raw = pd.DataFrame({"amount": [4.0, np.nan]})
    bundle = build_bundle(pipeline, input_stage="raw", feature_order=("amount",))
    save_bundle(bundle, destination)
    restored = load_bundle(destination)
    result = predict_local(raw, restored)
    prepared = pipeline.feature_engineer.transform(raw)
    features_bundle = build_bundle(pipeline, input_stage="features", feature_order=("amount",))
    np.testing.assert_allclose(result.prediction, [40.0, 20.0], atol=1e-12)
    np.testing.assert_allclose(
        predict_local(prepared, features_bundle).prediction, result.prediction
    )
    assert restored.semantic_digest == bundle.semantic_digest
    print(f"Restored predictions: {result.prediction.to_list()}")
    print(f"Bundle semantic digest: {bundle.semantic_digest}")


def main() -> None:
    """Let the caller retain the artifact or use an isolated temporary example directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bundle-dir", type=Path, help="New directory to create; existing paths are rejected."
    )
    args = parser.parse_args()
    if args.bundle_dir is not None:
        run_example(args.bundle_dir)
    else:
        with TemporaryDirectory(prefix="skyulf-bundle-") as temporary:
            run_example(Path(temporary) / "model")


if __name__ == "__main__":
    main()
