"""Training must persist enough context for inference to describe its own inputs.

Two gaps surfaced from a real deployment: the bundled artifact recorded
``dropped_columns: []`` even though the graph had a Drop Columns node, and it
carried no dtypes at all, so the Inference page labelled every input "unknown".
"""

import pandas as pd
import pytest

from backend.data.catalog import FileSystemCatalog
from backend.ml_pipeline._execution.engine import PipelineEngine
from backend.ml_pipeline._execution.schemas import NodeConfig
from backend.ml_pipeline.artifacts.local import LocalArtifactStore
from skyulf.data.dataset import SplitDataset


@pytest.fixture
def engine(tmp_path):
    return PipelineEngine(
        LocalArtifactStore(str(tmp_path / "artifacts")), catalog=FileSystemCatalog()
    )


@pytest.fixture
def train_frame():
    return pd.DataFrame(
        {
            "SepalLengthCm": [1.0, 2.0],
            "SepalLengthCm_was_missing": [True, False],
            "rank": [1, 2],
            "Species": ["a", "b"],
        }
    )


class TestResolveTrainFeatureDtypes:
    def test_records_dtypes_for_feature_columns_only(self, engine, train_frame):
        dtypes = engine._resolve_train_feature_dtypes(
            train_frame, ["SepalLengthCm", "SepalLengthCm_was_missing", "rank"]
        )

        assert dtypes == {
            "SepalLengthCm": "float64",
            "SepalLengthCm_was_missing": "bool",
            "rank": "int64",
        }

    def test_covers_engineered_columns_not_present_in_the_raw_dataset(self, engine, train_frame):
        dtypes = engine._resolve_train_feature_dtypes(train_frame, ["SepalLengthCm_was_missing"])
        assert dtypes["SepalLengthCm_was_missing"] == "bool"

    def test_reads_the_train_split_of_a_split_dataset(self, engine, train_frame):
        data = SplitDataset(train=train_frame, test=train_frame, validation=None)
        assert engine._resolve_train_feature_dtypes(data, ["rank"]) == {"rank": "int64"}

    def test_returns_none_without_feature_columns(self, engine, train_frame):
        assert engine._resolve_train_feature_dtypes(train_frame, None) is None

    def test_returns_none_for_a_payload_that_is_not_tabular(self, engine):
        assert engine._resolve_train_feature_dtypes(object(), ["rank"]) is None

    def test_records_polars_dtypes_that_the_schema_builder_can_label(self, engine):
        import datetime

        import polars as pl

        from backend.ml_pipeline.deployment.service import DeploymentService

        train = pl.DataFrame(
            {
                "SepalLengthCm": [1.0, 2.0],
                "rank": [1, 2],
                "Species": ["a", "b"],
                "day": [datetime.date(2026, 1, 1), datetime.date(2026, 1, 2)],
            }
        )
        dtypes = engine._resolve_train_feature_dtypes(
            train, ["SepalLengthCm", "rank", "Species", "day"]
        )

        assert dtypes is not None
        labels = {name: DeploymentService._pretty_dtype(raw) for name, raw in dtypes.items()}
        assert labels == {
            "SepalLengthCm": "float",
            "rank": "integer",
            "Species": "text",
            "day": "date",
        }


class TestBundledDroppedColumns:
    def test_bundle_records_columns_dropped_upstream_of_the_training_node(self, engine):
        engine._node_configs = {
            "drop": NodeConfig(
                node_id="drop",
                step_type="DropMissingColumns",
                params={"columns": ["Id"]},
                inputs=[],
            ),
            "train": NodeConfig(node_id="train", step_type="training", params={}, inputs=["drop"]),
        }

        assert engine._upstream_dropped_columns(engine._node_configs["train"]) == ["Id"]
