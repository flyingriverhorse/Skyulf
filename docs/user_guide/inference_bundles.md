# Standalone inference bundles

The 0.9.0 development API packages a fitted feature pipeline and a Python
estimator with an explicit inference contract. It supports local pandas/Polars
prediction and provides the package boundary for the later Spark worker runner.
Importing `skyulf.inference` requires neither PySpark nor MLflow.

## Raw data and prepared features

Choose the input stage when building a bundle:

| `input_stage` | Expected input | Execution |
| --- | --- | --- |
| `raw` | Original feature columns | Apply saved FE once, then predict |
| `features` | Already transformed model features | Predict directly |

Both bundles contain the same frozen FE state and estimator. Their declared
stages differ. A stage is an explicit caller contract: numeric values alone
cannot reveal whether a column has already been scaled. Keep raw and prepared
data on the corresponding path.

```python
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from skyulf.data.dataset import SplitDataset
from skyulf.inference import build_bundle, load_bundle, predict_local, save_bundle
from skyulf.pipeline import SkyulfPipeline

training = pd.DataFrame({"amount": [1.0, 2.0, 3.0], "target": [10.0, 20.0, 30.0]})
pipeline = SkyulfPipeline({
    "preprocessing": [
        {"name": "fill", "transformer": "SimpleImputer",
         "params": {"columns": ["amount"], "strategy": "mean"}},
        {"name": "scale", "transformer": "StandardScaler",
         "params": {"columns": ["amount"]}},
    ],
    "modeling": {"type": "linear_regression"},
})
pipeline.fit(SplitDataset(train=training, test=training.head(0)), target_column="target")

raw = pd.DataFrame({"amount": [4.0, np.nan]})
bundle = build_bundle(pipeline, input_stage="raw", feature_order=("amount",))
with TemporaryDirectory() as temporary:
    destination = Path(temporary) / "model"
    save_bundle(bundle, destination)
    restored = load_bundle(destination)
    output = predict_local(raw, restored)
    np.testing.assert_allclose(output["prediction"], [40.0, 20.0], atol=1e-12)

prepared = pipeline.feature_engineer.transform(raw)
features_bundle = build_bundle(pipeline, input_stage="features", feature_order=("amount",))
np.testing.assert_allclose(predict_local(prepared, features_bundle)["prediction"],
                           output["prediction"], atol=1e-12)
```

The repository also includes `skyulf-core/examples/inference_bundle.py`. Run
it from the root with `.venv/Scripts/python.exe skyulf-core/examples/inference_bundle.py`.
Add `--bundle-dir path/to/new-model` to retain the artifact. An existing
destination is rejected, so an earlier package cannot be overwritten accidentally.

## Input and output contracts

A successful `SkyulfPipeline.fit` records two schemas: original training
features with the target excluded, and the actual features supplied to model
training. They contain names and dtypes only, never sample rows. A failed
replacement fit clears this metadata along with the fitted model.

`feature_order` must exactly match the recorded model order. `predict_local`
rejects missing, extra, duplicate or reordered columns and incompatible dtypes.
It does not sort columns or silently cast them. Equivalent pandas/Polars numeric
and boolean dtype labels are normalized; an integer feature is still distinct
from a floating-point feature. Select/cast columns explicitly before calling
when the source schema differs intentionally.

Supply feature columns only. Extra targets or identity columns are rejected.
The returned pandas DataFrame preserves a pandas input index; Polars input
receives a RangeIndex. The later Spark runner will carry explicit `FrameSpec`
keys separately from model features. Calling `predict_local` with a Spark frame
fails before any collection or local conversion.

Regression returns a float64 `prediction` column. Supported classifiers return
`prediction` and `probability_0`, `probability_1`, etc. `bundle.classes` records
the original labels in exactly that probability order; labels are not converted
into column names. Binary `positive_label` is the second estimator class, matching
the existing probability/threshold convention. Empty inputs return the declared
empty output schema.

Tuning's saved decision thresholds remain active by default, as in the existing
tuning applier. Thresholds saved by `pipeline.optimize_thresholds(...)` remain
opt-in: pass `use_tuned_thresholds=True` to `build_bundle`. The manifest retains
both saved sets and the selected source; opting in without saved pipeline
thresholds fails. Probabilities remain unchanged by threshold decisions.

## Package contents and loading

Each new directory contains three fixed-name files:

| File | Contents |
| --- | --- |
| `manifest.json` | Version, stage, ordered schemas, class/threshold metadata, runtime versions and digests |
| `features.json` | The supported fitted FeatureEngineer state |
| `model.pkl` | The frozen fitted sklearn estimator |

The bundle holds immutable metadata and bytes. Later mutation of the training
pipeline does not alter it. Building or predicting never fits a transformer or
estimator. The model payload excludes the pipeline, training reports, runtime
sessions and registry clients. Runtime requirements are a fixed list of package
versions, not a copy of environment variables or connection configuration.
An estimator can retain training observations as part of its own fitted state;
K-neighbors models are one example. Bundling preserves that state and its size
is subject to the model byte limit.

**Only load bundles from trusted producers.** The estimator uses pickle, which
can execute Python during deserialization. Checksums detect corruption; they
are not signatures and do not establish trust. Size, structural, payload-checksum
and runtime checks run before the pickle loader. Model class, feature width,
classes and fitted semantic digest are checked after deserialization.

The semantic digest covers the inference contract and learned state. Exact
payload checksums also protect transported bytes, but changing only the pickle
protocol does not change semantic identity. Neither digest promises portability
across arbitrary library versions.

Loading requires the same Python major/minor and exact recorded `skyulf-core`,
scikit-learn, NumPy and SciPy versions. pandas/Polars versions are recorded for
provenance; the input schema remains the execution check. There is no automatic
dependency installation. Prepare a matching environment before loading.

The optional `options=ExecutionOptions(...)` argument to build/save/load/predict
sets the existing budgets: by default 8 MiB for manifest plus FE bytes and 256 MiB
for estimator bytes. Oversized file reads and pickle serialization are bounded.
These are wire-size checks, not total process-memory guarantees. A wider input
batch or a loaded estimator can use more memory than its serialized form.

## Current support and legacy adapters

The first version accepts a successfully fitted standalone `SkyulfPipeline`
with recorded schemas, primitive numeric/boolean features, supported sklearn
regression/classification behavior and portable FE. Currently that FE is
SimpleImputer mean/constant and StandardScaler, including an empty chain.
Classifiers must expose class probabilities. Custom appliers and unsupported
FE nodes require a separate adapter; they are not serialized as an opaque
whole pipeline. Training-only split nodes in the FE chain are therefore not
portable yet; an explicit `SplitDataset` can supply training/test partitions.

Existing `SkyulfPipeline.save/load` pickle behavior is unchanged. A standalone
pipeline saved after schema capture can be loaded through that API and passed
to `build_bundle`. Historical standalone artifacts without captured schemas
must be fitted successfully again; the bundle does not invent missing column
names or types. Backend artifact dictionaries are rejected. Their explicit
adapter and threshold/label compatibility checks belong to SM-18.

This delivery verifies local packaging and inference. Distributed Python model
execution, MLflow/Unity Catalog, Databricks runtime validation, endpoints and
template generation remain later stages of the [Spark work](spark.md).
