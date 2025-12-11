# Execution Engine

The `PipelineEngine` orchestrates the end-to-end execution of machine learning pipelines, from data loading to model training.

## PipelineEngine

The engine is responsible for:
1.  **Data Loading**: Loading the dataset from the source.
2.  **Splitting**: Splitting data into train/test sets (if not already split).
3.  **Feature Engineering**: Executing the DAG of preprocessing steps.
4.  **Modeling**: Training the model using the processed data.
5.  **Artifact Management**: Storing intermediate and final artifacts.

### Usage

```python
from core.ml_pipeline.execution.schemas import PipelineConfig, NodeConfig

config = PipelineConfig(
    pipeline_id="demo_pipeline",
    nodes=[
        NodeConfig(node_id="load", step_type="data_loader", params={"path": "data.csv"}),
        NodeConfig(node_id="train", step_type="model_training", inputs=["load"], params={"model": "logreg"})
    ],
    metadata={"owner": "docs"}
)

print(config.pipeline_id)
print([node.node_id for node in config.nodes])
```

## Job Management

The `core.ml_pipeline.execution.jobs` module handles the asynchronous execution of these pipelines using Celery. It manages job states (pending, running, completed, failed) and persistence to the database.

