import asyncio
from core.feature_engineering.schemas import TrainingJobCreate, FeatureGraph
from core.feature_engineering.modeling.training.jobs import create_training_job
from core.feature_engineering.modeling.training.tasks import dispatch_training_job
from core.database.engine import init_db, create_tables
from core.database.models import get_database_session

DATASET_SOURCE_ID = "4ba2100e"
PIPELINE_ID = "4ba2100e_d15eaefb"
TRAINING_NODE_ID = "train-model-1"
USER_ID = 5

TRAINING_GRAPH = FeatureGraph(
    nodes=[
        {
            "id": "dataset-source",
            "type": "featureNode",
            "data": {
                "catalogType": "dataset-source",
                "label": "Dataset input",
                "isDataset": True,
                "config": {
                    "dataset_source_id": DATASET_SOURCE_ID,
                    "source_id": DATASET_SOURCE_ID,
                },
            },
        },
        {
            "id": "split-node-1",
            "type": "featureNode",
            "data": {
                "catalogType": "train_test_split",
                "config": {
                    "test_size": 0.2,
                    "validation_size": 0.0,
                    "shuffle": True,
                    "random_state": 42,
                    "stratify": True,
                    "target_column": "Result",
                },
            },
        },
        {
            "id": TRAINING_NODE_ID,
            "type": "featureNode",
            "data": {
                "catalogType": "train_model_draft",
                "config": {
                    "target_column": "Result",
                    "problem_type": "classification",
                    "model_type": "random_forest_classifier",
                    "hyperparameters": {
                        "n_estimators": 50,
                        "max_depth": None,
                    },
                },
            },
        },
    ],
    edges=[
        {
            "id": "edge-1",
            "source": "dataset-source",
            "target": "split-node-1",
            "sourceHandle": "dataset-source-output",
            "targetHandle": "split-node-1-input",
        },
        {
            "id": "edge-2",
            "source": "split-node-1",
            "target": TRAINING_NODE_ID,
            "sourceHandle": "split-node-1-output",
            "targetHandle": "train-model-1-input",
        },
    ],
)


async def main() -> None:
    await init_db()
    await create_tables()

    payload = TrainingJobCreate(
        dataset_source_id=DATASET_SOURCE_ID,
        pipeline_id=PIPELINE_ID,
        node_id=TRAINING_NODE_ID,
        model_types=["random_forest_classifier"],
        hyperparameters={"n_estimators": 50},
        metadata={"initiated_by": "script"},
        run_training=True,
        graph=TRAINING_GRAPH,
    )

    async with get_database_session(expire_on_commit=False) as session:
        job = await create_training_job(session, payload, user_id=USER_ID)
        dispatch_training_job(job.id)
        print({"job_id": job.id, "status": job.status, "version": job.version})


if __name__ == "__main__":
    asyncio.run(main())

