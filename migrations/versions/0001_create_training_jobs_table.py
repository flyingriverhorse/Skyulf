"""create training_jobs table

Revision ID: 0001
Revises:
Create Date: 2025-10-24
"""

from alembic import op
import sqlalchemy as sa
revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "training_jobs",
        sa.Column("id", sa.String(length=64), nullable=False),
        sa.Column("created_at", sa.DateTime(), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=False),
        sa.Column("updated_at", sa.DateTime(), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=False),
        sa.Column("pipeline_id", sa.String(length=150), nullable=False),
        sa.Column("node_id", sa.String(length=150), nullable=False),
        sa.Column("dataset_source_id", sa.String(length=100), nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=True),
        sa.Column("status", sa.String(length=20), nullable=False),
        sa.Column("version", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("model_type", sa.String(length=100), nullable=False),
        sa.Column("hyperparameters", sa.JSON(), nullable=True),
        sa.Column("metadata", sa.JSON(), nullable=True),
        sa.Column("metrics", sa.JSON(), nullable=True),
        sa.Column("graph", sa.JSON(), nullable=False),
        sa.Column("artifact_uri", sa.String(length=500), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("started_at", sa.DateTime(), nullable=True),
        sa.Column("finished_at", sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], name="fk_training_jobs_user_id"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_training_jobs_dataset_source_id", "training_jobs", ["dataset_source_id"], unique=False)
    op.create_index("ix_training_jobs_node_id", "training_jobs", ["node_id"], unique=False)
    op.create_index("ix_training_jobs_pipeline_id", "training_jobs", ["pipeline_id"], unique=False)
    op.create_index("ix_training_jobs_status", "training_jobs", ["status"], unique=False)
    op.create_index("ix_training_jobs_user_id", "training_jobs", ["user_id"], unique=False)


def downgrade() -> None:
    op.drop_index("ix_training_jobs_user_id", table_name="training_jobs")
    op.drop_index("ix_training_jobs_status", table_name="training_jobs")
    op.drop_index("ix_training_jobs_pipeline_id", table_name="training_jobs")
    op.drop_index("ix_training_jobs_node_id", table_name="training_jobs")
    op.drop_index("ix_training_jobs_dataset_source_id", table_name="training_jobs")
    op.drop_table("training_jobs")
