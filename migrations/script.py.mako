"""Generic Alembic revision script."""

from alembic import op
import sqlalchemy as sa


def upgrade() -> None:
    ${upgrades if upgrades is not None else "pass"}


def downgrade() -> None:
    ${downgrades if downgrades is not None else "pass"}
