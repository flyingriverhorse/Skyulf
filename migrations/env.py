"""
© 2025 Murat Unsal — Skyulf Project
"""
import sys
from pathlib import Path

from alembic import context
from sqlalchemy import engine_from_config, pool
from logging.config import fileConfig

# Ensure application modules are importable
ROOT_PATH = Path(__file__).resolve().parents[1]
if str(ROOT_PATH) not in sys.path:
    sys.path.insert(0, str(ROOT_PATH))

from backend.config import get_settings  # noqa: E402
from backend.database.engine import Base  # noqa: E402
from backend.database import models  # noqa: F401,E402  # ensure model registration

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def _resolve_database_url() -> str:
    settings = get_settings()
    url = settings.DATABASE_URL

    if url.startswith("sqlite"):
        # Normalize to sync SQLite driver and absolute path for Alembic.
        _, _, path = url.partition("///")
        absolute_path = (ROOT_PATH / path).resolve()
        return f"sqlite:///{absolute_path.as_posix()}"

    if url.startswith("postgresql+asyncpg://"):
        return url.replace("postgresql+asyncpg://", "postgresql+psycopg2://")

    return url


def run_migrations_offline() -> None:
    url = _resolve_database_url()
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    url = _resolve_database_url()
    config.set_main_option("sqlalchemy.url", url)

    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
