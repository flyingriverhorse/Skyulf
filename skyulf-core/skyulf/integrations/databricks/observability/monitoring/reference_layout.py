"""Preserve logical training populations across immutable reference storage formats."""

from typing import Any

import pandas as pd

from ...shared._contracts import column_name
from .monitoring_config import json_digest, qualified_name


def combine_reference_population(
    train: pd.DataFrame,
    holdout: pd.DataFrame,
    columns: list[str],
    split_strategy: str,
) -> tuple[pd.DataFrame, str]:
    """Keep train membership and duplicate rows without consuming a caller's feature name."""
    membership = "is_train"
    while membership.casefold() in {name.casefold() for name in columns}:
        membership = "_" + membership
    population = train.loc[:, columns].assign(**{membership: True})
    if split_strategy != "temporal":
        population = pd.concat(
            [population, holdout.loc[:, columns].assign(**{membership: False})],
            ignore_index=True,
        )
    return population, membership


def reference_population_records(receipt: dict[str, Any]) -> dict[str, dict]:
    """Expose train/seen/source records while preserving format-1 receipt compatibility."""
    layout = receipt.get("format", 1)
    tables = receipt["tables"]
    if layout == 1:
        return tables
    if layout != 2:
        raise ValueError("Unsupported prepared monitoring reference format.")
    membership = receipt["membership_column"]
    column_name(membership)
    population = tables["population"] | {"membership_column": membership}
    return {
        "train": population | {"train_only": True},
        "seen": population | {"train_only": False},
        "source": tables["source"],
    }


def source_snapshot_identity(
    table_id: str, version: int, source_evidence: dict, schema: dict
) -> dict:
    """Bind sharing to verified row multiplicity, selected columns and exact stored types.

    Callers must validate the source evidence against the materialized population first.
    Its content digest ignores row order but retains duplicate counts. Model/dataset ids
    are deliberately absent: only equal stored source populations may share a snapshot.
    """
    return {
        "format": 2,
        "table_id": table_id,
        "version": version,
        "columns": source_evidence["columns"],
        "rows": source_evidence["rows"],
        "content_sha256": source_evidence["content_sha256"],
        "schema": schema,
    }


def shared_source_table_name(namespace: str, identity: dict) -> str:
    """Name a shared source by its complete verified immutable population identity."""
    return qualified_name(f"{namespace}.monitor_src_{json_digest(identity)}")
