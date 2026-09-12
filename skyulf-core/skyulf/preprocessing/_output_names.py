"""Validate generated columns against the schema a preprocessing node will emit."""

from collections.abc import Iterable


def validate_generated_column_names(
    input_columns: Iterable[str],
    generated_columns: Iterable[str],
    *,
    dropped_columns: Iterable[str] = (),
    node_name: str,
) -> None:
    """Reject duplicate generated names and names shared with retained inputs.

    Names of dropped sources remain available for generated features. Check
    fitted names again at apply time because incoming frames can add columns.
    """
    seen = set(input_columns).difference(dropped_columns)
    collisions = set()
    for name in generated_columns:
        if name in seen:
            collisions.add(name)
        seen.add(name)
    if collisions:
        names = ", ".join(repr(name) for name in sorted(collisions))
        raise ValueError(
            f"{node_name}: generated output column names collide: {names}. "
            "Rename the conflicting input columns or adjust the encoding/flag settings."
        )
