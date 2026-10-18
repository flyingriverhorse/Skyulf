"""Row-local builtin callbacks copied into each YAML-owned saved project source.

These functions have local imports and no external helpers so their captured
definitions remain self-contained. Existing project scoring validates their
results and existing lint/probes admit them for partitioned execution.
"""


def _skyulf_yaml_required_values(frame, params):
    """Describe missing observed inputs in declaration order, retaining every row."""
    import pandas as pd  # noqa: PLC0415 - captured callback must be self-contained

    missing = frame.loc[:, params["columns"]].isna()
    return pd.Series(
        [
            "missing:" + ",".join(missing.columns[row].tolist()) if row.any() else None
            for row in missing.to_numpy()
        ],
        index=frame.index,
        dtype="string",
    )


def _skyulf_yaml_value_range(frame, params):
    """Accept finite numbers or numeric strings within saved inclusive bounds."""
    import numpy as np  # noqa: PLC0415 - captured callback must be self-contained
    import pandas as pd  # noqa: PLC0415 - captured callback must be self-contained

    column = params["column"]
    values = pd.to_numeric(frame[column], errors="coerce")
    accepted = values.notna() & np.isfinite(values)
    if params.get("minimum") is not None:
        accepted &= values >= params["minimum"]
    if params.get("maximum") is not None:
        accepted &= values <= params["maximum"]
    return pd.Series(None, index=frame.index, dtype="string").mask(
        ~accepted.fillna(False), f"outside_range:{column}"
    )


def _skyulf_yaml_prediction_band(frame, predictions, params):
    """Place each estimate in a saved band, including thresholds in the higher band."""
    import numpy as np  # noqa: PLC0415 - captured callback must be self-contained
    import pandas as pd  # noqa: PLC0415 - captured callback must be self-contained

    values = predictions[params.get("column", "prediction")]
    positions = np.searchsorted(params["thresholds"], values.to_numpy(dtype=float), side="right")
    bands = pd.Series(
        [params["labels"][position] for position in positions], index=frame.index, dtype="string"
    )
    return pd.DataFrame({params.get("output", "band"): bands.mask(values.isna())})


# Preserve these callback names in captured project source and saved model policies.
YAML_SCORING_CALLBACKS = {
    "required_values": _skyulf_yaml_required_values,
    "value_range": _skyulf_yaml_value_range,
    "prediction_band": _skyulf_yaml_prediction_band,
}
