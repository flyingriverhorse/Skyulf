"""Resolve Skyulf's automatic LightGBM sampling policy at the native boundary."""

from typing import Any


def resolve_sampling_frequency(params: dict[str, Any]) -> dict[str, Any]:
    """Enable automatic bagging outside GOSS while preserving explicit frequencies.

    ``None`` marks Skyulf's automatic frequency. Resolve it after candidate
    ``set_params`` calls, without changing estimator constructor state. GOSS
    uses gradient sampling and rejects ordinary row bagging. Native alias
    precedence decides the effective boosting mode and any explicit frequency.
    """
    if params.get("subsample_freq") is not None:
        return params

    if params.get("bagging_freq") is not None:
        params.pop("subsample_freq", None)
        return params

    from lightgbm.basic import (  # ty: ignore[unresolved-import]  # noqa: PLC0415 - optional boosting extra
        _ConfigAliases,
    )

    boosting = next(
        (params[name] for name in _ConfigAliases.get_sorted("boosting") if name in params),
        "gbdt",
    )
    is_goss = (
        str(boosting).lower() == "goss" or str(params.get("data_sample_strategy")).lower() == "goss"
    )
    params["subsample_freq"] = 0 if is_goss else 1
    return params
