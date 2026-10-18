"""Trip filters selected from config YAML as trip_rules.<factory>."""

from skyulf.preprocessing import filter_step


def zips_known(df, params):
    """Keep trips whose pickup and dropoff zones are both recorded."""
    return df[params["columns"]].notna().all(axis=1)


def known_zips(columns):
    """Fixed eligibility filter: both zone columns must be present."""
    return filter_step(
        "known_zips", zips_known, columns=list(columns), params={"columns": list(columns)}
    )
