"""Automated EDA plus the Polars/Arrow integration boundary.

Run from the repository root:
    python skyulf-core/examples/07_eda_and_polars.py

Install ``skyulf-core[viz]`` to use ``EDAVisualizer.summary()`` and
``EDAVisualizer.plot()`` in an interactive environment.  This script keeps
execution terminal-safe by printing the profile summary only.
"""

import polars as pl

from skyulf.engines.polars_engine import PolarsEngine
from skyulf.profiling.analyzer import EDAAnalyzer


def main() -> None:
    """Profile a Polars frame and show its Arrow representation without pandas."""
    frame = pl.DataFrame(
        {
            "customer_id": ["a", "b", "c", "d"],
            "spend": [10.0, 21.5, None, 30.0],
            "region": ["north", "south", "north", "west"],
            "converted": [False, True, False, True],
        }
    )
    profile = EDAAnalyzer(frame).analyze(target_col="converted")
    arrow_table = PolarsEngine.wrap(frame).to_arrow()

    print(f"Profiled columns: {profile.columns.keys()}")
    print(f"Recommendations: {len(profile.recommendations)}")
    print(f"Arrow schema: {arrow_table.schema}")
    print("EDA/Polars example completed without importing pandas.")


if __name__ == "__main__":
    main()
