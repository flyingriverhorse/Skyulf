import pandas as pd
from core.feature_engineering.nodes.data_consistency.replace_aliases import (
    apply_replace_aliases_typos,
)


def test_apply_replace_aliases_canonicalizes_country_alias():
    frame = pd.DataFrame(
        {
            "country": ["america", "USA", None],
            "other": [1, 2, 3],
        }
    )

    node = {
        "id": "alias-node",
        "data": {
            "config": {
                "columns": ["country"],
                "mode": "canonicalize_country_codes",
            }
        },
    }

    transformed, summary, signal = apply_replace_aliases_typos(frame, node)

    assert transformed.at[0, "country"] == "USA"
    assert transformed.at[1, "country"] == "USA"
    assert signal.replacements == 1
    assert "processed 1 column" in summary
