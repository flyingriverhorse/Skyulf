import pytest
import pandas as pd

from core.feature_engineering.preprocessing.feature_generation.feature_math import (
    FeatureMathConfig,
    FeatureMathOperation,
    _apply_similarity_operation,
)


@pytest.mark.parametrize(
    "method, normalize",
    [
        ("token_sort_ratio", True),
        ("token_set_ratio", True),
        ("ratio", False),
    ],
)
def test_apply_similarity_operation_supports_methods(method, normalize):
    frame = pd.DataFrame(
        {
            "text_a": ["hello world", "hello", "good night"],
            "text_b": ["world hello", "hola", "good evening"],
        }
    )

    operation = FeatureMathOperation(
        operation_id=f"sim_{method}",
        operation_type="similarity",
        method=method,
        input_columns=["text_a", "text_b"],
        output_column=f"similarity_score_{method}",
        normalize=normalize,
    )

    config = FeatureMathConfig(allow_overwrite=True)

    created, message = _apply_similarity_operation(frame, operation, config)

    assert created == [operation.output_column]
    assert method in message

    scores = frame[operation.output_column]
    if normalize:
        assert scores.between(0.0, 1.0).all()
    else:
        assert scores.between(0.0, 100.0).all()

    assert scores.iloc[0] > scores.iloc[1]
