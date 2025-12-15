import numpy as np
import pandas as pd
import pytest
from skyulf.preprocessing.imputation import (
    IterativeImputerApplier,
    IterativeImputerCalculator,
    KNNImputerApplier,
    KNNImputerCalculator,
    SimpleImputerApplier,
    SimpleImputerCalculator,
)


def test_simple_imputer_mean():
    df = pd.DataFrame({"A": [1, 2, np.nan, 4, 5], "B": [10, 20, 30, 40, 50]})

    config = {"strategy": "mean", "columns": ["A"]}

    calc = SimpleImputerCalculator()
    result = calc.fit(df, config)

    assert result["type"] == "simple_imputer"
    assert result["strategy"] == "mean"
    assert result["fill_values"]["A"] == 3.0

    applier = SimpleImputerApplier()
    df_out = applier.apply(df, result)

    assert df_out["A"].isnull().sum() == 0
    assert df_out["A"][2] == 3.0


def test_simple_imputer_constant():
    df = pd.DataFrame({"A": [1, 2, np.nan, 4, 5], "B": ["x", "y", np.nan, "z", "w"]})

    config = {"strategy": "constant", "fill_value": 99, "columns": ["A"]}

    calc = SimpleImputerCalculator()
    result = calc.fit(df, config)

    assert result["fill_values"]["A"] == 99

    applier = SimpleImputerApplier()
    df_out = applier.apply(df, result)
    assert df_out["A"][2] == 99


def test_knn_imputer():
    df = pd.DataFrame(
        {"A": [1, 2, np.nan, 4, 5], "B": [1, 2, 3, 4, 5], "C": [1, 2, 3, 4, 5]}
    )

    config = {"n_neighbors": 2, "columns": ["A", "B", "C"]}

    calc = KNNImputerCalculator()
    result = calc.fit(df, config)

    assert result["type"] == "knn_imputer"
    assert "imputer_object" in result

    applier = KNNImputerApplier()
    df_out = applier.apply(df, result)

    assert df_out["A"].isnull().sum() == 0
    # With neighbors 2 and 4 (values 2 and 4), average is 3
    assert abs(df_out["A"][2] - 3.0) < 0.1


def test_iterative_imputer():
    df = pd.DataFrame(
        {"A": [1, 2, np.nan, 4, 5], "B": [1, 2, 3, 4, 5], "C": [1, 2, 3, 4, 5]}
    )

    config = {"max_iter": 5, "estimator": "bayesian_ridge", "columns": ["A", "B", "C"]}

    calc = IterativeImputerCalculator()
    result = calc.fit(df, config)

    assert result["type"] == "iterative_imputer"
    assert "imputer_object" in result

    applier = IterativeImputerApplier()
    df_out = applier.apply(df, result)

    assert df_out["A"].isnull().sum() == 0
    # Should be close to 3
    assert abs(df_out["A"][2] - 3.0) < 0.1


if __name__ == "__main__":
    # Manual run if pytest not available or for quick check
    try:
        test_simple_imputer_mean()
        print("Simple Imputer Mean: PASS")
        test_simple_imputer_constant()
        print("Simple Imputer Constant: PASS")
        test_knn_imputer()
        print("KNN Imputer: PASS")
        test_iterative_imputer()
        print("Iterative Imputer: PASS")
    except Exception as e:
        print(f"FAILED: {e}")
        import traceback

        traceback.print_exc()
