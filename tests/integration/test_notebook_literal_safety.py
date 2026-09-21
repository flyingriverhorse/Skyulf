"""Exported code must preserve data without interpreting it as Python source."""

import ast
from pathlib import PureWindowsPath
from unittest.mock import Mock

import pandas as pd
import pytest

from backend.ml_pipeline._internal._routers._notebook_builders import (
    _feat_target_cells,
    _NodeIn,
    _to_py_literal,
    _train_test_cells,
    compact_load_cells,
    full_intro_cells,
)


def test_python_literal_preserves_json_words_and_escaped_strings():
    """JSON keywords inside strings must survive export exactly."""
    config = {
        "true": "false null true",
        "false": [True, False, None, {"null": 'a"b\\c\ntrue'}],
        "path": PureWindowsPath("C:/data/train.csv"),
    }
    result = ast.literal_eval(_to_py_literal(config))
    assert result == {**config, "path": str(config["path"])}


@pytest.mark.parametrize("mode", ["compact", "full"])
@pytest.mark.parametrize("path", ['C:\\data\\a"b.csv', "C:\\data\\", 'a\n"; injected = True; #'])
def test_load_cell_keeps_path_and_target_as_data(mode, path):
    """Generated load cells must accept arbitrary path and target string literals."""
    target = 'label"\\\ntrue false null'
    cells = compact_load_cells(path, target) if mode == "compact" else full_intro_cells(path)
    source = next("".join(c["source"]) for c in cells if "pd.read_csv(" in "".join(c["source"]))
    reader = Mock(return_value=pd.DataFrame({"x": [1]}))
    namespace = {"pd": Mock(read_csv=reader)}
    exec(compile(source, "exported-load", "exec"), namespace)
    reader.assert_called_once_with(path)
    assert "injected" not in namespace
    assert mode == "full" or namespace["TARGET_COLUMN"] == target


def test_feature_target_cell_preserves_unusual_column_name():
    """Quoted and multiline column names must remain usable in the exported split."""
    target = 'label"\\\ntrue false null'
    node = _NodeIn(
        node_id="target", step_type="feature_target_split", params={"target_column": target}
    )
    source = "".join(_feat_target_cells(node)[1]["source"])
    frame = pd.DataFrame({"x": [1, 2], target: [0, 1]})
    namespace = {"df": frame}
    exec(compile(source, "exported-target", "exec"), namespace)
    pd.testing.assert_series_equal(namespace["y"], frame[target])
    assert list(namespace["X"].columns) == ["x"]


@pytest.mark.parametrize("parameter", ["test_size", "random_state"])
def test_train_test_parameters_are_literals(parameter, monkeypatch):
    """Even invalid numeric parameters must reach validation as data, never execute."""
    payload = "__import__('builtins').print('injected')"
    node = _NodeIn(node_id="split", step_type="train_test_split", params={parameter: payload})
    frame = pd.DataFrame({"x": [1, 2]})
    split = Mock(return_value=(frame, frame, frame["x"], frame["x"]))
    monkeypatch.setattr("sklearn.model_selection.train_test_split", split)
    namespace = {"X": frame, "y": frame["x"]}
    exec(
        compile("".join(_train_test_cells(node)[1]["source"]), "exported-split", "exec"), namespace
    )
    assert split.call_args.kwargs[parameter] == payload
