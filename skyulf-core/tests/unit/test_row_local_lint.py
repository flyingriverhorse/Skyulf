"""Heuristic lint for batch-level code in apply-time project callbacks."""

import importlib.util
import uuid

import pytest

from skyulf.inference.row_local_lint import lint_callbacks

SOURCE = '''
import numpy as np

CACHE = {}
LIMITS = [1.0]


def helper(frame):
    """Same-module helper reached from apply()."""
    return frame["x"].rank()


def learn(df, y):
    """Learning on training rows may use batch statistics."""
    return {"mean": float(df["x"].mean()), "max": float(df["x"].max())}


def mean_apply(df, state):
    return df["x"].fillna(df["x"].mean())


def sum_apply(df, state):
    return df.sum()


def row_sum_apply(df, state):
    return df[["x", "y"]].sum(axis=1) - state["mean"]


def alias_apply(df, state):
    column = df["x"]
    return column - np.mean(column)


def state_apply(df, state):
    return df["x"].clip(upper=max(state["values"])) + np.array(state["values"]).max()


def helper_apply(df, state):
    return helper(df)


def length_apply(df, state):
    return df["x"] * len(df)


def shape_apply(df, state):
    return df["x"] * df.shape[0]


def any_apply(df, state):
    return df["x"] * 2.0 if (df["x"] > 1e6).any() else df["x"]


def global_apply(df, state):
    if "mean" not in CACHE:
        CACHE["mean"] = state["mean"]
    return df["x"] - CACHE["mean"]


def append_apply(df, state):
    LIMITS.append(1.0)
    return df["x"]


def declared_global_apply(df, state):
    global LIMITS
    LIMITS = [2.0]
    return df["x"]


def local_shadow_apply(df, state):
    CACHE = {}
    CACHE["x"] = 1
    return df["x"] * CACHE["x"]


COLUMN = "z"


def constant_key_apply(df, state):
    out = df.copy()
    out[COLUMN] = out["x"] * 2.0
    return out[COLUMN]


import os


class Cfg:
    mean = 0.0


def class_attr_learn(df, y):
    Cfg.mean = float(df["x"].mean())
    return {}


def fn_attr_learn(df, y):
    constant_key_apply.mean = 1.0
    return {}


def environ_learn(df, y):
    os.environ["SKYULF_MEAN"] = "1"
    return {}


def default_cache_apply(df, state, _cache=[]):
    _cache.append(1)
    return df["x"]


def globals_apply(df, state):
    g = globals()
    g["_m"] = 1.0
    globals()["_n"] = 2.0
    return df["x"]


def str_count_apply(df, state):
    return df["s"].str.count(",") + df["s"].str.len()


def guards_apply(df, state):
    if len(df) == 0:
        return df["x"]
    if (df["x"] < 0).any():
        raise ValueError("negative")
    if not df["x"].between(0, 1).all():
        raise ValueError("range")
    return df["x"] * df.shape[1]


def positional_axis_apply(df, state):
    return df[["a", "b"]].sum(1) + np.mean(df[["a", "b"]].to_numpy(), 1)


def repeated_constant_apply(df, state):
    return pd.Series([None] * len(df.index), index=df.index, dtype=object)


def conditional_any_apply(df, state):
    if (df["x"] > 1e6).any():
        return df["x"] * 3.0
    return df["x"]
'''


@pytest.fixture(scope="module")
def module(tmp_path_factory):
    """Load the fixture from a real file, as local preview does."""
    path = tmp_path_factory.mktemp("lint") / "steps.py"
    path.write_text(SOURCE, encoding="utf-8")
    spec = importlib.util.spec_from_file_location(f"lint_fixture_{uuid.uuid4().hex}", path)
    assert spec is not None and spec.loader is not None
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


def _calls(module, name, learn=()):
    """Return the flagged call texts for one apply function."""
    findings = lint_callbacks([getattr(module, name)], [getattr(module, n) for n in learn])
    return [(finding.problem, finding.call) for finding in findings]


def test_mean_and_sum_without_axis_are_flagged(module):
    """Column statistics over the batch differ per Spark partition."""
    assert _calls(module, "mean_apply") == [("reads a batch statistic", 'df["x"].mean()')]
    assert _calls(module, "sum_apply") == [("reads a batch statistic", "df.sum()")]


def test_row_wise_sum_with_axis_one_is_accepted(module):
    """An explicit axis=1 call works within each row and is not a batch statistic."""
    assert _calls(module, "row_sum_apply") == []


def test_learn_may_use_batch_statistics(module):
    """Learning from training rows is the point of learn(); only apply paths are linted."""
    assert _calls(module, "row_sum_apply", learn=("learn",)) == []


def test_aliases_numpy_and_same_module_helpers_are_followed(module):
    """Simple aliases, numpy reductions and helpers called from apply are all apply-time."""
    assert _calls(module, "alias_apply") == [("reads a batch statistic", "np.mean(column)")]
    assert _calls(module, "helper_apply") == [("depends on other rows", 'frame["x"].rank()')]


def test_saved_state_reductions_are_not_flagged(module):
    """Reducing learned state reads no batch rows and must not be a false alarm."""
    assert _calls(module, "state_apply") == []


def test_batch_size_and_batch_conditions_are_flagged(module):
    """len(df), df.shape and .any() make results depend on the whole batch."""
    assert _calls(module, "length_apply") == [("reads the batch size", "len(df)")]
    assert _calls(module, "shape_apply") == [("reads the batch size", "df.shape")]
    assert _calls(module, "any_apply") == [("reads a batch statistic", '(df["x"] > 1e6).any()')]


def test_module_global_writes_are_flagged(module):
    """Caches in module globals differ per worker process and are never row-local."""
    assert _calls(module, "global_apply") == [("writes module state", 'CACHE["mean"]')]
    assert _calls(module, "append_apply") == [("writes module state", "LIMITS.append(1.0)")]
    assert _calls(module, "declared_global_apply") == [("writes module state", "global LIMITS")]


def test_local_names_shadowing_globals_are_not_flagged(module):
    """A local dict with a module-level name is per call, not module state."""
    assert _calls(module, "local_shadow_apply") == []
    assert _calls(module, "constant_key_apply") == []


def test_findings_carry_file_and_line(module):
    """Users get a precise location for every finding."""
    (finding,) = lint_callbacks([module.mean_apply])
    assert finding.file == "steps.py"
    assert SOURCE.splitlines()[finding.line - 1].strip().startswith('return df["x"].fillna(')


def test_learn_time_process_state_writes_are_flagged(module):
    """Class/function attributes and os.environ set in learn() survive only in that process."""
    for name, call in (
        ("class_attr_learn", "Cfg.mean"),
        ("fn_attr_learn", "constant_key_apply.mean"),
        ("environ_learn", 'os.environ["SKYULF_MEAN"]'),
    ):
        findings = lint_callbacks([module.row_sum_apply], [getattr(module, name)])
        assert [(f.problem, f.call) for f in findings] == [("writes module state", call)], name


def test_mutable_defaults_and_globals_writes_are_flagged(module):
    """A mutated default list or a globals() write is per-process memory."""
    assert _calls(module, "default_cache_apply") == [("writes module state", "_cache.append(1)")]
    assert _calls(module, "globals_apply") == [
        ("writes module state", 'g["_m"]'),
        ("writes module state", 'globals()["_n"]'),
    ]


def test_row_wise_idioms_are_not_flagged(module):
    """String accessors, input guards, column counts and positional axis 1 are row-local."""
    for name in (
        "str_count_apply",
        "guards_apply",
        "positional_axis_apply",
        "repeated_constant_apply",
    ):
        assert _calls(module, name) == [], name


def test_batch_condition_that_changes_results_is_still_flagged(module):
    """A batch-wide any() deciding the returned values is not an input guard."""
    assert _calls(module, "conditional_any_apply") == [
        ("reads a batch statistic", '(df["x"] > 1e6).any()')
    ]
