"""Regression tests for unambiguous, deterministic artifact fingerprints."""

import dataclasses
import json
import os
import subprocess
import sys
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest

from skyulf.pipeline import SkyulfPipeline
from skyulf.pipeline.seal import artifact_digest
from skyulf.preprocessing.inspection import DataSnapshotApplier, DataSnapshotCalculator


@pytest.mark.parametrize(
    "value",
    [
        date(2024, 1, 1),
        datetime(2024, 1, 1, tzinfo=UTC),
        time(12, 30),
        timedelta(days=1),
        pd.Timestamp("2024-01-01"),
        pd.Timestamp("2024-02-01", tz="UTC"),
        pd.Timedelta("1 day 1 ns"),
        pd.NaT,
        pd.Period("2024-01", freq="M"),
        pd.Period("2024-02", freq="M"),
        np.timedelta64(1, "ns"),
        np.timedelta64(1, "ps"),
    ],
)
def test_temporal_scalars_cannot_hide_values_in_object_state(value):
    """Temporal values need a real scalar codec, never an empty instance dictionary digest."""
    with pytest.raises(TypeError):
        artifact_digest(value)


@pytest.mark.parametrize(
    "values",
    [
        [pd.Timestamp("2024-01-01"), pd.Timestamp("2024-02-01")],
        [pd.Period("2024-01", freq="M"), pd.Period("2024-02", freq="M")],
    ],
)
def test_real_snapshot_temporal_artifacts_fail_closed_instead_of_colliding(values):
    """Different saved temporal cells must not silently share the same semantic identity."""
    first = pd.DataFrame({"when": [values[0]]})
    second = pd.DataFrame({"when": [values[1]]})
    first_state: Any = DataSnapshotCalculator().fit(first, {})
    second_state: Any = DataSnapshotCalculator().fit(second, {})
    assert first_state != second_state
    assert DataSnapshotApplier().apply(first, first_state) is first
    assert DataSnapshotApplier().apply(second, second_state) is second
    for state in (first_state, second_state):
        with pytest.raises(TypeError):
            artifact_digest(state)


@pytest.mark.parametrize(
    "dtype,expected",
    [
        ("datetime64[ns]", "ae0e7af25366386ecbdf9440ba1abb18abd3c151db90a0dcd4f686301deea9b1"),
        ("timedelta64[ns]", "5e84b50b7e8a519a709530917da0982c76af4d0d2897db700a63670fc06e54d4"),
    ],
)
def test_temporal_arrays_retain_existing_canonical_bytes(dtype, expected):
    """Rejecting scalar fallback must not change supported temporal-array digests."""
    values = np.array([0, 1, -9223372036854775808], dtype=dtype)
    assert artifact_digest(values).hex() == expected


def test_pipeline_fingerprint_rejects_real_temporal_snapshot_state():
    """A fitted pipeline must expose unsupported snapshot values through its public seal."""
    pipeline = SkyulfPipeline(
        {"preprocessing": [{"name": "snapshot", "transformer": "DataSnapshot", "params": {}}]}
    )
    frame = pd.DataFrame({"when": pd.to_datetime(["2024-01-01", "2024-02-01"]), "target": [0, 1]})
    pipeline.fit(frame, "target")
    with pytest.raises(TypeError, match="temporal scalar"):
        pipeline.fingerprint()


def test_tree_missing_value_routing_changes_digest():
    """Distinct NaN predictions must never share a fitted-tree artifact identity."""
    from sklearn.tree import DecisionTreeClassifier

    model = DecisionTreeClassifier(max_depth=1).fit([[0.0], [1.0]], [0, 1])
    before_digest = artifact_digest(model)
    before_prediction = model.predict([[np.nan]])
    model.tree_.missing_go_to_left[0] = 1 - model.tree_.missing_go_to_left[0]

    assert model.predict([[np.nan]])[0] != before_prediction[0]
    assert artifact_digest(model) != before_digest


@dataclasses.dataclass
class ArtifactState:
    """Hold fitted state that can contain a nested reference."""

    value: Any


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (
            [
                None,
                True,
                np.bool_(False),
                0,
                np.int64(-9),
                -0.0,
                np.float64(np.inf),
                complex(2, -3),
                "λ:str:",
                b"\0bytes:",
                bytearray(b"x"),
                memoryview(b"y"),
                {"b": {1, 2}, "a": ("c",)},
                frozenset({4, 5}),
                int,
                SimpleNamespace(value=[1, 2]),
            ],
            "240510dfef5056634a40a4bb93b74ea65a4f3f2cc68b62742096d1ad5a75045a",
        ),
        (
            np.array([[1.5, -0.0], [np.nan, np.inf]], dtype="<f8"),
            "4050de9236b84c5f41764e36c8652674e34dd6fc8400fa0a6724697a157d4f52",
        ),
        (
            np.array([["a", None], [3, b"b"]], dtype=object),
            "230765659f8b31488b70433fb5121b761b84c97681f13b2eb509ac6315ad2469",
        ),
        (
            np.array([(1, 0.5), (2, 1.5)], dtype=[("count", "<i8"), ("value", "<f8")]),
            "1ca5bc2b341218b1947e421ce9c7dd167b546b85ba2c615a375ae9af02712214",
        ),
    ],
    ids=["scalars-and-containers", "numeric-array", "object-array", "structured-array"],
)
def test_artifact_digest_preserves_canonical_encoding(value: Any, expected: str) -> None:
    """Refactoring serialization must preserve previously stored semantic digests."""
    assert artifact_digest(value).hex() == expected


@pytest.mark.parametrize(
    ("left", "right"),
    [
        (
            np.array(["a", "bstr:c"], dtype=object),
            np.array(["astr:b", "c"], dtype=object),
        ),
        (["a", "b,str:c"], ["a,str:b", "c"]),
        (("a", "b,str:c"), ("a,str:b", "c")),
        ([b"a", b"b,bytes:c"], [b"a,bytes:b", b"c"]),
        ({"a": "b=str:c"}, {"a=str:b": "c"}),
        ({"a": ["a", "b,str:c"]}, {"a": ["a,str:b", "c"]}),
        (ArtifactState(["a", "b,str:c"]), ArtifactState(["a,str:b", "c"])),
    ],
    ids=["object-array", "list", "tuple", "bytes", "dict", "nested", "dataclass"],
)
def test_artifact_digest_preserves_value_boundaries(left: Any, right: Any) -> None:
    """Embedded type tags and separators must not conceal different fitted values."""
    assert artifact_digest(left) != artifact_digest(right)


def test_fitted_label_encoder_fingerprints_distinguish_different_transformations() -> None:
    """A seal must distinguish encoders that map the same input to 0 and -1."""
    config = {
        "preprocessing": [
            {"name": "encode", "transformer": "LabelEncoder", "params": {"columns": ["x"]}}
        ]
    }
    first = SkyulfPipeline(config)
    second = SkyulfPipeline(config)
    first.fit(pd.DataFrame({"x": ["a", "bstr:c"], "target": [0, 1]}), "target")
    second.fit(pd.DataFrame({"x": ["astr:b", "c"], "target": [0, 1]}), "target")
    probe = pd.DataFrame({"x": ["a"]})

    assert first.feature_engineer.transform(probe)["x"].tolist() == [0]
    assert second.feature_engineer.transform(probe)["x"].tolist() == [-1]
    assert first.fingerprint() != second.fingerprint()


@pytest.mark.parametrize("kind", ["list", "dict", "tuple", "object", "dataclass", "array"])
def test_artifact_digest_rejects_reference_cycles(kind: str) -> None:
    """Cyclic fitted state must raise the documented TypeError before stack exhaustion."""
    value: Any
    if kind == "list":
        value = []
        value.append(value)
    elif kind == "dict":
        value = {}
        value["self"] = value
    elif kind == "tuple":
        child: list[Any] = []
        value = (child,)
        child.append(value)
    elif kind == "object":
        value = SimpleNamespace()
        value.self = value
    elif kind == "dataclass":
        value = ArtifactState(None)
        value.value = value
    else:
        value = np.empty(1, dtype=object)
        value[0] = value

    with pytest.raises(TypeError, match="[Cc]ycl"):
        artifact_digest(value)


def test_pipeline_fingerprint_rejects_cyclic_fitted_state() -> None:
    """The public pipeline seal must preserve the documented unsupported-artifact error."""
    pipeline = SkyulfPipeline({"preprocessing": [], "modeling": {"type": "logistic_regression"}})
    pipeline.fit(pd.DataFrame({"x": [0, 1, 2, 3], "target": [0, 0, 1, 1]}), "target")
    assert pipeline.model_estimator is not None
    model = pipeline.model_estimator.model
    assert model is not None
    model.cyclic_state_ = {"model": model}

    with pytest.raises(TypeError, match="[Cc]ycl"):
        pipeline.fingerprint()


def test_artifact_digest_accepts_deep_acyclic_lists() -> None:
    """Cycle detection must retain the nesting capacity of the original single-frame walk."""
    first: Any = 0
    second: Any = 1
    for _ in range(600):
        first = [first]
        second = [second]

    assert artifact_digest(first) != artifact_digest(second)


@pytest.mark.parametrize(
    ("kind", "depth", "expected"),
    [
        ("dict", 220, "c36185a7740aa4e4a6d1ed00b6f4c4721560495e9966efbdb92ca91602740109"),
        ("mixed", 220, "b7ab6efea93a50b090b790368e3837414fa0a4635d69e25476b75de7c5dcfe49"),
        ("array", 350, "5bf1426d9538fe26653efc5241eb36549e6ce1b26de619f4b0ecc0724643d5b4"),
    ],
)
def test_artifact_digest_preserves_deep_container_capacity(
    kind: str, depth: int, expected: str
) -> None:
    """Encoder helpers must not consume recursive frames and reject existing deep artifacts."""
    value: Any = 0
    for _ in range(depth):
        if kind == "dict":
            value = {"child": value}
        elif kind == "mixed":
            value = [{"child": value}]
        else:
            parent = np.empty(1, dtype=object)
            parent[0] = value
            value = parent

    assert artifact_digest(value).hex() == expected


def test_artifact_digest_rejects_deep_reference_cycle() -> None:
    """A back reference below hundreds of containers must still report a cycle clearly."""
    root: list[Any] = []
    current = root
    for _ in range(600):
        child: list[Any] = []
        current.append(child)
        current = child
    current.append(root)

    with pytest.raises(TypeError, match="[Cc]ycl"):
        artifact_digest(root)


def test_artifact_digest_reports_excessive_nesting_as_type_error() -> None:
    """Stack exhaustion must use the public error contract without misreporting a cycle."""
    value: Any = 0
    for _ in range(sys.getrecursionlimit() + 10):
        value = [value]

    with pytest.raises(TypeError, match="nesting.*depth"):
        artifact_digest(value)


@pytest.mark.parametrize("kind", ["list", "dict", "tuple", "object", "dataclass", "array"])
def test_artifact_digest_accepts_shared_acyclic_references(kind: str) -> None:
    """Repeated references must hash like equal independent copies, not count as cycles."""
    constructors = {
        "list": lambda: ["a", "b"],
        "dict": lambda: {"a": "b"},
        "tuple": lambda: tuple(iter(("a", "b"))),
        "object": lambda: SimpleNamespace(value=["a", "b"]),
        "dataclass": lambda: ArtifactState(["a", "b"]),
        "array": lambda: np.array(["a", "b"], dtype=object),
    }
    shared = constructors[kind]()

    assert artifact_digest([shared, shared]) == artifact_digest(
        [constructors[kind](), constructors[kind]()]
    )


def test_artifact_digest_distinguishes_nested_types() -> None:
    """Empty values, container types, and nesting remain part of an artifact's identity."""
    values = [None, "", b"", [], (), {}, set(), [1], (1,), [[1]], [()], {"1": 1}]
    assert len({artifact_digest(value) for value in values}) == len(values)


def test_artifact_digest_distinguishes_dataclass_modules() -> None:
    """Equal class names from different modules must not identify different artifact types."""
    first = dataclasses.make_dataclass("State", [("value", int)], module="first")
    second = dataclasses.make_dataclass("State", [("value", int)], module="second")
    assert artifact_digest(first(1)) != artifact_digest(second(1))


@pytest.mark.parametrize("kind", ["values", "cycle", "nested-cycle"])
def test_artifact_digest_rejects_structured_object_arrays(kind: str) -> None:
    """Unsupported object fields cannot silently hash pointers or conceal cycles."""
    dtype = np.dtype(
        [("nested", [("value", object)])] if kind == "nested-cycle" else [("value", object)]
    )
    array = np.empty(1, dtype=dtype)
    values = array["nested"]["value"] if kind == "nested-cycle" else array["value"]
    values[0] = array if "cycle" in kind else "category"

    with pytest.raises(TypeError, match="object-containing"):
        artifact_digest(array)


def test_artifact_digest_supports_numeric_structured_arrays() -> None:
    """Numeric field arrays remain digestible and sensitive to learned value changes."""
    first = np.array([(1, 0.5), (2, 1.5)], dtype=[("count", "i8"), ("value", "f8")])
    second = first.copy()

    assert artifact_digest(first) == artifact_digest(second)
    second["value"][0] = 2.5
    assert artifact_digest(first) != artifact_digest(second)


@pytest.mark.parametrize("layout", ["direct", "nested", "subarray"])
def test_artifact_digest_ignores_structured_array_padding(layout: str) -> None:
    """Uninitialized record padding must not change a seal when all field values agree."""
    record_dtype = np.dtype([("flag", "i1"), ("weight", "f8")], align=True)
    if layout == "nested":
        dtype = np.dtype([("record", record_dtype), ("count", "i4")], align=True)
    elif layout == "subarray":
        dtype = np.dtype([("record", record_dtype, (2,))])
    else:
        dtype = record_dtype
    first = np.zeros(2, dtype=dtype)
    records = first if layout == "direct" else first["record"]
    records["flag"] = 1
    records["weight"] = 0.5
    second = first.copy()
    second.view(np.uint8).reshape(2, dtype.itemsize)[:, 1:8] = 255

    assert np.array_equal(first, second)
    assert first.tobytes() != second.tobytes()
    assert artifact_digest(first) == artifact_digest(second)


def test_artifact_digest_preserves_structured_array_dtype_and_shape() -> None:
    """Field traversal must retain the array's schema and dimensions in its identity."""
    first = np.array([(1, 0.5), (2, 1.5)], dtype=[("count", "i8"), ("value", "f8")])
    renamed = first.view(dtype=[("count", "i8"), ("score", "f8")])

    assert artifact_digest(first) != artifact_digest(renamed)
    assert artifact_digest(first) != artifact_digest(first.reshape(1, 2))


def test_artifact_digest_remains_stable_across_processes(tmp_path: Path) -> None:
    """Hash seeds and allocation addresses cannot affect nested unordered artifact state."""
    script = tmp_path / "fingerprint_probe.py"
    script.write_text(
        "import json\n"
        "import numpy as np\n"
        "import pandas as pd\n"
        "from skyulf.pipeline import SkyulfPipeline\n"
        "from skyulf.pipeline.seal import artifact_digest\n"
        "keys = {frozenset({'red', 'green'}), frozenset({'blue', 'yellow'})}\n"
        "state = {key: np.array(['a', 'bstr:c'], dtype=object) for key in keys}\n"
        "pipeline = SkyulfPipeline({'preprocessing': ["
        "{'name': 'encode', 'transformer': 'LabelEncoder', 'params': {'columns': ['x']}}]})\n"
        "pipeline.fit(pd.DataFrame({'x': ['a', 'bstr:c'], 'target': [0, 1]}), 'target')\n"
        "print(json.dumps([artifact_digest(state).hex(), artifact_digest(keys).hex(), "
        "pipeline.fingerprint()]))\n",
        encoding="utf-8",
    )
    outputs = []
    for seed in ("1", "2"):
        env = os.environ.copy()
        env["PYTHONHASHSEED"] = seed
        env["PYTHONPATH"] = str(Path(__file__).resolve().parents[2])
        result = subprocess.run(
            [sys.executable, str(script)],
            env=env,
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
        outputs.append(json.loads(result.stdout))

    assert outputs[0] == outputs[1]
