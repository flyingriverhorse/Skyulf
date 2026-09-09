"""Regression tests for unambiguous, deterministic artifact fingerprints."""

import dataclasses
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest

from skyulf.pipeline import SkyulfPipeline
from skyulf.pipeline.seal import artifact_digest


@dataclasses.dataclass
class ArtifactState:
    """Hold fitted state that can contain a nested reference."""

    value: Any


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
