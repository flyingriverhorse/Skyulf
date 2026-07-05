"""Load small parametrized test-case fixtures from JSON files.

This mirrors the "data-driven test" pattern: test *data* (scenario name,
inputs, expected outputs) lives in a JSON file under ``tests/test_cases/``,
while the test *logic* stays in the ``.py`` file. This keeps large tables of
edge cases readable and lets new scenarios be added without touching Python.

**One fixture file per module.** A test module (e.g. ``test_casting.py``)
should have exactly one JSON fixture (e.g. ``test_cases/preprocessing/casting.json``),
even if it covers several independent scenario groups with different
parameter shapes. Use the ``"groups"`` shape for that:

```json
{
  "groups": {
    "coerce_boolean_value": {
      "params": "raw, expected",
      "test_cases": {
        "scenario_one": {"raw": "yes", "expected": true},
        "scenario_two": {"raw": "0", "expected": false}
      }
    },
    "coerce_float_value": {
      "params": "raw, expected",
      "test_cases": {
        "scenario_one": {"raw": "1.5", "expected": 1.5}
      }
    }
  }
}
```

```python
from tests.utils.test_case_loader import TestCaseLoader

cases = TestCaseLoader("preprocessing/casting", group="coerce_boolean_value").load()

@pytest.mark.parametrize(*cases)
def test_coerce_boolean_value(raw, expected):
    assert _coerce_boolean_value(raw) == expected
```

For the rare module with only ONE scenario group, the flat (ungrouped) shape
is still supported for brevity:

```json
{
  "params": "input, expected",
  "test_cases": {
    "scenario_one": {"input": 1, "expected": 2},
    "scenario_two": {"input": 2, "expected": 4}
  }
}
```

```python
cases = TestCaseLoader("preprocessing/my_calculator_edge_cases").load()

@pytest.mark.parametrize(*cases)
def test_my_calculator(input, expected):
    assert my_function(input) == expected
```
"""

import json
from pathlib import Path
from typing import Any

TEST_CASES_DIR = Path(__file__).resolve().parents[1] / "test_cases"


class TestCaseLoader:
    """Loads parametrized test cases from a JSON fixture file.

    Attributes:
        source: The parsed JSON document — either ``{"params": ..., "test_cases": ...}``
            (flat, single-group shape) or ``{"groups": {name: {"params": ..., "test_cases": ...}}}``
            (multi-group shape, preferred when a module has several scenario groups).
    """

    __test__ = False  # pytest: this is a helper class, not a test class.

    def __init__(self, source_path: str, group: str | None = None) -> None:
        """Read and parse the JSON fixture for ``source_path``.

        Args:
            source_path: Path (without extension) relative to
                ``tests/test_cases/``, e.g. ``"preprocessing/casting"``.
            group: Name of the scenario group to load, for fixtures using the
                multi-group ``{"groups": {...}}`` shape. Required when the
                fixture file has a top-level ``"groups"`` key; must be omitted
                (or left as ``None``) for flat single-group fixtures.

        Raises:
            FileNotFoundError: If no matching ``.json`` file exists.
            json.JSONDecodeError: If the file is not valid JSON.
            KeyError: If ``group`` names a group that doesn't exist in the
                fixture, or if ``group`` is required (fixture has a
                ``"groups"`` key) but wasn't provided.
        """
        fixture_path = TEST_CASES_DIR / f"{source_path}.json"
        with fixture_path.open(encoding="utf-8") as f:
            document: dict[str, Any] = json.load(f)

        if "groups" in document:
            if group is None:
                raise KeyError(
                    f"{source_path}.json uses the multi-group shape — "
                    "TestCaseLoader(..., group=<name>) must specify which group to load."
                )
            try:
                self.source: dict[str, Any] = document["groups"][group]
            except KeyError as exc:
                available = ", ".join(sorted(document["groups"]))
                raise KeyError(
                    f"Group {group!r} not found in {source_path}.json. Available groups: {available}"
                ) from exc
        else:
            if group is not None:
                raise KeyError(
                    f"{source_path}.json is a flat (single-group) fixture — "
                    "the 'group' argument must not be passed."
                )
            self.source = document

    def load(self) -> list[Any]:
        """Build a ``pytest.mark.parametrize(*result)``-ready pair.

        Returns:
            A two-element list: the comma-separated parameter names string,
            and a list of value tuples (one tuple per scenario, in file order).
        """
        test_params = self.source["params"]
        test_cases = self.source["test_cases"]

        params = [p.strip() for p in test_params.split(",")]
        scenarios = [
            tuple(test_case.get(param) for param in params) for test_case in test_cases.values()
        ]

        return [test_params, scenarios]

    def load_with_ids(self) -> tuple[str, list[Any], list[str]]:
        """Same as :meth:`load`, but also returns scenario names as pytest IDs.

        Returns:
            A ``(params_string, scenarios, ids)`` tuple suitable for
            ``@pytest.mark.parametrize(params_string, scenarios, ids=ids)`` —
            failures then print the scenario name instead of a raw tuple.
        """
        params_string, scenarios = self.load()
        ids = list(self.source["test_cases"].keys())
        return params_string, scenarios, ids
