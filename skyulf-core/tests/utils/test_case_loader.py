"""Load small parametrized test-case fixtures from JSON files.

This mirrors the "data-driven test" pattern: test *data* (scenario name,
inputs, expected outputs) lives in a JSON file under ``tests/test_cases/``,
while the test *logic* stays in the ``.py`` file. This keeps large tables of
edge cases readable and lets new scenarios be added without touching Python.

Example JSON shape (``tests/test_cases/<group>/<name>.json``):

```json
{
  "params": "input, expected",
  "test_cases": {
    "scenario_one": {"input": 1, "expected": 2},
    "scenario_two": {"input": 2, "expected": 4}
  }
}
```

Example usage in a test file:

```python
from tests.utils.test_case_loader import TestCaseLoader

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
        source: The parsed JSON document (``{"params": ..., "test_cases": ...}``).
    """

    __test__ = False  # pytest: this is a helper class, not a test class.

    def __init__(self, source_path: str) -> None:
        """Read and parse the JSON fixture for ``source_path``.

        Args:
            source_path: Path (without extension) relative to
                ``tests/test_cases/``, e.g. ``"preprocessing/scaling_edge_cases"``.

        Raises:
            FileNotFoundError: If no matching ``.json`` file exists.
            json.JSONDecodeError: If the file is not valid JSON.
        """
        fixture_path = TEST_CASES_DIR / f"{source_path}.json"
        with fixture_path.open(encoding="utf-8") as f:
            self.source: dict[str, Any] = json.load(f)

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
