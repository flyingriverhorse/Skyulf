"""Refresh the model guide's marked tables from Core and the Bundle schema.

Run from any directory; --check reports drift without writing. Only the marked
regions are generated, so explanations and runnable examples remain Markdown.
The registry loader is shared with build_model_spaces.py and needs only stdlib.
"""

import argparse
import ast
import importlib.util
import json
import re
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

ROOT = Path(__file__).resolve().parent
DOCUMENT = ROOT.parents[2] / "docs/user_guide/databricks_model_catalog.md"


def load_registry() -> ModuleType:
    """Reuse the initializer's actual registry loader without importing ML packages."""
    spec = importlib.util.spec_from_file_location("model_spaces", ROOT / "build_model_spaces.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load build_model_spaces.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module._load_registry()


def wizard_models() -> dict[str, str]:
    """Map every ordinary competition selector choice to its task."""
    properties = json.loads((ROOT / "databricks_template_schema.json").read_text("utf-8"))[
        "properties"
    ]
    return {
        model: task
        for task in ("regression", "classification")
        for model in properties[f"competition_{task}_model_1"]["enum"]
    }


def _registered_key(node: ast.ClassDef) -> str | None:
    """Read the literal key on a calculator's existing registration decorator."""
    for decorator in node.decorator_list:
        if (
            isinstance(decorator, ast.Call)
            and isinstance(decorator.func, ast.Attribute)
            and decorator.func.attr == "register"
            and decorator.args
            and isinstance(decorator.args[0], ast.Constant)
        ):
            return decorator.args[0].value
    return None


def constructor_defaults() -> dict[str, dict[str, str]]:
    """Read explicit constructor overrides without importing optional estimators.

    These are distinct from UI catalog defaults. Expressions are displayed as
    source, never evaluated; unspecified library defaults are not invented.
    """
    result = {}
    for path in sorted((ROOT.parents[1] / "skyulf/modeling").glob("*.py")):
        for node in ast.walk(ast.parse(path.read_text("utf-8"))):
            if not isinstance(node, ast.ClassDef) or not (key := _registered_key(node)):
                continue
            for field in ast.walk(node):
                if (
                    isinstance(field, ast.keyword)
                    and field.arg == "default_params"
                    and isinstance(field.value, ast.Dict)
                ):
                    result[key] = {
                        str(ast.literal_eval(name)): ast.unparse(value)
                        for name, value in zip(field.value.keys, field.value.values, strict=True)
                        if name is not None
                    }
    return result


def _code(value: Any) -> str:
    """Render JSON-compatible values as compact Markdown code."""
    return "`" + json.dumps(value, ensure_ascii=False).replace("|", "\\|") + "`"


def search_space(registry: ModuleType, key: str, task: str, strategy: str) -> dict[str, Any]:
    """Match the initializer's fixed-structure, default-member ensemble expansion."""
    if not key.startswith(("voting_", "stacking_")):
        return registry.get_default_search_space(key, strategy)
    defaults = {field["name"]: field["default"] for field in registry.get_hyperparameters(key)}
    space = registry.build_ensemble_search_space(
        key,
        defaults["base_estimators"],
        defaults.get("final_estimator", ""),
        strategy=strategy,
        problem_type=task,
    )
    # Bundle initialization fixes voting/cv/composition in params, tuning only
    # the selected learners. Core's general helper also includes meta-axes.
    return {name: choices for name, choices in space.items() if "__" in name}


def render_tables(
    registry: ModuleType, key: str, task: str, constructors: dict[str, dict[str, str]]
) -> str:
    """Render registered controls, explicit runtime defaults and both search families."""
    defaults = constructors[key]
    fields = registry.get_hyperparameters(key)
    rows = [
        "### Parameters and defaults",
        "",
        "| Parameter | Core catalog default | Explicit Core constructor default | Meaning |",
        "| --- | --- | --- | --- |",
    ]
    for field in fields:
        explicit = (
            f"`{defaults[field['name']]}`" if field["name"] in defaults else "Library/resolver"
        )
        description = field["description"].replace("|", "\\|").replace("\n", " ")
        rows.append(
            f"| `{field['name']}` | {_code(field['default'])} | {explicit} | {description} |"
        )
    rows.extend(
        f"| `{name}` | Not a catalog control | `{defaults[name]}` | Core constructor setting. |"
        for name in sorted(set(defaults) - {field["name"] for field in fields})
    )
    rows.extend(["", "### Default search spaces", ""])
    if key.startswith(("voting_", "stacking_")):
        rows.extend(
            [
                "These axes use the catalog's default members shown above. The wizard",
                "expands your selected members instead; it keeps composition, voting and",
                "internal CV fixed. Calibrated members add `estimator__` after their name.",
                "",
            ]
        )
    grid = search_space(registry, key, task, "grid")
    random = search_space(registry, key, task, "random")
    rows.extend(
        ["| Axis | Grid / halving_grid | Random / halving_random / Optuna |", "| --- | --- | --- |"]
    )
    rows.extend(
        f"| `{name}` | {_code(grid.get(name, []))} | {_code(random.get(name, []))} |"
        for name in sorted(grid.keys() | random.keys())
    )
    resource = json.loads((ROOT / "schema/tuning.json").read_text("utf-8"))["search_resource"]
    rows.extend(["", f"**Halving resource:** `{resource['default']}` (the wizard default)."])
    if "n_estimators" in defaults:
        rows.append(
            "This model also exposes `n_estimators`; selecting it requires explicit "
            "`max_resources` and removing that axis from the search space."
        )
    else:
        rows.append(
            "This model has no top-level `n_estimators` resource. Use `n_samples` "
            "in the wizard; another estimator parameter needs an explicit compatible configuration."
        )
    return "\n".join(rows) + "\n"


def render_document(source: str) -> str:
    """Replace exactly one marked table region for each selectable model."""
    registry = load_registry()
    constructors = constructor_defaults()
    for key, task in wizard_models().items():
        start, end = f"<!-- model-tables:{key}:start -->", f"<!-- model-tables:{key}:end -->"
        pattern = re.escape(start) + r"\n.*?" + re.escape(end)
        replacement = start + "\n" + render_tables(registry, key, task, constructors) + end
        source, count = re.subn(pattern, lambda _, text=replacement: text, source, flags=re.DOTALL)
        if count != 1:
            raise ValueError(f"Expected one table region for {key}; found {count}.")
    return source


def main() -> int:
    """Refresh generated regions or fail a read-only consistency check."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Fail when generated tables are stale")
    options = parser.parse_args()
    source = DOCUMENT.read_text(encoding="utf-8")
    rendered = render_document(source)
    if options.check:
        if source != rendered:
            print("Model catalog is stale; run build_model_catalog.py.", file=sys.stderr)
            return 1
        print("Model catalog is up to date.")
        return 0
    DOCUMENT.write_text(rendered, encoding="utf-8", newline="\n")
    print(f"Updated {DOCUMENT.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
