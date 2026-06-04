"""Guard against re-introducing inline engine dispatch in node modules.

Node modules must not branch on ``engine.name == EngineName.POLARS`` directly.
Engine-specific behaviour belongs in the dual-engine dispatcher
(``preprocessing/dispatcher.py``) or the shared engine-bridge / column-detection
helpers (``_helpers.py``, ``encoding/_common.py``). Centralising the branch keeps
every node engine-agnostic and makes adding a new engine a single-file change.
"""

import re
from pathlib import Path

# Files where an inline POLARS branch is intentional (the dispatch/bridge layer).
_SANCTIONED = {
    "dispatcher.py",
    "_helpers.py",
    Path("encoding") / "_common.py",
}

# Matches an actual ``if <x>.name == EngineName.POLARS`` statement, not docstrings.
_INLINE_DISPATCH = re.compile(r"^\s*(?:el)?if\s+\w+\.name\s*==\s*EngineName\.POLARS")

_PREPROCESSING_DIR = Path(__file__).resolve().parent.parent / "skyulf" / "preprocessing"


def _is_sanctioned(path: Path) -> bool:
    rel = path.relative_to(_PREPROCESSING_DIR)
    return rel.name in _SANCTIONED or rel in _SANCTIONED


def test_no_inline_engine_dispatch_in_node_modules() -> None:
    offenders = []
    for py_file in _PREPROCESSING_DIR.rglob("*.py"):
        if _is_sanctioned(py_file):
            continue
        for lineno, line in enumerate(py_file.read_text(encoding="utf-8").splitlines(), 1):
            if _INLINE_DISPATCH.match(line):
                offenders.append(f"{py_file.relative_to(_PREPROCESSING_DIR)}:{lineno}")

    assert not offenders, (
        "Inline engine dispatch found in node modules; route through the "
        "dual-engine dispatcher or _helpers.is_polars()/to_pandas() instead:\n  "
        + "\n  ".join(offenders)
    )
