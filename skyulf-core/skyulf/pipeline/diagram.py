"""Mermaid diagram rendering for pipeline topology.

Extracted from ``pipeline.py`` (F-19): diagram generation is a read-only
presentation concern and has no business living next to the fit path.
"""

import re
from collections.abc import Mapping, Sequence
from typing import Any

_MAX_DETAIL_KEYS = 3
_MAX_VALUE_LEN = 24
_MAX_DETAIL_LEN = 72


def mermaid_escape(text: str) -> str:
    """Escape characters that would break a Mermaid node label.

    Labels are always wrapped in double quotes by ``build_mermaid_diagram``,
    which neutralises shape syntax like ``(``, ``[``, ``#`` and ``|`` — the
    only character that still needs handling inside a quoted label is the
    quote itself.
    """
    return text.replace('"', "'")


def humanize_algorithm(raw: str) -> str:
    """Turn a machine-facing algorithm name into a readable title.

    ``random_forest_classifier`` -> ``Random Forest Classifier``,
    ``RandomForestClassifier`` -> ``Random Forest Classifier``.
    """
    text = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", str(raw))
    words = text.replace("_", " ").replace("-", " ").split()
    return " ".join(word.capitalize() for word in words) or str(raw)


def _format_value(value: Any) -> str:
    if isinstance(value, (list, tuple, set)):
        items = list(value)
        text = ", ".join(str(item) for item in items[:6])
        if len(items) > 6:
            text += ", …"
    else:
        text = str(value)
    if len(text) > _MAX_VALUE_LEN:
        text = text[: _MAX_VALUE_LEN - 1] + "…"
    return text


def params_summary(params: Mapping[str, Any] | None) -> str | None:
    """Compact one-line digest of the most informative config entries.

    Skips internal ``_``-prefixed keys, nested containers and ``None``
    values; keeps at most three entries so labels stay readable.
    """
    if not params:
        return None
    parts: list[str] = []
    for key, value in params.items():
        if str(key).startswith("_") or value is None or isinstance(value, Mapping):
            continue
        parts.append(f"{key}: {_format_value(value)}")
        if len(parts) == _MAX_DETAIL_KEYS:
            break
    if not parts:
        return None
    text = " · ".join(parts)
    if len(text) > _MAX_DETAIL_LEN:
        text = text[: _MAX_DETAIL_LEN - 1] + "…"
    return text


def mermaid_markdown(diagram: str) -> str:
    """Wrap a diagram in a ``mermaid`` code fence for Markdown documents.

    Renders natively on GitHub, in VS Code previews, and in Jupyter
    markdown cells.
    """
    return f"```mermaid\n{diagram}\n```\n"


def _node_label(head: str, detail: str | None) -> str:
    label = mermaid_escape(head)
    if detail:
        label += f"<br/>{mermaid_escape(detail)}"
    return label


def build_mermaid_diagram(
    preprocessing_steps: Sequence[Mapping[str, Any]],
    modeling_config: Mapping[str, Any],
) -> str:
    """Render a pipeline topology as a Mermaid ``flowchart`` string.

    Produces a top-down graph ``data -> [preprocessing steps] -> model``.
    Useful in docs and PR descriptions. Pure function over the config parts.

    Labels are human-readable (no internal node ids) and may carry a second
    line of detail — either an explicit ``details`` entry (e.g. a runtime
    summary) or a compact digest of the step ``params``.

    All labels are double-quoted: mermaid rejects unquoted parentheses and
    brackets inside ``[...]`` labels (e.g. a node named
    ``3-a415-… (DropMissingColumns)``), so quoting is mandatory, not an
    optimization.
    """
    lines = ["flowchart TD", '    data["Input Data"]']
    prev = "data"

    for i, step in enumerate(preprocessing_steps):
        node = f"pp{i}"
        name = str(step.get("name") or f"step_{i}")
        transformer = str(step.get("transformer") or "")
        head = f"{name} ({transformer})" if transformer and transformer != name else name
        detail = step.get("details") or params_summary(step.get("params") or {})
        lines.append(f'    {node}["{_node_label(head, detail)}"]')
        lines.append(f"    {prev} --> {node}")
        prev = node

    if modeling_config:
        raw_type = str(modeling_config.get("type", "model"))
        detail = modeling_config.get("details") or params_summary(
            {key: value for key, value in modeling_config.items() if key not in ("type", "node_id")}
        )
        lines.append(f'    model(["{_node_label(humanize_algorithm(raw_type), detail)}"])')
        lines.append(f"    {prev} --> model")

    return "\n".join(lines)
