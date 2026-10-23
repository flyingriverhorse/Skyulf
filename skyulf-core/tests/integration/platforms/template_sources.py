"""Read Databricks template sources as a default (include_examples=true) project holds them.

Some generated files are templates only so that ``include_examples=false`` can drop
example sections. Tests that copy template sources use these helpers, so they keep
reading the default output without running ``bundle init``.
"""

import shutil
from pathlib import Path

TEMPLATE = Path(__file__).resolve().parents[3] / "templates/databricks/template/{{.project_name}}"
# "{{end -}}" also trims the newline after it, which only the file's final marker uses.
_MARKERS = ('{{if eq .include_examples "true"}}', "{{end -}}\n", "{{end}}")


def template_text(path: str | Path) -> str:
    """Return a plain source, or its ``.tmpl`` sibling rendered with examples kept."""
    path = Path(path)
    if path.exists():
        return path.read_text(encoding="utf-8")
    text = path.with_name(path.name + ".tmpl").read_text(encoding="utf-8")
    for marker in _MARKERS:
        text = text.replace(marker, "")
    if "{{" in text:
        raise ValueError(f"{path} needs a full template render, not the examples-only shortcut.")
    return text


def copy_template(source: str | Path, destination: str | Path) -> Path:
    """Copy a template file or directory, rendering its examples-only ``.tmpl`` files."""
    source, destination = Path(source), Path(destination)
    if not source.is_dir():
        destination.write_text(template_text(source), encoding="utf-8")
        return destination
    shutil.copytree(source, destination, ignore=shutil.ignore_patterns("*.tmpl", "__pycache__"))
    for path in source.rglob("*.tmpl"):
        target = destination / path.relative_to(source).with_suffix("")
        target.write_text(template_text(path.with_suffix("")), encoding="utf-8")
    return destination
