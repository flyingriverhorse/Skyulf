"""Validate and inspect exact dependency pins saved with trusted project packages."""

import ast
import re
from importlib.metadata import PackageNotFoundError, version

_PIN = re.compile(r"([A-Za-z0-9][A-Za-z0-9._-]*)==([0-9][A-Za-z0-9.!+_-]*)")


def parse_project_requirements(text: str) -> tuple[str, ...]:
    """Accept distinct exact distribution pins, blank lines, and full-line comments.

    Extras, environment markers, ranges, URLs, pip options and recursive files
    are intentionally unsupported. No dependency installation occurs here.
    """
    pins: dict[str, str] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        match = _PIN.fullmatch(line)
        if match is None:
            raise ValueError("Project requirement must be an exact distribution==version pin.")
        name = re.sub(r"[-_.]+", "-", match[1]).lower()
        if name in pins:
            raise ValueError(f"Duplicate project dependency: {name}.")
        pins[name] = match[2]
    return tuple(f"{name}=={pins[name]}" for name in sorted(pins))


def verify_project_requirements(requirements: tuple[str, ...]) -> None:
    """Fail before executing project code if a pinned distribution is unavailable."""
    for pin in parse_project_requirements("\n".join(requirements)):
        name, expected = pin.split("==")
        try:
            actual = version(name)
        except PackageNotFoundError as exc:
            raise ValueError(f"Project dependency {pin} is not installed.") from exc
        if actual != expected:
            raise ValueError(f"Project dependency {name} requires {expected}, found {actual}.")


def source_project_requirements(source: str) -> tuple[str, ...]:
    """Read generated snapshot pins without importing or executing trusted user code.

    Legacy single-file source and old package snapshots have no declared pins.
    Generated packages use a literal ``requirements`` keyword on their installer.
    """
    statements = ast.parse(source).body
    common = _common_source(statements)
    if common is not None:
        statements = ast.parse(common).body
        if _common_source(statements) is not None:
            raise ValueError("Saved common project source cannot contain another shared snapshot.")
    for statement in statements:
        if isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Call):
            call = statement.value
            if isinstance(call.func, ast.Name) and call.func.id == "install_project_package":
                return _call_requirements(call)
    return ()


def _common_source(statements: list[ast.stmt]) -> str | None:
    """Read only the bounded literal common snapshot emitted by competition composition."""
    from .project_code import project_source_digest  # noqa: PLC0415 - source/package import cycle

    captured = [
        ast.literal_eval(statement.value)
        for statement in statements
        if isinstance(statement, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "__skyulf_common_source__"
            for target in statement.targets
        )
    ]
    if not captured:
        return None
    if len(captured) != 1:
        raise ValueError("Saved common project source must have one literal snapshot.")
    project_source_digest(captured[0])
    return captured[0]


def _call_requirements(call: ast.Call) -> tuple[str, ...]:
    """Validate the literal dependency metadata emitted by the snapshot writer."""
    for keyword in call.keywords:
        if keyword.arg == "requirements":
            requirements = ast.literal_eval(keyword.value)
            if not isinstance(requirements, tuple) or any(
                not isinstance(pin, str) for pin in requirements
            ):
                raise ValueError("Saved project requirements must be a tuple of exact pins.")
            return parse_project_requirements("\n".join(requirements))
    return ()
