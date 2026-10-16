"""Probe strategies shared by saved preprocessing steps and scoring rules.

Each strategy calls the same ``apply`` on a reshaped copy of the sample and
compares with the fixed floating output policy. ``apply`` must copy its input
and check its own mutations; it is the caller's real execution path.
"""

from collections.abc import Callable

from ._probe_frames import (
    Frame,
    ProbeFailure,
    assert_same,
    perturb_row,
    renumber_frame,
    reverse_frame,
    slice_frame,
    without_row,
)

# Rows whose inputs the perturb check changes one at a time.
PERTURB_ROWS = 32

Apply = Callable[[Frame], Frame]


def run_check(
    name: str,
    apply: Apply,
    frame: Frame,
    expected: Frame | None,
    *,
    exact_columns: tuple[str, ...] = (),
) -> tuple[dict, Frame | None]:
    """Return redacted evidence for one strategy, preserving mutation failures on empty input."""
    try:
        result = run_strategy(name, apply, frame, expected, exact_columns=exact_columns)
        return {"name": name, "status": "passed"}, result
    except ProbeFailure as exc:
        return {"name": name, "status": "failed", "reason": str(exc)}, None
    except Exception as exc:  # noqa: BLE001 - report arbitrary trusted callback failures
        detail = {
            "name": name,
            "status": "not_supported" if name == "empty" else "failed",
            "reason": "apply_error",
            "error_type": type(exc).__name__,
        }
        if name == "fresh_module":
            # Admission-only check: the message explains why fresh project code failed.
            detail["error_message"] = str(exc)[:300]
        return detail, None


def run_strategy(
    name: str,
    apply: Apply,
    frame: Frame,
    expected: Frame | None,
    *,
    exact_columns: tuple[str, ...] = (),
) -> Frame | None:
    """Use unchanged apply code for full, repeated, partitioned and reordered requests."""
    if name == "full":
        return apply(frame)
    if expected is None:
        raise ProbeFailure("missing_expected_output")
    if name.startswith("chunks:"):
        _check_chunks(apply, frame, expected, int(name.split(":")[1]), exact_columns)
    elif name == "reverse":
        assert_same(
            expected, reverse_frame(apply(reverse_frame(frame))), exact_columns=exact_columns
        )
    elif name == "index_shift":
        actual = apply(renumber_frame(frame))
        assert_same(
            expected,
            renumber_frame(actual, getattr(expected, "index", None)),
            exact_columns=exact_columns,
        )
    elif name == "empty":
        assert_same(
            slice_frame(expected, 0, 0),
            apply(slice_frame(frame, 0, 0)),
            exact_columns=exact_columns,
        )
    elif name == "perturb":
        _check_perturb(apply, frame, expected, exact_columns)
    elif name == "fresh_module":
        # The caller passes an apply bound to a freshly loaded copy of the project code.
        assert_same(expected, apply(frame), exact_columns=exact_columns)
    else:
        # Identical requests must remain exact; tolerance is for changed batch layouts.
        for _ in range(2):
            assert_same(expected, apply(frame), exact=True, exact_columns=exact_columns)
    return None


def _check_chunks(
    apply: Apply, frame: Frame, expected: Frame, size: int, exact_columns: tuple[str, ...]
) -> None:
    """Compare each chunk before concatenation could conceal dtype promotion."""
    for start in range(0, len(frame), size):
        assert_same(
            slice_frame(expected, start, size),
            apply(slice_frame(frame, start, size)),
            exact_columns=exact_columns,
        )


def _check_perturb(
    apply: Apply, frame: Frame, expected: Frame, exact_columns: tuple[str, ...]
) -> None:
    """Change one row's inputs at a time; every other row's output must stay the same."""
    for position in range(min(len(frame), PERTURB_ROWS)):
        actual = _perturbed(apply, frame, position)
        assert_same(
            without_row(expected, position),
            without_row(actual, position),
            exact_columns=exact_columns,
        )


def _perturbed(apply: Apply, frame: Frame, position: int) -> Frame:
    """Apply to a perturbed row, retrying with text left unchanged if parsing it fails.

    A row-local function may legitimately reject unknown text (a date or number
    stored as text). Only a failure on numbers, booleans and categories as well
    reports ``apply_error``: the step raised on a changed but valid-looking value.
    """
    try:
        return apply(perturb_row(frame, position))
    except ProbeFailure:
        raise
    except Exception:  # noqa: BLE001 - retry without changing text values
        pass
    try:
        return apply(perturb_row(frame, position, text=False))
    except ProbeFailure:
        raise
    except Exception:  # noqa: BLE001 - trusted callback failures become evidence
        raise ProbeFailure("apply_error") from None
