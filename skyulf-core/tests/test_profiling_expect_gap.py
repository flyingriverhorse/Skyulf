"""Gap-closing test for skyulf.profiling.expect._as_pandas unsupported-type branch."""

import pytest

from skyulf.profiling.expect import expect_no_nulls


def test_expect_no_nulls_raises_type_error_for_unsupported_frame():
    """Passing a plain list (no to_pandas, not a DataFrame) should raise TypeError."""
    with pytest.raises(TypeError, match="Unsupported frame type"):
        expect_no_nulls([1, 2, 3])
