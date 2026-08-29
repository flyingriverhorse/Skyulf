"""Re-export shim: schema primitives now live in :mod:`skyulf.core.schema`.

Kept for backward-compatible relative imports within the preprocessing package
(``from ._schema import ...``). New code should import from ``skyulf.core``.
The explicit names + ``__all__`` (instead of ``import *``) mark this module as
a deliberate re-export so unused-import checkers don't flag it.
"""

from skyulf.core.schema import SchemaMismatchError, SkyulfSchema, validate_schema

__all__ = ["SchemaMismatchError", "SkyulfSchema", "validate_schema"]
