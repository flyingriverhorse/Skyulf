"""Re-export shim: schema primitives now live in :mod:`skyulf.core.schema`.

Kept for backward-compatible relative imports within the preprocessing package
(``from ._schema import ...``). New code should import from ``skyulf.core``.
"""

from skyulf.core.schema import *
from skyulf.core.schema import (
    SchemaMismatchError,
    SkyulfSchema,
    validate_schema,
)
