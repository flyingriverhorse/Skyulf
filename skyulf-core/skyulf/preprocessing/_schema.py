"""Re-export shim: schema primitives now live in :mod:`skyulf.core.schema`.

Kept for backward-compatible relative imports within the preprocessing package
(``from ._schema import ...``). New code should import from ``skyulf.core``.
"""

from skyulf.core.schema import *  # noqa: F401,F403
from skyulf.core.schema import (  # noqa: F401
    SchemaMismatchError,
    SkyulfSchema,
    validate_schema,
)