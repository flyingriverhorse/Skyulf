"""Re-export shim: artifact TypedDicts now live in :mod:`skyulf.core.artifacts`.

Kept for backward-compatible relative imports within the preprocessing package
(``from ._artifacts import ...``). New code should import from ``skyulf.core``.
"""

from skyulf.core.artifacts import *  # noqa: F401,F403
