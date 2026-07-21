"""Compatibility shims for sklearn API deprecations.

sklearn >=1.8 deprecates ``LogisticRegression(penalty=...)`` in favor of
``l1_ratio``/``C`` (0 = pure L2, 1 = pure L1, in-between = elasticnet mixing;
``C=inf`` = no penalty at all) and will remove the ``penalty`` constructor
argument entirely in sklearn 1.10.

We keep ``penalty`` (``"l1"``/``"l2"``/``"elasticnet"``/``None``) as *our*
public config/UI field everywhere (hyperparameter registry, tuning search
spaces, node params) since it's the familiar, well-documented concept users
expect to configure — but translate it to the newer kwargs right before
construction, so we never pass a bare ``penalty=`` to sklearn and never
trigger the deprecation warning, on any sklearn version.
"""

import math
from typing import Any


def normalize_logistic_regression_params(params: dict[str, Any]) -> dict[str, Any]:
    """Translates a ``penalty`` key into sklearn's newer ``l1_ratio``/``C`` kwargs.

    No-op if ``penalty`` isn't present. Returns a new dict — never mutates
    *params* in place. An explicit ``l1_ratio``/``C`` the caller already set
    always wins over the value this function would otherwise inject.

    Mapping (matches sklearn's own deprecation-warning guidance):
      - ``penalty="l2"``        -> ``l1_ratio=0.0``
      - ``penalty="l1"``        -> ``l1_ratio=1.0``
      - ``penalty="elasticnet"``-> ``l1_ratio=0.5`` if not already set (a real
        elasticnet mix; sklearn's own new default of ``l1_ratio=0.0`` would
        otherwise silently degrade an "elasticnet" choice into pure L2)
      - ``penalty=None``        -> ``C=math.inf`` (no penalty at all)
    """
    if "penalty" not in params:
        return params
    params = dict(params)
    penalty = params.pop("penalty")
    if penalty == "l2":
        params.setdefault("l1_ratio", 0.0)
    elif penalty == "l1":
        params.setdefault("l1_ratio", 1.0)
    elif penalty == "elasticnet":
        params.setdefault("l1_ratio", 0.5)
    elif penalty is None:
        params["C"] = math.inf
    return params
