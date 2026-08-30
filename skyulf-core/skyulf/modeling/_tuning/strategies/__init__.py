"""Searcher-based tuning strategies (F-18 split of ``engine.py``).

``halving`` builds the sklearn halving searchers, ``optuna`` owns the lazy
Optuna loader + OptunaSearchCV builder, and ``runner`` executes fitted
searchers and extracts their results. The grid/random custom loop lives one
level up in ``grid_random.py``.
"""
