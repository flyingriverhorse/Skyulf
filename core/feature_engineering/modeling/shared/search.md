# Search Helpers Reference

This note summarizes how `shared/search.py` orchestrates hyperparameter search configuration for both training and hyperparameter tuning flows.

## SearchConfiguration lifecycle
- `_build_search_configuration(job, node_config)` merges persisted job fields with node JSON to produce a `SearchConfiguration` dataclass. It always validates the search space (must be a JSON object mapping parameter names to candidate lists) and normalizes non-JSON-friendly values (e.g., numeric strings, `"None"`).
- Strategy resolution happens through `resolve_strategy_selection`; the helper keeps the original user input (`selected_strategy`) and the concrete implementation (`strategy`).
- Random-state, scoring, and iteration counts fall back to node defaults if the job omits them. Halving strategies automatically drop the explicit iteration count because the estimator controls candidate pruning.

## Cross-validation behavior
- Hyperparameter tuning always performs CV. `_build_search_configuration` feeds `job.cross_validation` (or the node config) through `_coerce_cross_validation_config` and, if the user disabled CV, forces an enabled configuration with at least three folds. This guarantees searchers have comparable fold metrics.
- `_resolve_cv_config` adds one more guard: halving strategies with `shuffle=True` but no random state borrow `SearchConfiguration.random_state` (falling back to `42`).
- Training tasks can still use `_parse_cross_validation_config` from `shared/common.py` to respect the node-level toggle. Expectation: quick “train-only” runs may disable CV, while advanced users run the tuning flow for fold-based comparisons.

## Search space sanitation
- `_sanitize_parameters` and `_coerce_none_strings` keep baseline hyperparameters consistent regardless of whether values arrive from JSON blobs or Celery kwargs.
- `_sanitize_logistic_regression_hyperparameters` enforces solver/penalty combos supported in production (only `lbfgs` and `saga` with `l2` or `none`, removing `l1_ratio`). Apply it whenever a logistic-regression spec is active to avoid invalid CV failures.

## Usage tips
- Always call `_filter_supported_parameters` with the union of estimator defaults and metadata-field names. This prevents typos from sneaking into Grid/RandomSearchCV.
- Prefer `_coerce_search_space` over `json.loads` so that tuples, scalars, or mixed casing (`"None"`, `"null"`) resolve correctly.
- When adding a new search strategy, derive it from `SearchConfiguration` rather than re-reading user inputs. This ensures new behavior inherits CV coercion, scoring defaults, and random-state normalization.
