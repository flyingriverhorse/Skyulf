"""PREPROCESSING: prepare the columns before the model sees them.

Runs AFTER the train/test split. A step that learns something (an average,
a mapping) learns it from training rows only; scoring reuses what was learned.
Shared Spark table calculations belong in groups/. Do not fit this recipe on
the entire merged table or repeat a transformation already applied upstream.

HOW TO USE
  1. Put steps in the list of _default_recipe() below. They run top to bottom.
     You can mix two kinds of step in one list:
       - Built-in step: a dict naming a Skyulf node, e.g.
           {"name": "fill", "transformer": "SimpleImputer",
            "params": {"columns": ["income"], "strategy": "mean"}}
         See all of them: python src/tools/preview.py --list-preprocessors
       - Your own step: a function from custom/preprocessing_custom.py, e.g.
           frequency_encoding(columns=["city"])
         Uncomment its import line below first.
  2. Check: python src/tools/preview.py --action train

RECIPES
  A recipe is a named list of steps. config/training.yml selects it through
  preprocessing_recipe in defaults or a model entry; the default is "default".
  For example: preprocessing_recipe: example_all.
  Recipes starting with "example_" are ready-made lists to read or copy from;
  they do nothing unless a model selects them. "none" means no steps.
"""

from .custom import preprocessing_custom

# from .custom.preprocessing_custom import frequency_encoding, log_feature, rare_categories
# from .custom.preprocessing_custom import city_region  # needs an asset, see its Example 4


def build_preprocessing(recipe="default"):
    """Return the step list of one named recipe."""
    recipes = {
        "default": _default_recipe,
        "none": lambda: [],
        "example_frequency": _example_frequency,
        "example_imputer": _example_imputer,
        "example_imputer_frequency": lambda: _example_imputer() + _example_frequency(),
        "example_all": _example_all,
    }
    if recipe not in recipes:
        raise ValueError(f"Unknown preprocessing recipe: {recipe}. Choose from {list(recipes)}.")
    return recipes[recipe]()


def _default_recipe():
    """Your main recipe. Empty until you uncomment or add steps."""
    return [
        # Built-in steps:
        # {"name": "impute", "transformer": "SimpleImputer",
        #  "params": {"columns": ["feature_value"], "strategy": "mean"}},
        # {"name": "scale", "transformer": "StandardScaler",
        #  "params": {"columns": ["feature_value"]}},
        # Fill each row with its own category's median (learned on training rows):
        # {"name": "group_fill", "transformer": "GroupImputer",
        #  "params": {"columns": ["feature_value"], "group_by": "category",
        #             "strategy": "median"}},
        # Cap values at fixed limits; no rows are removed:
        # {"name": "cap", "transformer": "ClipValues",
        #  "params": {"bounds": {"feature_value": {"lower": 0, "upper": 1000}}}},
        # Your own steps (custom/preprocessing_custom.py):
        # log_feature("feature_value"),
        # rare_categories("category", min_share=0.05),
        # frequency_encoding(columns=["category"]),
        # city_region(),  # reads assets/city_region.json
        # Time-based history (after the split). observation_time must be an
        # input column and differ from the job's event_column:
        # {"name": "recent_value", "transformer": "RollingAggregate",
        #  "params": {"columns": ["feature_value"], "window": 5,
        #             "sort_by": "observation_time", "group_by": ["entity"],
        #             "history_mode": "carry", "history_max_rows": 1000,
        #             "history_max_bytes": 48000}},
        # {"name": "drop_clock", "transformer": "DropMissingColumns",
        #  "params": {"columns": ["observation_time"], "missing_threshold": None}},
    ]


def _example_frequency():
    """Only encode category by its training frequency."""
    return [preprocessing_custom.frequency_encoding(columns=["category"])]


def _example_imputer():
    """Only fill missing feature_value with its training average."""
    return [
        {
            "name": "impute",
            "transformer": "SimpleImputer",
            "params": {"columns": ["feature_value"], "strategy": "mean"},
        },
    ]


def _example_all():
    """Built-in and own steps together, in the order they run."""
    return [
        *_example_imputer(),  # built-in: fill missing feature_value
        preprocessing_custom.log_feature("feature_value"),  # own: new column, nothing learned
        preprocessing_custom.rare_categories("category"),  # own: rare values -> "Other"
        preprocessing_custom.frequency_encoding(columns=["category"]),  # own: category -> number
    ]
