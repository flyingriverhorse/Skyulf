"""PRE-SPLIT: choose which rows may be used for training.

Runs BEFORE the train/test split. Filters only remove rows: they never
change values and never learn from data (that belongs in preprocessing.py).

HOW TO USE
  1. Put filters in the list of _default_recipe() below. They run top to bottom.
     You can mix two kinds of filter in one list:
       - Built-in step: a dict naming a Skyulf node, e.g.
           {"name": "known_target", "transformer": "DropMissingRows",
            "params": {"subset": ["target"], "how": "any"}}
       - Your own filter: a function from custom/pre_split_custom.py, e.g.
           value_range("age", minimum=0, maximum=120)
         Uncomment its import line below first.
  2. Check: python src/tools/preview.py --action train

RECIPES
  A recipe is a named list of filters. Single-model training always uses
  "default". multi_model.py chooses one per model with "pre_split_recipe".
  Recipes starting with "example_" are ready-made lists to read or copy from.
  "none" means no filters.

scoring.py can reuse these filters for scoring rows. A filter that reads the
target column needs SKIP_TARGET_PRE_SPLIT_STEPS=True there.
"""

from .custom import pre_split_custom

# from .custom.pre_split_custom import minimum_completeness, value_range, allowed_values
# from .custom.pre_split_custom import allowed_countries  # needs an asset, see its Example 4


def build_pre_split_steps(recipe="default"):
    """Return the filter list of one named recipe."""
    recipes = {
        "default": _default_recipe,
        "none": lambda: [],
        "example_complete_inputs": _example_complete_inputs,
        "example_all": _example_all,
    }
    if recipe not in recipes:
        raise ValueError(f"Unknown pre-split recipe: {recipe}. Choose from {list(recipes)}.")
    return recipes[recipe]()


def _default_recipe():
    """Your main recipe. Empty until you uncomment or add filters."""
    return [
        # Built-in step (use your real target column name):
        # {"name": "known_target", "transformer": "DropMissingRows",
        #  "params": {"subset": ["target"], "how": "any"}},
        # Your own filters (custom/pre_split_custom.py):
        # minimum_completeness(columns=["field_a", "field_b", "field_c"], min_present=2),
        # value_range("age", minimum=0, maximum=120),
        # allowed_values("country", ["NL", "DE", "BE"]),
        # allowed_countries(),  # reads assets/countries.json
    ]


def _example_complete_inputs():
    """Keep rows where at least one input has a value."""
    return [
        pre_split_custom.minimum_completeness(columns=["feature_value", "category"], min_present=1)
    ]


def _example_all():
    """Two own filters, in the order they run."""
    return [
        pre_split_custom.minimum_completeness(columns=["feature_value", "category"], min_present=1),
        pre_split_custom.value_range("feature_value", minimum=0, maximum=1_000_000),
    ]
