"""Export optional scoring hooks captured with each trained model.

Ordered feature recipes live in config/pre_split.yml and config/preprocessing.yml.
The loader captures those declarations and supplies their builders during replay.
Keep custom calculations in pre_split.py and preprocessing.py; use relative imports
for model helpers. The groups/ folder is reserved for upstream Spark table producers
and excluded from model snapshots. Existing model versions use their saved source.
"""

from .scoring import build_combined_rules, build_model_rules, build_scoring

__all__ = ["build_scoring", "build_model_rules", "build_combined_rules"]
