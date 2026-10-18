"""Package custom functions captured with each trained model.

Ordered recipes live in config/pre_split.yml and config/preprocessing.yml;
scoring policy lives in config/scoring.yml. The loader captures these declarations
and supplies their builders during replay. Keep optional custom calculations in
pre_split.py, preprocessing.py and scoring.py, selected explicitly from YAML.
Use relative imports for helpers. The groups/ folder contains upstream Spark
producers and is excluded from model snapshots. Existing versions use saved code.
"""
