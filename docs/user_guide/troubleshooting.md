# Troubleshooting

## “Unknown transformer type”

This means your step’s `transformer` string does not match the `FeatureEngineer` dispatcher.
Check spelling and casing.

## Resampling errors about non-numeric columns

Oversampling/undersampling require numeric features.
Apply an encoder first (e.g., OneHotEncoder / OrdinalEncoder) before resampling.

## Pickle loading errors

If `SkyulfPipeline.load()` fails, ensure you are using compatible versions of:

- Python
- scikit-learn
- pandas

If you need portability across environments, consider constraining versions.
