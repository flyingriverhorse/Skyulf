# Publishing Skyulf Core Kaggle Notebooks

## Curated notebooks

Publish these repository notebooks in this order:

1. `skyulf-core/examples/01_house_prices_regression.ipynb`
2. `skyulf-core/examples/02_disaster_tweets_text_classification.ipynb`
3. `skyulf-core/examples/07_spaceship_titanic_classification.ipynb`

They demonstrate regression, text classification, and tabular classification while each generates a submission artifact.

## Pre-publication verification

Run each notebook locally from the repository checkout before uploading it:

```bash
python -m pip install -e "skyulf-core[eda,viz,tuning,preprocessing-imbalanced,modeling-xgboost,modeling-lightgbm,explainability]"
python -m pip install jupyter
jupyter nbconvert --to notebook --execute --inplace skyulf-core/examples/01_house_prices_regression.ipynb
jupyter nbconvert --to notebook --execute --inplace skyulf-core/examples/02_disaster_tweets_text_classification.ipynb
jupyter nbconvert --to notebook --execute --inplace skyulf-core/examples/07_spaceship_titanic_classification.ipynb
```

## Publication rules

- Publish from the tested notebook revision and name the exact released `skyulf-core` version in the notebook introduction.
- Link to `https://github.com/flyingriverhorse/Skyulf` and `https://pypi.org/project/skyulf-core/`.
- Preserve each dataset source and sampling caveat already documented in `skyulf-core/examples/data/`.
- Do not include credentials, API tokens, local file paths, customer data, or unpublished SaaS details.
- Add a short note that Skyulf Core is the public Python library and that the hosted platform is not publicly available.
