from pathlib import Path

from setuptools import find_packages, setup

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8")

setup(
    name="skyulf-core",
    version="0.5.2",
    description="The core machine learning library for Skyulf.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Murat H. Unsal",
    project_urls={
        "Documentation": "https://flyingriverhorse.github.io/Skyulf",
        "Source": "https://github.com/flyingriverhorse/Skyulf",
        "Changelog": "https://github.com/flyingriverhorse/Skyulf/releases",
    },
    include_package_data=True,
    packages=find_packages(),
    package_data={"skyulf": ["py.typed"]},
    install_requires=[
        "pandas>=2.0.0,<3.0.0",
        "numpy>=1.24.0",
        "scikit-learn>=1.4.0,<2.0.0",
        "polars>=1.36.0",
        "pyarrow>=21.0.0",
        "pydantic>=2.0.0",
        "scipy>=1.10.0",
        "statsmodels>=0.14.0",
    ],
    extras_require={
        "dev": [
            "pytest",
            "pytest-cov>=4.1.0,<5.0.0",
            "twine",
            "build",
            "lizard>=1.17.0",
            "hypothesis>=6.100",
            "syrupy>=4.0.0",
            "pytest-benchmark>=5.0.0,<6.0.0",
        ],
        "viz": ["matplotlib>=3.7.0", "rich>=13.0.0"],
        "eda": [
            "vaderSentiment>=3.3.2",
            "causal-learn>=0.1.3.0",
        ],
        "text": ["vaderSentiment>=3.3.2"],
        "nlp": ["sentence-transformers>=2.2.0"],
        "geo": [
            "geopandas>=0.14.0,<1.2.0",
            "shapely>=2.0.2,<2.2.0",
            "pyproj>=3.6.0,<3.8.0",
            "rtree>=1.3.0,<2.0.0",
            "libpysal>=4.10.0,<5.0.0",
            "esda>=2.5.0,<3.0.0",
            "h3>=4.0.0,<5.0.0",
        ],
        "tuning": [
            "optuna>=3.0.0",
            "optuna-integration>=3.0.0",
            "cmaes>=0.10.0",  # Required by optuna's CmaEsSampler (not bundled with optuna itself)
        ],
        "preprocessing-imbalanced": ["imbalanced-learn>=0.13.0"],
        "modeling-xgboost": ["xgboost>=2.1.4"],
        "modeling-lightgbm": ["lightgbm>=4.0.0"],
        "explainability": ["shap>=0.46.0,<1.0.0"],
        # Convenience aggregate: every optional runtime feature (excludes dev/geo
        # which carry heavy native deps and are opt-in on their own).
        "all": [
            "matplotlib>=3.7.0",
            "rich>=13.0.0",
            "vaderSentiment>=3.3.2",
            "causal-learn>=0.1.3.0",
            "optuna>=3.0.0",
            "optuna-integration>=3.0.0",
            "cmaes>=0.10.0",
            "imbalanced-learn>=0.13.0",
            "xgboost>=2.1.4",
            "lightgbm>=4.0.0",
            "sentence-transformers>=2.2.0",
            "shap>=0.46.0,<1.0.0",
        ],
    },
    python_requires=">=3.12",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.12",
        "Operating System :: OS Independent",
    ],
)
