from pathlib import Path

from setuptools import find_packages, setup

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8")

setup(
    name="skyulf-core",
    version="0.1.12",
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
    install_requires=[
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "scikit-learn>=1.4.0,<2.0.0",
        "polars>=1.36.0",
        "pyarrow>=21.0.0",
        "pydantic>=2.0.0",
        "scipy>=1.10.0",
        "statsmodels>=0.14.0",
    ],
    extras_require={
        "dev": ["pytest", "twine", "build"],
        "viz": ["matplotlib>=3.7.0", "rich>=13.0.0"],
        "eda": [
            "vaderSentiment>=3.3.2",
            "causal-learn>=0.1.3.0",
        ],
        "tuning": [
            "optuna>=3.0.0",
            "optuna-integration>=3.0.0",
        ],
        "preprocessing-imbalanced": ["imbalanced-learn>=0.13.0"],
        "modeling-xgboost": ["xgboost>=2.1.4"],
    },
    python_requires=">=3.9",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Operating System :: OS Independent",
    ],
)
