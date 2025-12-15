from pathlib import Path

from setuptools import find_packages, setup

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8")

setup(
    name="skyulf-core",
    version="0.1.2",
    description="The core machine learning library for Skyulf.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Murat H. Unsal",
    include_package_data=True,
    packages=find_packages(),
    install_requires=[
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "scikit-learn>=1.3.0",
        "polars>=0.19.0",
        "imbalanced-learn>=0.12.0",
        "pydantic>=2.0.0",
        "optuna>=3.0.0",
        "optuna-integration>=3.0.0",
    ],
    extras_require={"dev": ["pytest", "twine", "build"]},
    python_requires=">=3.9",
)
