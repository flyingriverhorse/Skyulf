# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.1] - 2025-11-01

### Added
- Lightweight domain inference utilities (`DomainAnalyzer`, `infer_domain`) promoted to the public API
- New API documentation describing domain analysis capabilities
- Example script (`examples/11_domain_analyzer.py`) showcasing dataset domain detection

### Changed
- Bumped package metadata for PyPI publication (version 0.1.1)
- Updated publishing guide and README references for the new EDA functionality

## [0.1.0] - 2025-10-31

### Added
- Initial alpha release
- Basic data ingestion capabilities
- Core feature engineering transformers:
  - Encoding (one-hot, label, ordinal, target, hash)
  - Scaling (standard, minmax, robust)
  - Imputation (simple, advanced)
  - Sampling (over-sampling, under-sampling)
- Pipeline system with automatic split handling
- Preprocessing utilities
- Basic documentation and examples
[Unreleased]: https://github.com/skyulf/skyulf-mlflow/compare/v0.1.1...HEAD
[0.1.1]: https://github.com/skyulf/skyulf-mlflow/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/skyulf/skyulf-mlflow/releases/tag/v0.1.0
