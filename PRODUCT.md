# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Stack

The public landing surface is static HTML, CSS, and lightweight JavaScript. The product includes a React node-canvas frontend, FastAPI backend, Celery or background-thread job execution, and the standalone Python package `skyulf-core`.

## Users

Data scientists, ML engineers, developers, and product teams that need to explore data, build reproducible machine-learning pipelines, train and compare models, and operate deployments without handing their workflow to a black-box SaaS product.

## Product Purpose

Skyulf connects the machine-learning lifecycle from data exploration through monitoring. Users can work through a self-hosted visual platform or use the independent `skyulf-core` Python package directly in scripts and notebooks.

## Positioning

One machine-learning workflow with two first-class interfaces: a visual, self-hosted MLOps workspace and an Apache-2.0 Python engine. Visual work remains portable through Jupyter notebook export rather than being trapped inside the canvas.

## Operating Context

- Evaluate the product through the hosted demo without signup.
- Run the platform locally with the repository start scripts.
- Self-host the full stack with Docker Compose.
- Install `skyulf-core` from PyPI for use without the web platform.
- Store data and artifacts locally or in S3-compatible storage.
- Begin with SQLite and use PostgreSQL for larger deployments.

## Capabilities and Constraints

- Data ingestion for CSV, Excel, JSON, Parquet, and S3-compatible storage.
- Automated EDA with statistical profiling, distributions, correlations, outliers, segmentation, causal discovery, and target analysis.
- Node-based preprocessing, feature engineering, splitting, and modeling pipelines.
- Background training through Celery and Redis or background threads.
- Experiment comparison, model registry, deployment, inference, drift monitoring, audit history, and operational diagnostics.
- Full and compact Jupyter notebook export.
- No fabricated benchmarks, customer claims, usage statistics, testimonials, or commercial metrics may appear in the landing-page concepts.

## Brand Commitments

- Product name: Skyulf.
- Communicate with direct, technically credible language.
- Present the visual platform and Python engine as equal, complementary entry paths.
- Avoid generic SaaS language, fake numbers, robotic interface decoration, and simulated product experiences that duplicate the real live demo.

## Evidence on Hand

- Public repository: `flyingriverhorse/Skyulf`.
- Hosted demo: `https://api.skyulf.com`.
- PyPI package: `skyulf-core`.
- Existing product screenshots under `static/img/`.
- Repository documentation, tests, CI workflows, licenses, and runnable setup instructions.

## Product Principles

1. Visual and code workflows must remain connected.
2. Users retain control of infrastructure, data, code, and exported artifacts.
3. Product claims must be demonstrated with real interfaces, documentation, and capabilities.
4. The workflow continues beyond training into model operation and monitoring.

## Accessibility & Inclusion

Landing surfaces must support keyboard navigation, semantic HTML, responsive layouts, reduced-motion preferences, visible focus states, and WCAG AA text contrast.
