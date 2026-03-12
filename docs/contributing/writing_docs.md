# Writing Documentation

We use [MkDocs](https://www.mkdocs.org/) with the [Material for MkDocs](https://squidfunk.github.io/mkdocs-material/) theme to build our documentation.

## Setup

1.  **Install Dependencies**:
    ```bash
    pip install mkdocs mkdocs-material mkdocstrings[python]
    ```

2.  **Serve Locally**:
    ```bash
    mkdocs serve
    ```
    This will start a local server at `http://127.0.0.1:8000` that auto-reloads when you change files.

## Directory Structure

The `docs/` folder is organized by audience and purpose:

```
docs/
  index.md                  # Landing page
  architecture.md           # System architecture overview
  data_architecture.md      # Data flow and storage design
  performance.md            # Benchmark results
  guides/                   # Onboarding and how-to guides
    getting_started.md
    recipes.md
    ...
  user_guide/               # In-depth usage documentation
    installation.md
    pipeline_quickstart.md
    configuration.md
    hyperparameter_tuning.md
    drift_monitoring.md
    troubleshooting.md
    extending_custom_nodes.md
    ...
  examples/                 # Runnable examples and proofs
    quickstart.md
    leakage_proof.md
  reference/                # Auto-generated API docs (mkdocstrings)
    preprocessing_nodes.md
    modeling_nodes.md
    api/                    # Module-level API reference
  contributing/             # Contributor guides
    writing_docs.md
```

*   `mkdocs.yml`: The main configuration file (nav, plugins, theme).

## Adding a New Page

1.  Create a new `.md` file in `docs/`.
2.  Add the file to the `nav` section in `mkdocs.yml`.

## Writing API Documentation

We use `mkdocstrings` to auto-generate API docs from Python docstrings.

To document a module, class, or function, use the `:::` directive:

```markdown
::: core.my_module.MyClass
```

### Docstring Style

We follow the **Google Style Guide** for docstrings.

```python
def my_function(arg1: int, arg2: str) -> bool:
    """
    Does something amazing.

    Args:
        arg1: The first argument.
        arg2: The second argument.

    Returns:
        True if successful, False otherwise.
    """
    ...
```
