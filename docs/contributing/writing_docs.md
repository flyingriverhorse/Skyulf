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

*   `docs/`: Contains all markdown files.
*   `mkdocs.yml`: The main configuration file.

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
