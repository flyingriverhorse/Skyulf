from dataclasses import dataclass, field
from typing import Any


@dataclass
class NodeMetadata:
    id: str  # noqa: A002, A003  # pylint: disable=redefined-builtin
    name: str
    category: str
    description: str
    params: dict[str, Any] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)


def node_meta(
    id: str,  # noqa: A002, A003  # pylint: disable=redefined-builtin
    name: str,
    category: str,
    description: str,
    params: dict[str, Any] | None = None,
    tags: list[str] | None = None,
):
    """Attach registry metadata and a fallback docstring to a node class."""

    def decorator(cls):
        if not cls.__doc__:
            cls.__doc__ = description
        cls.__node_meta__ = NodeMetadata(
            id=id,
            name=name,
            category=category,
            description=description,
            params=params or {},
            tags=tags or [],
        )
        return cls

    return decorator
