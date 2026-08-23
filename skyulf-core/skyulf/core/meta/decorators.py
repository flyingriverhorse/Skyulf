from dataclasses import dataclass, field
from typing import Any


@dataclass
class NodeMetadata:
    id: str  # noqa: A002, A003  # pylint: disable=redefined-builtin
    name: str
    category: str
    description: str
    learns_from_data: bool
    params: dict[str, Any] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)
    is_splitter: bool = False


def node_meta(
    id: str,  # noqa: A002, A003  # pylint: disable=redefined-builtin
    name: str,
    category: str,
    description: str,
    params: dict[str, Any] | None = None,
    tags: list[str] | None = None,
    *,
    learns_from_data: bool,
    is_splitter: bool = False,
):
    """Attach registry metadata and a fallback docstring to a node class.

    ``learns_from_data`` is required: it feeds the registry-derived leakage
    gate, and omitting it is a decoration error rather than a silent opt-out.
    ``is_splitter`` marks nodes that create a train/test boundary.
    """

    def decorator(cls):
        if not cls.__doc__:
            cls.__doc__ = description
        cls.__node_meta__ = NodeMetadata(
            id=id,
            name=name,
            category=category,
            description=description,
            learns_from_data=learns_from_data,
            params=params or {},
            tags=tags or [],
            is_splitter=is_splitter,
        )
        return cls

    return decorator
