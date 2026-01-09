from typing import Any, Dict, Optional
from dataclasses import dataclass, field

@dataclass
class NodeMetadata:
    id: str
    name: str
    category: str
    description: str
    params: Dict[str, Any] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)


def node_meta(
    id: str,
    name: str,
    category: str,
    description: str,
    params: Optional[Dict[str, Any]] = None,
    tags: Optional[list[str]] = None,
):
    def decorator(cls):
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
