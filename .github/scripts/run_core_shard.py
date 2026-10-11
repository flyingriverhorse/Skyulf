"""Run a deterministic file partition of a normally collected pytest suite."""

import argparse
import hashlib

import pytest


class CoreShard:
    """Keep every test in one file on the same runner, retaining fixture locality."""

    def __init__(self, index: int, count: int) -> None:
        """Reject invalid matrix coordinates before any tests are collected."""
        if count < 1 or not 0 <= index < count:
            raise ValueError("The shard index must be within a positive shard count.")
        self.index = index
        self.count = count

    def pytest_collection_modifyitems(
        self, config: pytest.Config, items: list[pytest.Item]
    ) -> None:
        """Select stable repository-relative file hashes after normal collection."""
        selected = []
        excluded = []
        for item in items:
            path = item.path.relative_to(config.rootpath).as_posix()
            digest = hashlib.sha256(path.encode("utf-8")).digest()
            owner = int.from_bytes(digest[:8], "big") % self.count
            (selected if owner == self.index else excluded).append(item)
        config.hook.pytest_deselected(items=excluded)
        items[:] = selected


def main() -> int:
    """Pass explicit pytest arguments through with the selected collection plugin."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--shard-count", type=int, required=True)
    options, pytest_args = parser.parse_known_args()
    if not pytest_args:
        parser.error("Provide explicit pytest paths and options.")
    try:
        plugin = CoreShard(options.shard_index, options.shard_count)
    except ValueError as exc:
        parser.error(str(exc))
    return int(pytest.main(pytest_args, plugins=[plugin]))


if __name__ == "__main__":
    raise SystemExit(main())
