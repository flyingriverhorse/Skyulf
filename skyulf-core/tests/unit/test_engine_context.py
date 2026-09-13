"""Engine fallback selection must stay with the caller that configured it."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextvars import Context, copy_context
from threading import Event

import pandas as pd
import polars as pl
import pytest

from skyulf.engines.pandas_engine import PandasEngine
from skyulf.engines.polars_engine import PolarsEngine
from skyulf.engines.registry import BaseEngine, EngineRegistry


@pytest.mark.parametrize("data", [None, [1, 2], object()], ids=["none", "list", "unknown"])
def test_fresh_context_defaults_to_polars(data):
    """Unconfigured callers must use Polars regardless of earlier context overrides."""
    resolved = Context().run(EngineRegistry.resolve, data)
    assert resolved is PolarsEngine
    frame = resolved.create_dataframe({"value": [1, 2]})
    assert isinstance(frame, pl.DataFrame)
    assert frame.to_dict(as_series=False) == {"value": [1, 2]}


@pytest.mark.parametrize("data", [None, [1, 2], object()], ids=["none", "list", "unknown"])
def test_overlapping_threads_keep_their_engine_fallback(data):
    """A second caller's setter must not redirect an already configured thread."""
    pandas_ready = Event()
    polars_ready = Event()

    def resolve_pandas():
        """Resolve only after the other thread has installed its override."""
        EngineRegistry.set_active_engine("pandas")
        pandas_ready.set()
        assert polars_ready.wait(timeout=5)
        return EngineRegistry.resolve(data)

    def resolve_polars():
        """Force the second setter between the first setter and its resolve."""
        assert pandas_ready.wait(timeout=5)
        EngineRegistry.set_active_engine("polars")
        polars_ready.set()
        return EngineRegistry.resolve(data)

    original = EngineRegistry.resolve()
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            pandas_result = pool.submit(resolve_pandas)
            polars_result = pool.submit(resolve_polars)
            assert pandas_result.result(timeout=10) is PandasEngine
            assert polars_result.result(timeout=10) is PolarsEngine
    finally:
        EngineRegistry.set_active_engine(original.name)


def test_async_siblings_keep_overrides_and_children_inherit():
    """Sibling tasks must isolate overrides while child tasks inherit their parent."""

    async def exercise():
        """Order sibling writes so the former shared-state race is deterministic."""
        pandas_ready = asyncio.Event()
        polars_ready = asyncio.Event()
        EngineRegistry.set_active_engine("pandas")

        async def read_current():
            """Observe the selection inherited when this task was created."""
            return EngineRegistry.resolve()

        async def pandas_task():
            """Observe the inherited default after the sibling changes its own."""
            pandas_ready.set()
            await polars_ready.wait()
            return EngineRegistry.resolve([1])

        async def polars_task():
            """Install an override and pass it to a new child task."""
            await pandas_ready.wait()
            EngineRegistry.set_active_engine("polars")
            polars_ready.set()
            return await asyncio.create_task(read_current())

        siblings = await asyncio.wait_for(asyncio.gather(pandas_task(), polars_task()), timeout=5)
        return siblings, EngineRegistry.resolve()

    original = EngineRegistry.resolve()
    try:
        siblings, parent = asyncio.run(exercise())
        assert siblings == [PandasEngine, PolarsEngine]
        assert parent is PandasEngine
    finally:
        EngineRegistry.set_active_engine(original.name)


def test_independent_thread_starts_with_default_engine():
    """A caller override must not become the implicit default of a new thread."""
    original = EngineRegistry.resolve()
    try:
        EngineRegistry.set_active_engine("pandas")
        with ThreadPoolExecutor(max_workers=1) as pool:
            resolved = pool.submit(EngineRegistry.resolve).result(timeout=5)
        assert resolved is PolarsEngine
        assert EngineRegistry.resolve() is PandasEngine
    finally:
        EngineRegistry.set_active_engine(original.name)


def test_registered_plugin_can_be_context_fallback_without_changing_detection(monkeypatch):
    """Plugin selection must remain supported without overriding native input detection."""

    class PluginEngine(BaseEngine):
        """Represent a registered third-party adapter with its own registration name."""

    monkeypatch.setitem(EngineRegistry._engines, "plugin_context_test", PluginEngine)

    def check_plugin():
        """Exercise only the public setter and resolution API in a copied context."""
        EngineRegistry.set_active_engine("plugin_context_test")
        assert EngineRegistry.resolve() is PluginEngine
        assert EngineRegistry.resolve([1]) is PluginEngine
        assert EngineRegistry.resolve(pd.DataFrame({"a": [1]})) is PandasEngine
        assert EngineRegistry.resolve(pl.DataFrame({"a": [1]})) is PolarsEngine
        with pytest.raises(ValueError, match="not found"):
            EngineRegistry.set_active_engine("missing_context_engine")
        assert EngineRegistry.resolve() is PluginEngine

    original = EngineRegistry.resolve()
    try:
        copy_context().run(check_plugin)
        assert EngineRegistry.resolve() is original
    finally:
        EngineRegistry.set_active_engine(original.name)
