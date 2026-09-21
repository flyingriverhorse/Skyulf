"""Keep node warnings scoped to the execution that emitted them."""

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from threading import Barrier

import pytest

from backend.ml_pipeline._execution.engine import PipelineEngine
from backend.ml_pipeline._execution.engine._warning_capture import WarningCaptureHandler
from backend.ml_pipeline._execution.schemas import (
    NodeConfig,
    NodeExecutionResult,
    PipelineConfig,
    PipelineExecutionResult,
)


def test_overlapping_engine_runs_keep_warnings_and_cleanup_isolated():
    """Concurrent node execution and a failing run must not cross-label warnings."""
    ready = Barrier(3, timeout=5)
    emitted = Barrier(3, timeout=5)
    log = logging.getLogger("skyulf.isolation")
    before = tuple(logging.getLogger("skyulf").handlers)

    def run(name, fail):
        """Use the real engine capture lifecycle around a controlled node."""
        engine = object.__new__(PipelineEngine)

        def execute(node, job_id):
            """Overlap both captures before either node emits its warning."""
            ready.wait()
            log.warning("warning-%s", name)
            emitted.wait()
            if fail:
                raise ValueError(f"failure-{name}")
            return NodeExecutionResult(node_id=node.node_id, status="success")

        engine._execute_node = execute
        config = PipelineConfig(pipeline_id=name, nodes=[NodeConfig(name, "probe")])
        result = PipelineExecutionResult(
            pipeline_id=name, status="success", start_time=datetime.now(UTC)
        )
        engine._run_node_loop(config, name, result)
        return result

    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(run, "first", False)
        second = executor.submit(run, "second", True)
        ready.wait()
        log.warning("outside-any-run")
        emitted.wait()
        results = [first.result(timeout=5), second.result(timeout=5)]

    assert tuple(logging.getLogger("skyulf").handlers) == before
    for result in results:
        other = "second" if result.pipeline_id == "first" else "first"
        assert any(w["message"] == f"warning-{result.pipeline_id}" for w in result.node_warnings)
        assert all(w["node_id"] == result.pipeline_id for w in result.node_warnings)
        assert all(other not in w["message"] for w in result.node_warnings)
        assert all(w["message"] != "outside-any-run" for w in result.node_warnings)
    assert [result.status for result in results] == ["success", "failed"]


@pytest.mark.asyncio
async def test_overlapping_task_contexts_keep_node_warnings_separate():
    """Independent async contexts on one thread must not receive each other's logs."""
    ready = asyncio.Event()
    count = 0

    async def capture(name):
        """Keep both handlers attached across a task suspension."""
        nonlocal count
        handler = WarningCaptureHandler()
        with handler.attach():
            handler.set_current_node(name, "probe")
            count += 1
            if count == 2:
                ready.set()
            await ready.wait()
            logging.getLogger("skyulf.isolation").warning(name)
            await asyncio.sleep(0)
        return handler.drain()

    first, second = await asyncio.gather(capture("first"), capture("second"))
    assert [w["message"] for w in first] == ["first"]
    assert [w["message"] for w in second] == ["second"]


def test_nested_capture_restores_outer_context_after_failure():
    """Nested execution must restore its parent's capture even when it raises."""
    log = logging.getLogger("backend.ml_pipeline.isolation")
    outer = WarningCaptureHandler()
    inner = WarningCaptureHandler()
    before = tuple(logging.getLogger("backend.ml_pipeline").handlers)
    with outer.attach():
        outer.set_current_node("outer", "probe")
        log.warning("before")
        with pytest.raises(ValueError), inner.attach():
            inner.set_current_node("inner", "probe")
            log.warning("inside")
            raise ValueError("controlled failure")
        log.warning("after")
    log.warning("outside")
    assert [w["message"] for w in outer.drain()] == ["before", "after"]
    assert [w["message"] for w in inner.drain()] == ["inside"]
    assert tuple(logging.getLogger("backend.ml_pipeline").handlers) == before
