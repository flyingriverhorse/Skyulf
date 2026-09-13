"""Concurrent embedding requests must share loading without serializing different models."""

import sys
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from types import ModuleType

import pytest

from skyulf.preprocessing.vectorization import sentence_embedder as embedder


@pytest.mark.parametrize("fails", [False, True])
def test_same_model_shares_one_load_and_releases_waiters(monkeypatch, fails):
    """Concurrent cold-cache callers share success or failure, and failed loads can retry."""
    started = threading.Event()
    ready = threading.Event()
    release = threading.Event()
    guard = threading.Lock()
    attempts = []
    waiters = []
    model = object()

    class ObservedFuture(Future):
        """Observe blocked followers without replacing actual Future synchronization."""

        def result(self, timeout=None):
            """Signal once every follower is waiting on the single model load."""
            with guard:
                waiters.append(threading.get_ident())
                if len(waiters) == 3:
                    ready.set()
            return super().result(timeout)

    def construct(name):
        """Pause the real cache boundary so duplicate constructions cannot finish early."""
        with guard:
            attempts.append(name)
            if len(attempts) == 4:
                ready.set()
        started.set()
        assert release.wait(timeout=10)
        if fails:
            raise ValueError("model unavailable")
        return model

    module = ModuleType("sentence_transformers")
    monkeypatch.setattr(module, "SentenceTransformer", construct, raising=False)
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)
    monkeypatch.setattr(embedder, "_MODEL_CACHE", {})
    monkeypatch.setattr(embedder, "Future", ObservedFuture, raising=False)
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(embedder._load_model, "shared")]
        assert started.wait(timeout=10)
        futures.extend(pool.submit(embedder._load_model, "shared") for _ in range(3))
        try:
            assert ready.wait(timeout=10)
            assert attempts == ["shared"]
        finally:
            release.set()
        if fails:
            for future in futures:
                with pytest.raises(ValueError, match="model unavailable"):
                    future.result(timeout=10)
        else:
            assert all(future.result(timeout=10) is model for future in futures)

    if fails:
        assert "shared" not in embedder._MODEL_CACHE
        monkeypatch.setattr(module, "SentenceTransformer", lambda name: model)
    assert embedder._load_model("shared") is model
    assert embedder._load_model("shared") is model


def test_different_model_loads_can_progress_concurrently(monkeypatch):
    """A slow model must not hold a global lock across other model constructors."""
    barrier = threading.Barrier(2)

    def construct(name):
        """Require both distinct constructors to overlap before either can finish."""
        barrier.wait(timeout=10)
        return name

    module = ModuleType("sentence_transformers")
    monkeypatch.setattr(module, "SentenceTransformer", construct, raising=False)
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)
    monkeypatch.setattr(embedder, "_MODEL_CACHE", {})
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(embedder._load_model, name) for name in ("first", "second")]
        assert [future.result(timeout=15) for future in futures] == ["first", "second"]


@pytest.mark.parametrize("error_type", [ValueError, KeyboardInterrupt])
def test_waiter_can_retry_before_failed_owner_finishes(monkeypatch, error_type):
    """Publishing a failure must permit immediate retries without deleting their pending load."""
    first_started = threading.Event()
    first_waiter = threading.Event()
    fail_first = threading.Event()
    release_owner = threading.Event()
    retry_observed = threading.Event()
    release_retry = threading.Event()
    last_joined = threading.Event()
    guard = threading.Lock()
    attempts = []
    waiters = []
    model = object()

    class PublishingFuture(Future):
        """Expose the scheduling window after real Future waiters have been notified."""

        def set_exception(self, exception):
            """Keep the failed owner alive while its waiter starts another model load."""
            super().set_exception(exception)
            assert release_owner.wait(timeout=10)

        def result(self, timeout=None):
            """Observe both the first failure waiter and the follower of its retry."""
            with guard:
                waiters.append(threading.get_ident())
                first_waiter.set()
                if len(waiters) == 2:
                    last_joined.set()
            return super().result(timeout)

    def construct(name):
        """Block both attempts and reveal an erroneous duplicate of the still-running retry."""
        with guard:
            attempts.append(name)
            attempt = len(attempts)
        if attempt == 1:
            first_started.set()
            assert fail_first.wait(timeout=10)
            raise error_type("model unavailable")
        retry_observed.set()
        if attempt > 2:
            last_joined.set()
        assert release_retry.wait(timeout=10)
        return model

    def retry_after_failure():
        """Retry directly from a notified waiter before the original loader has returned."""
        with pytest.raises(error_type, match="model unavailable"):
            embedder._load_model("retry")
        try:
            return embedder._load_model("retry")
        finally:
            retry_observed.set()

    module = ModuleType("sentence_transformers")
    monkeypatch.setattr(module, "SentenceTransformer", construct, raising=False)
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)
    monkeypatch.setattr(embedder, "_MODEL_CACHE", {})
    monkeypatch.setattr(embedder, "_MODEL_LOADS", {})
    monkeypatch.setattr(embedder, "Future", PublishingFuture)
    with ThreadPoolExecutor(max_workers=3) as pool:
        owner = pool.submit(embedder._load_model, "retry")
        try:
            assert first_started.wait(timeout=10)
            retry = pool.submit(retry_after_failure)
            assert first_waiter.wait(timeout=10)
            fail_first.set()
            assert retry_observed.wait(timeout=10)
            assert attempts == ["retry", "retry"]
            release_owner.set()
            with pytest.raises(error_type, match="model unavailable"):
                owner.result(timeout=10)
            follower = pool.submit(embedder._load_model, "retry")
            assert last_joined.wait(timeout=10)
            assert attempts == ["retry", "retry"]
        finally:
            fail_first.set()
            release_owner.set()
            release_retry.set()
        assert retry.result(timeout=10) is model
        assert follower.result(timeout=10) is model
    assert embedder._load_model("retry") is model
    assert attempts == ["retry", "retry"]
