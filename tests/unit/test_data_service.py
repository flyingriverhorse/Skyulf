"""Unit tests for DataService's Polars paths.

Covers the post-migration shapes that had no dedicated tests:

- ``get_sample``'s eager fallback when the Polars lazy scan is unavailable
  (returns ``None``), which must still load and sample the file.
- ``save_artifact``'s engine dispatch: Polars-native writes and the
  pandas→Polars conversion write.
"""

import pandas as pd
import polars as pl

from backend.services.data_service import DataService


async def test_get_sample_falls_back_to_eager_load(tmp_path, monkeypatch) -> None:
    """When the lazy Polars scan yields nothing, ``get_sample`` must fall back
    to an eager load and still return the first ``limit`` rows as dicts."""
    path = tmp_path / "data.csv"
    path.write_text("a\n1\n2\n3\n")
    service = DataService()
    monkeypatch.setattr(service, "_try_polars_lazy_sample", lambda *args: None)

    rows = await service.get_sample(path, limit=2)

    assert rows == [{"a": 1}, {"a": 2}]


async def test_get_sample_eager_fallback_samples_polars_frame(tmp_path) -> None:
    """The eager path's sampler must recognize a Polars frame and use its
    native ``to_dicts`` instead of the pandas ``to_dict`` protocol."""
    path = tmp_path / "data.parquet"
    pl.DataFrame({"a": [1, 2, 3]}).write_parquet(path)
    service = DataService()

    rows = await service.get_sample(path, limit=2)

    assert rows == [{"a": 1}, {"a": 2}]


async def test_save_artifact_writes_polars_native(tmp_path) -> None:
    """A Polars frame must be written directly (no pandas round-trip)."""
    path = tmp_path / "out.parquet"
    await DataService().save_artifact(pl.DataFrame({"a": [1, 2]}), path)
    assert pl.read_parquet(path)["a"].to_list() == [1, 2]


async def test_save_artifact_converts_pandas_via_polars(tmp_path) -> None:
    """A pandas frame must be converted and written via the Polars fast path."""
    path = tmp_path / "out.parquet"
    await DataService().save_artifact(pd.DataFrame({"a": [1, 2]}), path)
    assert pl.read_parquet(path)["a"].to_list() == [1, 2]


async def test_save_artifact_writes_wrapped_polars_native(tmp_path) -> None:
    """A SkyulfPolarsWrapper must be written via its native polars frame
    (unwrap, no pandas round-trip), pinning the wrapper save path."""
    from skyulf.engines.polars_engine import SkyulfPolarsWrapper

    path = tmp_path / "out.parquet"
    wrapper = SkyulfPolarsWrapper(pl.DataFrame({"a": [1, 2]}))
    await DataService().save_artifact(wrapper, path)
    assert pl.read_parquet(path)["a"].to_list() == [1, 2]


def test_save_polars_native_unwraps_via_to_native(tmp_path) -> None:
    """A Polars-routed object without its own ``write_parquet`` must be
    unwrapped through the public ``to_native()`` accessor."""

    class _NoWriteParquet:
        def __init__(self) -> None:
            self._frame = pl.DataFrame({"a": [3, 4]})

        def to_native(self) -> pl.DataFrame:
            return self._frame

    path = tmp_path / "out.parquet"
    DataService()._save_polars_native(_NoWriteParquet(), str(path))
    assert pl.read_parquet(path)["a"].to_list() == [3, 4]


def test_save_polars_native_falls_back_to_pandas_conversion(tmp_path) -> None:
    """When the unwrapped frame lacks ``write_parquet`` (should not happen
    for PolarsEngine data), the defensive path converts via pandas."""

    class _OddNative:
        def to_native(self) -> pd.DataFrame:
            return pd.DataFrame({"a": [5, 6]})

    path = tmp_path / "out.parquet"
    DataService()._save_polars_native(_OddNative(), str(path))
    assert pl.read_parquet(path)["a"].to_list() == [5, 6]


def test_should_use_polars_respects_force_type() -> None:
    service = DataService()
    assert service._should_use_polars(None) is True
    assert service._should_use_polars("polars") is True
    assert service._should_use_polars("pandas") is False


def test_sample_from_loaded_data_pandas_frame() -> None:
    """A pandas frame (no ``to_pandas`` attribute) must be sampled through the
    pandas ``to_dict`` protocol, not mistaken for Polars."""
    rows = DataService()._sample_from_loaded_data(pd.DataFrame({"a": [1, 2]}), 5)
    assert rows == [{"a": 1}, {"a": 2}]


def test_sample_from_loaded_data_wrapper_with_to_pandas() -> None:
    """A non-Polars wrapper exposing ``to_pandas`` must be sampled via its
    pandas conversion, not treated as a Polars frame."""

    class _Wrapper:
        def __init__(self, df: pd.DataFrame) -> None:
            self._df = df

        def to_pandas(self) -> pd.DataFrame:
            return self._df

    rows = DataService()._sample_from_loaded_data(_Wrapper(pd.DataFrame({"a": [1, 2]})), 5)
    assert rows == [{"a": 1}, {"a": 2}]
