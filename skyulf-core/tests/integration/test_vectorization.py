"""Integration tests for the text vectorization nodes.

The unit tests (``tests/unit/test_text_vectorization.py``,
``tests/unit/test_vectorization_gaps.py``) drive the Calculator/Applier classes
directly. This file takes the integration angle: every node under test is
resolved through :class:`~skyulf.registry.NodeRegistry` — the same path the
backend and the ML canvas use — proving the node-id → class wiring is intact,
then exercises the fixture-backed scenarios from
``tests/test_cases/preprocessing/{text_vectorization,vectorization_gaps}.json``
plus end-to-end checks on the frames the nodes produce.
"""

import importlib
import logging
from typing import Any, cast

import numpy as np
import pandas as pd
import polars as pl
import pytest
from tests.utils.test_case_loader import TestCaseLoader

from skyulf.preprocessing.base import BaseApplier, BaseCalculator
from skyulf.registry import NodeRegistry

VECTORIZER_IDS = ["count_vectorizer", "tfidf_vectorizer", "hashing_vectorizer", "tokenizer"]


def _load_single_param(source_path: str, group: str) -> list[Any]:
    """Load a fixture group whose param string is a single ``node`` name.

    The loader returns one 1-tuple per scenario; unwrapping them lets pytest
    bind the single param name directly.
    """
    _, scenarios = TestCaseLoader(source_path, group=group).load()
    return [scenario[0] for scenario in scenarios]


_NOOP_NODES = _load_single_param("preprocessing/text_vectorization", "empty_columns_noop")
_POLARS_ROUNDTRIP = TestCaseLoader(
    "preprocessing/text_vectorization", group="apply_polars_roundtrip"
).load_with_ids()
_MISSING_VECTORIZER = _load_single_param(
    "preprocessing/vectorization_gaps", "apply_missing_vectorizer"
)
_INVALID_FIT_COLUMNS = TestCaseLoader(
    "preprocessing/vectorization_gaps", group="invalid_fit_columns"
).load_with_ids()
_ALL_COLS_MISSING = TestCaseLoader(
    "preprocessing/vectorization_gaps", group="apply_all_columns_missing"
).load_with_ids()
_DROP_ORIGINAL = TestCaseLoader(
    "preprocessing/vectorization_gaps", group="drop_original"
).load_with_ids()
_LARGE_VOCAB = TestCaseLoader(
    "preprocessing/vectorization_gaps", group="large_vocab_warning"
).load_with_ids()
_POLARS_FIT_INPUT = TestCaseLoader(
    "preprocessing/vectorization_gaps", group="polars_fit_input"
).load_with_ids()


def _parametrize(cases: tuple[str, list[Any], list[str]]):
    """Parametrize decorator from a ``load_with_ids()`` 3-tuple.

    ``ids`` must be a keyword — the third positional slot of
    ``pytest.mark.parametrize`` is ``indirect``, not ``ids``.
    """
    params, scenarios, ids = cases
    return pytest.mark.parametrize(params, scenarios, ids=ids)


def _resolve(node: str) -> tuple[type[BaseCalculator], type[BaseApplier]]:
    """Resolve a registered node id to its (Calculator, Applier) classes."""
    # The registry getters are annotated to return bare ``type``.
    return cast(type[BaseCalculator], NodeRegistry.get_calculator(node)), cast(
        type[BaseApplier], NodeRegistry.get_applier(node)
    )


def _fit(calculator_cls: type[BaseCalculator], df: Any, config: dict[str, Any]) -> dict[str, Any]:
    """Fit ``calculator_cls`` on ``df`` and return the params dict.

    ``df`` is ``Any`` because the base ``fit`` annotation predates the polars
    inputs these nodes accept, and the return is cast because the base
    annotation returns ``Mapping`` while ``apply`` expects a ``dict``.
    """
    return cast(dict[str, Any], calculator_cls().fit(df, config))


def _corpus() -> pd.DataFrame:
    """Small, deterministic text corpus with a known bag-of-words."""
    return pd.DataFrame(
        {
            "text": [
                "the quick brown fox",
                "the lazy dog",
                "fox and dog sleep",
                "",
                "the end",
            ]
        }
    )


# ==================== Registry resolution ====================


class TestRegistryResolution:
    """The four vectorizer node ids resolve through the registry the UI uses."""

    @pytest.mark.parametrize("node", VECTORIZER_IDS)
    def test_node_resolves_via_registry(self, node: str) -> None:
        calculator_cls, applier_cls = _resolve(node)
        assert issubclass(calculator_cls, BaseCalculator)
        assert issubclass(applier_cls, BaseApplier)

    @pytest.mark.parametrize("node", VECTORIZER_IDS)
    def test_metadata_category_is_text(self, node: str) -> None:
        meta = NodeRegistry.get_all_metadata()[node]
        assert meta["id"] == node
        assert meta["category"] == "Text"

    def test_unknown_node_raises(self) -> None:
        with pytest.raises(ValueError):
            NodeRegistry.get_calculator("nonexistent_vectorizer")

    def test_text_nodes_in_transformer_listing(self) -> None:
        text_nodes = NodeRegistry.list_transformers(category="Text")
        assert set(VECTORIZER_IDS) <= set(text_nodes)


# ==================== No-op paths (fixture-backed) ====================


class TestNoOpPaths:
    """Invalid or empty configurations degrade to no-ops, never raise."""

    @pytest.mark.parametrize("node", _NOOP_NODES)
    def test_empty_columns_fit_is_noop(self, node: str) -> None:
        calculator_cls, _ = _resolve(node)
        assert calculator_cls().fit(_corpus(), {"columns": []}) == {}

    @_parametrize(_INVALID_FIT_COLUMNS)
    def test_invalid_columns_fit_is_noop(self, node: str, columns: list[str]) -> None:
        calculator_cls, _ = _resolve(node)
        assert calculator_cls().fit(_corpus(), {"columns": columns}) == {}

    @pytest.mark.parametrize("node", _MISSING_VECTORIZER)
    def test_apply_without_fitted_vectorizer_is_passthrough(self, node: str) -> None:
        _, applier_cls = _resolve(node)
        frame = _corpus()
        result = applier_cls().apply(frame, {"columns": ["text"]})
        assert list(result.columns) == ["text"]

    @_parametrize(_ALL_COLS_MISSING)
    def test_apply_when_columns_absent_is_passthrough(
        self, node: str, fit_extra: dict[str, Any]
    ) -> None:
        calculator_cls, applier_cls = _resolve(node)
        params = _fit(calculator_cls, _corpus(), {"columns": ["text"], **fit_extra})
        frame = pd.DataFrame({"other": ["x"]})
        result = applier_cls().apply(frame, params)
        assert list(result.columns) == ["other"]


# ==================== drop_original ====================


@pytest.mark.parametrize("node", ["count_vectorizer", "tfidf_vectorizer"])
class TestEmptyVocabulary:
    """An unlearnable vocabulary must be a visible, non-destructive no-op."""

    @pytest.mark.parametrize("drop_original", [False, True])
    @pytest.mark.parametrize(
        "texts,options",
        [
            (["", None, "  "], {}),
            (["the and", "is of"], {"stop_words": "english"}),
            (["a", "b b", "!"], {}),
            (["alpha", "beta"], {"ngram_range": [2, 2]}),
            ([], {}),
        ],
        ids=["blank", "stop-words", "no-tokens", "no-ngrams", "no-rows"],
    )
    def test_empty_vocabulary_warns_and_preserves_training_and_replay(
        self,
        node: str,
        texts: list[str | None],
        options: dict[str, Any],
        drop_original: bool,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """An empty vocabulary must preserve rows, targets and text in both drop modes."""
        calculator_cls, applier_cls = _resolve(node)
        frame = pd.DataFrame({"text": pd.array(texts, dtype="string"), "value": range(len(texts))})
        frame.index = pd.Index([9] * len(texts))
        original = frame.copy(deep=True)
        y = pd.Series(range(len(texts)), index=frame.index, name="target")
        with caplog.at_level(logging.WARNING, logger="skyulf"):
            params = _fit(
                calculator_cls,
                (frame, y),
                {"columns": ["text"], "drop_original": drop_original, **options},
            )

        assert params == {}
        assert "empty vocabulary" in caplog.text.lower()
        assert "unchanged" in caplog.text.lower()
        out, y_out = applier_cls().apply((frame, y), params)
        pd.testing.assert_frame_equal(out, original)
        pd.testing.assert_frame_equal(frame, original)
        pd.testing.assert_series_equal(y_out, y)

        later = pd.DataFrame({"text": ["new usable vocabulary"], "value": [42]})
        prediction = applier_cls().apply(later, params)
        pd.testing.assert_frame_equal(prediction, later)

    @pytest.mark.parametrize(
        "options,error",
        [
            ({"min_df": -1}, "min_df"),
            ({"ngram_range": [2, 1]}, "ngram_range"),
            ({"stop_words": "invalid"}, "stop_words"),
            ({"min_df": 2}, "After pruning"),
            ({"min_df": 2, "max_df": 0.1}, "max_df corresponds"),
        ],
    )
    def test_invalid_settings_and_frequency_pruning_still_raise(
        self, node: str, options: dict[str, Any], error: str
    ) -> None:
        """Empty-vocabulary recovery must not swallow distinct parameter or pruning errors."""
        calculator_cls, _ = _resolve(node)
        frame = pd.DataFrame({"text": ["alpha beta", "gamma delta"]})
        with pytest.raises(ValueError, match=error):
            _fit(calculator_cls, frame, {"columns": ["text"], **options})


class TestDropOriginal:
    @_parametrize(_DROP_ORIGINAL)
    def test_original_column_dropped(self, node: str, fit_extra: dict[str, Any]) -> None:
        calculator_cls, applier_cls = _resolve(node)
        params = _fit(
            calculator_cls, _corpus(), {"columns": ["text"], "drop_original": True, **fit_extra}
        )
        result = applier_cls().apply(_corpus(), params)
        assert "text" not in result.columns


# ==================== Polars round-trips (fixture-backed) ====================


class TestPolars:
    @_parametrize(_POLARS_ROUNDTRIP)
    def test_fit_pandas_apply_polars(
        self, node: str, fit_extra: dict[str, Any], check_columns: bool
    ) -> None:
        calculator_cls, applier_cls = _resolve(node)
        polars_frame = pl.from_pandas(_corpus())
        params = _fit(calculator_cls, polars_frame.to_pandas(), {"columns": ["text"], **fit_extra})
        result = applier_cls().apply(cast(Any, polars_frame), params)
        assert isinstance(result, pl.DataFrame)
        if check_columns:
            for col in params["output_columns"]:
                assert col in result.columns

    @_parametrize(_POLARS_FIT_INPUT)
    def test_fit_accepts_polars_input(self, node: str, fit_extra: dict[str, Any]) -> None:
        calculator_cls, _ = _resolve(node)
        frame = pl.DataFrame({"text": ["hello world", "foo bar baz"]})
        params = _fit(calculator_cls, frame, {"columns": ["text"], **fit_extra})
        assert params["columns"] == ["text"]
        assert len(params["output_columns"]) > 0


# ==================== Large-output warning path ====================


class TestLargeVocabWarning:
    @_parametrize(_LARGE_VOCAB)
    def test_large_output_warning_is_emitted(
        self,
        node: str,
        module_path: str,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        module = importlib.import_module(module_path)
        forced = "forced large-output warning for tests"
        monkeypatch.setattr(module, "_warn_large_output", lambda output_cols, **kwargs: forced)
        calculator_cls, _ = _resolve(node)
        with caplog.at_level(logging.WARNING):
            calculator_cls().fit(pd.DataFrame({"text": ["hello world"]}), {"columns": ["text"]})
        assert forced in caplog.text


# ==================== End-to-end value checks ====================


class TestCountVectorizerEndToEnd:
    """Bag-of-words counts verified by hand against the known corpus."""

    def test_fit_and_apply_token_counts(self) -> None:
        calculator_cls, applier_cls = _resolve("count_vectorizer")
        corpus = _corpus()
        params = _fit(calculator_cls, corpus, {"columns": ["text"]})
        result = applier_cls().apply(corpus, params)

        assert params["type"] == "count_vectorizer"
        assert params["columns"] == ["text"]
        assert params["vocabulary"] == {
            "and": 0,
            "brown": 1,
            "dog": 2,
            "end": 3,
            "fox": 4,
            "lazy": 5,
            "quick": 6,
            "sleep": 7,
            "the": 8,
        }
        assert params["output_columns"] == [
            f"text__count__{token}" for token in sorted(params["vocabulary"])
        ]
        assert result["text__count__fox"].tolist() == [1, 0, 1, 0, 0]
        assert result["text__count__the"].tolist() == [1, 1, 0, 0, 1]
        assert result["text__count__dog"].tolist() == [0, 1, 1, 0, 0]
        assert result["text__count__quick"].tolist() == [1, 0, 0, 0, 0]
        # drop_original defaults to False — the source column survives.
        assert "text" in result.columns

    def test_out_of_sample_tokens_are_ignored(self) -> None:
        calculator_cls, applier_cls = _resolve("count_vectorizer")
        params = _fit(calculator_cls, _corpus(), {"columns": ["text"]})
        unseen = applier_cls().apply(pd.DataFrame({"text": ["zebra quantum"]}), params)
        assert unseen["text__count__fox"].tolist() == [0]
        assert unseen["text__count__the"].tolist() == [0]


class TestTfidfVectorizerEndToEnd:
    def test_fit_and_apply_weights(self) -> None:
        calculator_cls, applier_cls = _resolve("tfidf_vectorizer")
        corpus = _corpus()
        params = _fit(calculator_cls, corpus, {"columns": ["text"]})
        result = applier_cls().apply(corpus, params)

        assert params["type"] == "tfidf_vectorizer"
        assert all(col.startswith("text__tfidf__") for col in params["output_columns"])
        assert len(params["idf"]) == len(params["vocabulary"])

        idf = {token: params["idf"][index] for token, index in params["vocabulary"].items()}
        # "fox" appears in 2 of 5 docs, "the" in 3 — rarer tokens get higher idf.
        assert idf["fox"] > idf["the"] > 0

        values = result[params["output_columns"]].to_numpy()
        assert (values >= 0).all()
        # Default norm="l2": every non-empty row has unit norm.
        norms = np.linalg.norm(values, axis=1)
        np.testing.assert_allclose(norms[norms > 0], 1.0, atol=1e-6)


class TestHashingVectorizerEndToEnd:
    @pytest.mark.parametrize(
        ("norm_config", "expected_norm", "expected_counts"),
        [
            pytest.param({"norm": "none"}, None, [3.0, 4.0], id="canvas-none"),
            pytest.param({"norm": None}, None, [3.0, 4.0], id="python-none"),
            pytest.param({"norm": ""}, None, [3.0, 4.0], id="empty-string"),
            pytest.param({"norm": "l1"}, "l1", [3 / 7, 4 / 7], id="l1"),
            pytest.param({"norm": "l2"}, "l2", [0.6, 0.8], id="l2"),
            pytest.param({}, "l2", [0.6, 0.8], id="default-l2"),
        ],
    )
    def test_normalization_controls_applied_counts(
        self,
        norm_config: dict[str, Any],
        expected_norm: str | None,
        expected_counts: list[float],
    ) -> None:
        """Canvas None must preserve token counts through fit and later transforms."""
        calculator_cls, applier_cls = _resolve("hashing_vectorizer")
        corpus = pd.DataFrame({"text": ["hello hello hello world world world world", ""]})
        params = _fit(
            calculator_cls,
            corpus,
            {
                "columns": ["text"],
                "n_features": 16,
                "alternate_sign": False,
                "drop_original": True,
                **norm_config,
            },
        )

        fit_out = applier_cls().apply(corpus, params)
        later_out = applier_cls().apply(corpus.copy(), params)

        assert params["norm"] == expected_norm
        assert fit_out.shape == (2, 16)
        np.testing.assert_allclose(
            np.sort(fit_out.iloc[0].to_numpy()), [0.0] * 14 + expected_counts
        )
        np.testing.assert_array_equal(fit_out.iloc[1].to_numpy(), np.zeros(16))
        pd.testing.assert_frame_equal(later_out, fit_out)

    def test_stateless_fit_and_apply(self) -> None:
        calculator_cls, applier_cls = _resolve("hashing_vectorizer")
        corpus = _corpus()
        params = _fit(calculator_cls, corpus, {"columns": ["text"], "n_features": 64})

        assert params["type"] == "hashing_vectorizer"
        assert len(params["output_columns"]) == 64
        assert all(col.startswith("text__hash__") for col in params["output_columns"])

        fit_out = applier_cls().apply(corpus, params)
        # Stateless: an unseen corpus produces the exact same output columns.
        unseen_out = applier_cls().apply(
            pd.DataFrame({"text": ["completely novel sentence"]}), params
        )
        assert list(unseen_out.columns) == list(fit_out.columns)
        # Hashing values are signed, so only assert the row is not all-zero.
        assert unseen_out[params["output_columns"]].to_numpy().any()


class TestTokenizerEndToEnd:
    def test_tokens_and_counts(self) -> None:
        calculator_cls, applier_cls = _resolve("tokenizer")
        params = _fit(calculator_cls, _corpus(), {"columns": ["text"], "add_token_count": True})
        assert params["type"] == "tokenizer"
        assert params["output_columns"] == ["text__tokens", "text__token_count"]

        result = applier_cls().apply(_corpus(), params)
        assert result["text__tokens"].tolist() == [
            "the quick brown fox",
            "the lazy dog",
            "fox and dog sleep",
            "",
            "the end",
        ]
        assert result["text__token_count"].tolist() == [4, 3, 4, 0, 2]
