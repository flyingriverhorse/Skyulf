from core.feature_engineering.nodes.feature_eng.utils import _coerce_string_list


def test_coerce_string_list_handles_nested_sequences():
    values = [["dataset", "column_a"], ["other", "column_b"]]
    assert _coerce_string_list(values) == ["column_a", "column_b"]


def test_coerce_string_list_handles_strings_with_commas():
    assert _coerce_string_list("a, b ,c") == ["a", "b", "c"]


def test_coerce_string_list_handles_mappings():
    values = [{"column": "foo"}, {"name": "bar"}, {"label": " baz "}]
    assert _coerce_string_list(values) == ["foo", "bar", "baz"]


def test_coerce_string_list_skips_empty_entries():
    values = [None, "  ", ["", "target"]]
    assert _coerce_string_list(values) == ["target"]
