from openai_io._utils import merge_headers


def test_merge_headers_overrides_case_insensitively() -> None:
    assert merge_headers({"Accept": "application/json"}, {"accept": "text/plain"}) == {"accept": "text/plain"}


def test_merge_headers_none_removes_header() -> None:
    assert merge_headers({"Authorization": "Bearer secret"}, {"authorization": None}) == {}
