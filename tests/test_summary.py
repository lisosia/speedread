from speedread.extractor import ExtractParams, _chunk_texts, _summarize_recursive


def test_chunk_texts_respects_limit() -> None:
    texts = ["a" * 5, "b" * 5, "c" * 5]
    chunks = _chunk_texts(texts, max_tokens=8)
    assert chunks == ["a" * 5, "b" * 5, "c" * 5]


def test_summarize_recursive_collapses_chunks() -> None:
    params = ExtractParams()

    def summarize(text: str) -> str:
        return "S"

    summary = _summarize_recursive(
        ["a" * 5, "b" * 5, "c" * 5],
        params,
        max_input_tokens=8,
        max_output_tokens=10,
        summarize_fn=summarize,
    )
    assert summary == "S"


def test_summarize_recursive_stops_if_not_reducing() -> None:
    params = ExtractParams()

    def summarize(text: str) -> str:
        return text

    summary = _summarize_recursive(
        ["a" * 5, "b" * 5],
        params,
        max_input_tokens=4,
        max_output_tokens=10,
        summarize_fn=summarize,
    )
    assert summary == ("a" * 5 + "\n\n" + "b" * 5)
