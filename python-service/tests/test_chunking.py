"""
Tests for chunking.py (advanced/hybrid version -- with overlap).

Run with:
    python test_chunking.py

This uses plain `assert` statements rather than a testing framework
(pytest/unittest) so it can be run directly with zero extra dependencies --
appropriate for a project this size. If this were a larger codebase, you'd
want pytest for better failure reporting and test discovery.
"""

from chunking import chunk_text, chunk_documents


def test_short_text_returns_single_chunk():
    """A document shorter than chunk_size should come back as ONE chunk,
    completely unchanged -- there's nothing to split."""
    text = "This is a short sentence with few words."
    result = chunk_text(text, chunk_size=120, overlap=30)
    assert len(result) == 1, f"expected 1 chunk, got {len(result)}"
    assert result[0] == text, "short text should be returned unchanged"
    print("PASS: short text returns single unchanged chunk")


def test_chunk_size_is_respected():
    """No chunk should ever contain more than chunk_size words."""
    # Build a fake document of exactly 300 words: "word0", "word1", ... "word299"
    text = " ".join(f"word{i}" for i in range(300))
    chunks = chunk_text(text, chunk_size=120, overlap=30)

    for i, chunk in enumerate(chunks):
        word_count = len(chunk.split())
        assert word_count <= 120, f"chunk {i} has {word_count} words, expected <=120"
    print(f"PASS: all {len(chunks)} chunks respect the 120-word limit")


def test_overlap_is_correct():
    """The last `overlap` words of chunk N should exactly match the first
    `overlap` words of chunk N+1 -- that's the whole point of overlap."""
    text = " ".join(f"word{i}" for i in range(300))
    chunks = chunk_text(text, chunk_size=120, overlap=30)

    for i in range(len(chunks) - 1):
        current_words = chunks[i].split()
        next_words = chunks[i + 1].split()

        # Last 30 words of the current chunk...
        tail = current_words[-30:]
        # ...should equal the first 30 words of the next chunk.
        head = next_words[:30]

        assert tail == head, (
            f"overlap mismatch between chunk {i} and {i+1}:\n"
            f"  tail: {tail}\n  head: {head}"
        )
    print(f"PASS: overlap is consistent across all {len(chunks)-1} chunk boundaries")


def test_no_words_lost():
    """Every word in the original document should appear in at least one
    chunk -- chunking should never silently drop content."""
    text = " ".join(f"word{i}" for i in range(300))
    chunks = chunk_text(text, chunk_size=120, overlap=30)

    all_words_in_chunks = set()
    for chunk in chunks:
        all_words_in_chunks.update(chunk.split())

    original_words = set(text.split())
    missing = original_words - all_words_in_chunks
    assert not missing, f"these words were lost during chunking: {missing}"
    print("PASS: no words were lost during chunking")


def test_invalid_overlap_raises_error():
    """overlap >= chunk_size should fail loudly (infinite loop otherwise),
    not silently misbehave."""
    text = " ".join(f"word{i}" for i in range(300))
    try:
        chunk_text(text, chunk_size=50, overlap=50)  # overlap == chunk_size, invalid
        assert False, "expected a ValueError, but no error was raised"
    except ValueError:
        print("PASS: invalid overlap correctly raises ValueError")


def test_chunk_documents_preserves_doc_id_and_metadata():
    """Every chunk produced from a document should carry that document's
    doc_id and a copy of its metadata (everything except id/text)."""
    docs = [
        {"id": "doc_1", "text": " ".join(f"word{i}" for i in range(200)),
         "customer": "Acme Corp", "deal_stage": "closed_won"},
    ]
    chunks = chunk_documents(docs, chunk_size=120, overlap=30)

    assert len(chunks) > 1, "expected this 200-word doc to split into multiple chunks"
    for c in chunks:
        assert c["doc_id"] == "doc_1", "doc_id should match the source document"
        assert c["metadata"]["customer"] == "Acme Corp", "metadata should be copied onto every chunk"
        assert c["metadata"]["deal_stage"] == "closed_won"
        assert "text" not in c["metadata"], "text itself should not leak into metadata"
    print(f"PASS: all {len(chunks)} chunks correctly carry doc_id and metadata")


def test_chunk_ids_are_unique_and_sequential():
    """chunk_id should be unique across the whole corpus and numbered
    sequentially within each document (doc_1_0, doc_1_1, ...)."""
    docs = [
        {"id": "doc_1", "text": " ".join(f"word{i}" for i in range(200))},
        {"id": "doc_2", "text": " ".join(f"word{i}" for i in range(50))},
    ]
    chunks = chunk_documents(docs, chunk_size=120, overlap=30)

    chunk_ids = [c["chunk_id"] for c in chunks]
    assert len(chunk_ids) == len(set(chunk_ids)), "chunk_ids should all be unique"

    doc_1_ids = [c["chunk_id"] for c in chunks if c["doc_id"] == "doc_1"]
    expected = [f"doc_1_{i}" for i in range(len(doc_1_ids))]
    assert doc_1_ids == expected, f"expected {expected}, got {doc_1_ids}"
    print("PASS: chunk_ids are unique and sequential per document")


if __name__ == "__main__":
    test_short_text_returns_single_chunk()
    test_chunk_size_is_respected()
    test_overlap_is_correct()
    test_no_words_lost()
    test_invalid_overlap_raises_error()
    test_chunk_documents_preserves_doc_id_and_metadata()
    test_chunk_ids_are_unique_and_sequential()
    print("\nAll chunking tests passed.")
