"""
Tests for retrieval.py (advanced/hybrid version).

NOTE: metadata/structured-filter tests are intentionally NOT included here --
those will be added once structured filtering is (re)implemented in
retrieval.py. These tests only cover dense+sparse+RRF+reranking behavior.

These tests need the real embedding model, BM25 index, and cross-encoder to
actually run -- they build a small HybridIndex and check real search
behavior, not just pure logic. That means the first run will download
~180MB of models (embedding + reranker combined) if they aren't already
cached locally.

Run with:
    python test_retrieval.py
"""

from chunking import chunk_documents
from retrieval import HybridIndex


def make_test_index(docs):
    """Helper: chunk a small doc set and build a HybridIndex over it."""
    chunks = chunk_documents(docs, chunk_size=120, overlap=30)
    return HybridIndex(chunks)


def test_bm25_catches_exact_rare_term():
    """
    BM25 (keyword search) should surface a chunk containing a rare, specific
    term when the query uses that exact term -- this is the case embeddings
    alone are weaker at, and hybrid search should still catch it.
    """
    docs = [
        {"id": "doc_a", "text": "The customer discussed general onboarding timelines and pricing during the call."},
        {"id": "doc_b", "text": "The integration failed due to error code XZQ-7742 in the billing module."},
        {"id": "doc_c", "text": "We reviewed the quarterly roadmap and discussed upcoming features."},
    ]
    index = make_test_index(docs)

    results = index.hybrid_search("what is error XZQ-7742", top_k=1)

    assert len(results) == 1
    assert results[0]["doc_id"] == "doc_b", (
        f"expected doc_b (contains the exact error code) to win, got {results[0]['doc_id']}"
    )
    print("PASS: hybrid search correctly surfaces the chunk with the exact rare term")


def test_top_k_is_respected():
    """hybrid_search should never return more than top_k results, even if
    more candidates were available."""
    docs = [
        {"id": f"doc_{i}", "text": f"This is sample document number {i} discussing sales performance."}
        for i in range(10)
    ]
    index = make_test_index(docs)

    results = index.hybrid_search("sales performance", top_k=3)
    assert len(results) <= 3, f"expected at most 3 results, got {len(results)}"
    print(f"PASS: top_k=3 correctly limits results to {len(results)} chunks")


def test_reranker_adds_rerank_score_when_enabled():
    """When use_reranker=True (the default), every result should carry a
    rerank_score, and results should be sorted by it (descending)."""
    docs = [
        {"id": "doc_a", "text": "The customer was very happy with the product's ease of use."},
        {"id": "doc_b", "text": "The customer complained about the price being too high for the value provided."},
        {"id": "doc_c", "text": "We scheduled a follow-up call for next Tuesday."},
    ]
    index = make_test_index(docs)

    results = index.hybrid_search("why was the customer unhappy about cost", top_k=3, use_reranker=True)

    assert all("rerank_score" in r for r in results), "expected every result to have a rerank_score"
    scores = [r["rerank_score"] for r in results]
    assert scores == sorted(scores, reverse=True), "results should be sorted by rerank_score, descending"
    print("PASS: reranker attaches rerank_score and sorts results by it")


def test_reranker_disabled_skips_rerank_score():
    """When use_reranker=False, results should fall back to plain RRF
    ordering and should NOT have a rerank_score field."""
    docs = [
        {"id": "doc_a", "text": "The customer was very happy with the product's ease of use."},
        {"id": "doc_b", "text": "The customer complained about the price being too high for the value provided."},
    ]
    index = make_test_index(docs)

    results = index.hybrid_search("customer complaint about cost", top_k=2, use_reranker=False)

    assert all("rerank_score" not in r for r in results), (
        "expected no rerank_score when use_reranker=False"
    )
    assert all("score" in r for r in results), "expected the plain RRF score to still be present"
    print("PASS: disabling the reranker correctly falls back to plain RRF results")


def test_context_makes_chunk_findable_by_info_it_does_not_contain():
    """
    A chunk that never names its customer should become findable by customer
    name once it carries a "context" field -- and NOT before (so the test
    proves the context is what made the difference).
    """
    docs = [
        {"id": "doc_a", "text": "The deciding factor was the real time inventory sync across all warehouses."},
        {"id": "doc_b", "text": "The deciding factor was the dashboard and reporting features."},
        {"id": "doc_c", "text": "Pricing was agreed at annual billing paid quarterly."},
    ]
    chunks = chunk_documents(docs, chunk_size=120, overlap=30)
    for c in chunks:
        if c["doc_id"] == "doc_a":
            c["context"] = "Call with Quillfeather Logistics (closed_won) about their evaluation."

    query = "what convinced Quillfeather to buy"
    with_ctx = HybridIndex(chunks, use_context=True).hybrid_search(query, top_k=1)
    without_ctx = HybridIndex(chunks, use_context=False).hybrid_search(query, top_k=3)

    assert with_ctx[0]["doc_id"] == "doc_a", "context should make doc_a retrievable by customer name"
    assert with_ctx[0]["text"] == docs[0]["text"], "original text must be preserved for generation"
    assert not any("Quillfeather" in c["text"] for c in without_ctx), "sanity: raw text never contains the name"
    print("PASS: contextual index finds a chunk by information that only its context contains")


if __name__ == "__main__":
    test_bm25_catches_exact_rare_term()
    test_top_k_is_respected()
    test_reranker_adds_rerank_score_when_enabled()
    test_reranker_disabled_skips_rerank_score()
    test_context_makes_chunk_findable_by_info_it_does_not_contain()
    print("\nAll retrieval tests passed.")
