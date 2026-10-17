"""
Tests for contextualize.py.

These use a fake Anthropic client, so they need no API key, no network, and
no embedding models.

Run with:
    python test_contextualize.py
"""

import contextualize
from contextualize import add_context, contextualized_text, metadata_context
from chunking import chunk_documents

DOCS = [
    {"id": "call_a", "customer": "Acme Corp", "date": "2026-01-05", "deal_stage": "churned",
     "text": " ".join(f"alpha{i}" for i in range(300))},   # long -> several chunks
    {"id": "call_b", "customer": "Birk GmbH", "date": "2026-02-01", "deal_stage": "closed_won",
     "text": "Short call about pricing."},
]


class _Block:
    type = "text"
    def __init__(self, text): self.text = text


class FakeClient:
    """Mimics client.messages.create(); records every call."""
    def __init__(self, fail_on=None):
        self.calls = []
        self.fail_on = fail_on
        self.messages = self

    def create(self, **kwargs):
        self.calls.append(kwargs)
        chunk_prompt = kwargs["messages"][0]["content"][1]["text"]
        if self.fail_on and self.fail_on in chunk_prompt:
            raise RuntimeError("boom")
        return type("R", (), {"content": [_Block("  Generated context.  ")]})()


def _use_tmp_cache(tmp_path):
    # Accept either a file path (script mode) or a directory (pytest's tmp_path).
    contextualize.CACHE_PATH = tmp_path / "cache.json" if tmp_path.is_dir() else tmp_path


def test_llm_context_added_and_text_untouched(tmp_path):
    _use_tmp_cache(tmp_path)
    chunks = chunk_documents(DOCS)
    out = add_context(chunks, DOCS, mode="llm", client=FakeClient())
    assert len(out) == len(chunks)
    for before, after in zip(chunks, out):
        assert after["text"] == before["text"], "original chunk text must not change"
        assert after["context"] == "Generated context.", "context should be stripped"
        assert "context" not in before, "input chunks must not be mutated"
    print("PASS: context added, original text and input list untouched")


def test_prompt_contains_document_and_is_cacheable(tmp_path):
    _use_tmp_cache(tmp_path)
    client = FakeClient()
    add_context(chunk_documents(DOCS)[:1], DOCS, mode="llm", client=client)
    content = client.calls[0]["messages"][0]["content"]
    assert "Acme Corp" in content[0]["text"] and "alpha299" in content[0]["text"], "full doc + metadata expected"
    assert content[0].get("cache_control") == {"type": "ephemeral"}
    print("PASS: prompt carries the full document (cacheable) plus the chunk")


def test_disk_cache_avoids_second_api_call(tmp_path):
    _use_tmp_cache(tmp_path)
    chunks = chunk_documents(DOCS)
    first = FakeClient()
    add_context(chunks, DOCS, mode="llm", client=first)
    assert len(first.calls) == len(chunks)

    second = FakeClient()
    out = add_context(chunks, DOCS, mode="llm", client=second)
    assert second.calls == [], "second run should be served entirely from cache"
    assert all(c["context"] == "Generated context." for c in out)
    print("PASS: second run makes zero API calls")


def test_failed_call_falls_back_and_is_not_cached(tmp_path):
    _use_tmp_cache(tmp_path)
    chunks = chunk_documents(DOCS)
    out = add_context(chunks, DOCS, mode="llm", client=FakeClient(fail_on="Short call"))
    short = next(c for c in out if c["doc_id"] == "call_b")
    assert short["context"] == metadata_context(short), "failed call should fall back to metadata context"

    retry = FakeClient()
    add_context(chunks, DOCS, mode="llm", client=retry)
    assert len(retry.calls) == 1, "only the previously-failed chunk should be retried"
    print("PASS: failure falls back per-chunk, and is retried (not cached) next time")


def test_auto_mode_without_key_is_offline(tmp_path):
    _use_tmp_cache(tmp_path)
    import os
    saved = os.environ.pop("ANTHROPIC_API_KEY", None)
    try:
        out = add_context(chunk_documents(DOCS), DOCS, mode="auto")
    finally:
        if saved is not None:
            os.environ["ANTHROPIC_API_KEY"] = saved
    assert all(c["context"].startswith("Sales call with") for c in out)
    print("PASS: no API key -> metadata-only context, no network")


def test_metadata_context_is_robust_to_missing_fields():
    assert metadata_context({"metadata": {}}) == "Sales call."
    assert metadata_context({}) == "Sales call."
    c = metadata_context({"metadata": {"customer": "X", "date": "2026-01-01", "deal_stage": "churned"}})
    assert c == "Sales call with X on 2026-01-01 (deal stage: churned)."
    print("PASS: metadata_context handles missing fields")


def test_contextualized_text():
    assert contextualized_text({"text": "body"}) == "body"
    assert contextualized_text({"text": "body", "context": ""}) == "body"
    assert contextualized_text({"text": "body", "context": "ctx"}) == "ctx\n\nbody"
    print("PASS: contextualized_text prefixes context only when present")


if __name__ == "__main__":
    import tempfile
    from pathlib import Path

    def fresh():
        return Path(tempfile.mkdtemp()) / "cache.json"

    test_llm_context_added_and_text_untouched(fresh())
    test_prompt_contains_document_and_is_cacheable(fresh())
    test_disk_cache_avoids_second_api_call(fresh())
    test_failed_call_falls_back_and_is_not_cached(fresh())
    test_auto_mode_without_key_is_offline(fresh())
    test_metadata_context_is_robust_to_missing_fields()
    test_contextualized_text()
    print("\nAll contextualize tests passed.")
