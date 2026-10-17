"""
Contextual embeddings (a.k.a. Contextual Retrieval).

Problem: once a document is split into chunks, a chunk can lose the
information that made it findable. "The deciding factor was the 12-warehouse
sync" never names the customer, the date, or the deal -- that lives elsewhere
in the call. A bi-encoder embedding and a BM25 index only see the chunk text,
so a query like "what convinced Nordwind to buy?" has nothing to match on.

Fix: before indexing, ask an LLM to write 1-2 sentences that situate each
chunk within its full document, and prepend that to the chunk. The result
("This chunk is from the Nordwind Logistics closed_won call on 2026-02-14,
covering the deciding factor in their evaluation. ...") is what gets embedded
AND what gets BM25-indexed (so both halves of the hybrid benefit).

Design choices:
  * The ORIGINAL chunk text is never modified. Context is stored in a separate
    "context" field, so generation and display still show the real transcript.
  * Results are cached on disk, keyed on (prompt version, model, document,
    chunk). Restarting uvicorn or re-running eval does not re-bill the API.
  * Without ANTHROPIC_API_KEY (or if a single call fails) we fall back to a
    deterministic context built from the chunk's metadata. That is weaker than
    LLM context but free, offline, and still fixes the "chunk never names the
    customer" case.
"""

import hashlib
import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional

log = logging.getLogger(__name__)

# Context generation is a cheap, high-volume, low-reasoning task -> small model.
CONTEXT_MODEL = os.environ.get("CONTEXT_MODEL", "claude-haiku-5-5")

# Bump this to invalidate every cached context after editing the prompt below.
PROMPT_VERSION = "v1"

CACHE_PATH = Path(
    os.environ.get("CONTEXT_CACHE_PATH", Path(__file__).parent / ".context_cache.json")
)

CHUNK_PROMPT = (
    "Here is the chunk we want to situate within the whole document:\n"
    "<chunk>\n{chunk}\n</chunk>\n\n"
    "Write a short, succinct context (1-2 sentences) that situates this chunk "
    "within the overall document, to improve search retrieval of the chunk. "
    "Name the customer and say which part of the call the chunk covers. "
    "Answer with only the context and nothing else."
)


def metadata_context(chunk: Dict[str, Any]) -> str:
    """Deterministic fallback context built purely from chunk metadata."""
    meta = chunk.get("metadata", {})
    customer = meta.get("customer")
    parts = [f"Sales call with {customer}" if customer else "Sales call"]
    if meta.get("date"):
        parts.append(f"on {meta['date']}")
    if meta.get("deal_stage"):
        parts.append(f"(deal stage: {meta['deal_stage']})")
    return " ".join(parts) + "."


def contextualized_text(chunk: Dict[str, Any]) -> str:
    """The text that gets embedded / BM25-indexed: context + original chunk."""
    ctx = chunk.get("context")
    return f"{ctx}\n\n{chunk['text']}" if ctx else chunk["text"]


def _document_block(doc: Dict[str, Any]) -> str:
    meta = {k: v for k, v in doc.items() if k not in ("id", "text")}
    header = ", ".join(f"{k}: {v}" for k, v in meta.items())
    return f"<document>\n{header}\n\n{doc['text']}\n</document>"


def _cache_key(model: str, doc_block: str, chunk_text: str) -> str:
    raw = f"{PROMPT_VERSION}|{model}|{doc_block}|{chunk_text}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _load_cache() -> Dict[str, str]:
    try:
        return json.loads(CACHE_PATH.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _save_cache(cache: Dict[str, str]) -> None:
    try:
        CACHE_PATH.write_text(json.dumps(cache, indent=1))
    except OSError as e:  # read-only filesystem etc. -- cache is an optimisation only
        log.warning("could not write context cache: %s", e)


def _llm_context(client: Any, doc_block: str, chunk_text: str) -> str:
    response = client.messages.create(
        model=CONTEXT_MODEL,
        max_tokens=150,
        messages=[{
            "role": "user",
            "content": [
                # The (long) document goes first and is marked cacheable, so when
                # a document has many chunks the API only pays full price for it
                # once. (Caching only kicks in above the model's minimum prompt
                # length, so it's a no-op for the short calls in the sample data.)
                {"type": "text", "text": doc_block, "cache_control": {"type": "ephemeral"}},
                {"type": "text", "text": CHUNK_PROMPT.format(chunk=chunk_text)},
            ],
        }],
    )
    text = "".join(b.text for b in response.content if b.type == "text").strip()
    if not text:
        raise ValueError("empty context returned")
    return text


def add_context(
    chunks: List[Dict[str, Any]],
    docs: List[Dict[str, Any]],
    mode: str = "auto",
    client: Optional[Any] = None,
    max_workers: int = 8,
) -> List[Dict[str, Any]]:
    """
    Return copies of `chunks`, each with an added "context" string.

    mode:
      "llm"      -- generate context with Claude (needs ANTHROPIC_API_KEY or `client`)
      "metadata" -- deterministic context from metadata only; no API calls
      "auto"     -- "llm" if a key/client is available, else "metadata"
    """
    if mode not in ("auto", "llm", "metadata"):
        raise ValueError(f"unknown mode: {mode!r}")

    if mode == "auto":
        mode = "llm" if (client is not None or os.environ.get("ANTHROPIC_API_KEY")) else "metadata"

    if mode == "metadata":
        log.info("contextual embeddings: metadata-only context (no LLM calls)")
        return [{**c, "context": metadata_context(c)} for c in chunks]

    if client is None:
        import anthropic
        client = anthropic.Anthropic()

    docs_by_id = {d["id"]: d for d in docs}
    cache = _load_cache()
    stats = {"cached": 0, "generated": 0, "fallback": 0}

    def work(chunk: Dict[str, Any]):
        doc_block = _document_block(docs_by_id[chunk["doc_id"]])
        key = _cache_key(CONTEXT_MODEL, doc_block, chunk["text"])
        if key in cache:
            return key, cache[key], "cached"
        try:
            return key, _llm_context(client, doc_block, chunk["text"]), "generated"
        except Exception as e:  # one bad call must not take down index building
            log.warning("context generation failed for %s: %s", chunk["chunk_id"], e)
            return key, metadata_context(chunk), "fallback"

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        results = list(pool.map(work, chunks))

    out = []
    for chunk, (key, ctx, source) in zip(chunks, results):
        stats[source] += 1
        if source == "generated":
            cache[key] = ctx  # never cache fallbacks, so a retry can upgrade them
        out.append({**chunk, "context": ctx})

    if stats["generated"]:
        _save_cache(cache)
    log.info("contextual embeddings: %s", stats)
    return out
