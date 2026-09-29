from mcp.server.mcpserver import MCPServer
import json

from chunking import chunk_documents
from retrieval import HybridIndex
from generate import generate_answer

mcp = MCPServer("hybrid-rag-sales-calls")

# Build the index once at startup, not per-request — loading the
# SentenceTransformer + CrossEncoder models is the expensive part.
with open("data/sales_calls.json") as f:
    docs = json.load(f)
chunks = chunk_documents(docs)
index = HybridIndex(chunks)


@mcp.tool()
def query_sales_calls(
    question: str,
    stage: str = None,
    customer: str = None,
    top_k: int = 3,
) -> str:
    """
    Answer a question about sales call transcripts using hybrid RAG
    (dense + sparse retrieval, RRF fusion, cross-encoder rerank).

    stage: optional exact-match deal stage filter (e.g. 'in_negotiation', 'closed_won', 'churned')
    customer: optional case-insensitive substring match on customer name
    top_k: how many chunks to retrieve (default 3)
    """
    metadata_filter = None
    if stage or customer:
        def metadata_filter(meta, stage=stage, customer=customer):
            if stage and meta.get("deal_stage") != stage:
                return False
            if customer and customer.lower() not in meta.get("customer", "").lower():
                return False
            return True

    retrieved = index.hybrid_search(question, top_k=top_k, metadata_filter=metadata_filter)

    if not retrieved:
        return "No matching call transcripts found for that question/filter combination."

    answer = generate_answer(question, retrieved)

    # Optional: surface which chunks backed the answer, useful for a client to show its work
    sources = "\n".join(
        f"- {r['metadata'].get('customer', '?')} ({r['metadata'].get('deal_stage', '?')}): "
        f"{r.get('rerank_score', r['score']):.4f}"
        for r in retrieved
    )
    return f"{answer}\n\nSources:\n{sources}"


if __name__ == "__main__":
    mcp.run()
