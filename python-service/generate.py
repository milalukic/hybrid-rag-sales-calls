import os
from typing import List, Dict, Any

SYSTEM_PROMPT = (
    "Answer the question using ONLY the information in the provided context. "
    "If the answer isn't in the context, say so explicitly instead of guessing."
)


def build_prompt(question: str, chunks: List[Dict[str, Any]]) -> str:
    """Combine the retrieved chunks and the question into one prompt string."""

    context = "\n\n---\n\n".join(c["text"] for c in chunks)
    return f"Context:\n\n{context}\n\nQuestion: {question}\n\nAnswer using only the context above."


def generate_answer(question: str, chunks: List[Dict[str, Any]]) -> str:
    api_key = os.environ.get("ANTHROPIC_API_KEY")

    if not api_key:
        # OFFLINE FALLBACK
        if not chunks:
            return "[no context retrieved -- cannot answer]"
        top_chunk = chunks[0]
        return f"[offline fallback -- set ANTHROPIC_API_KEY for a real answer]\n{top_chunk['text']}"

    prompt = build_prompt(question, chunks)
    import anthropic
    client = anthropic.Anthropic(api_key=api_key)

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=500,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )

    return "".join(block.text for block in response.content if block.type == "text")


'''
if __name__ == "__main__":
    import json
    from chunking import chunk_documents
    from retrieval import SimpleIndex

    with open("data/sales_calls.json") as f:
        docs = json.load(f)
    chunks = chunk_documents(docs)
    index = SimpleIndex(chunks)

    question = "Why did Halberg Retail Group churn?"

    # Step 1: retrieve the most relevant chunks for this question.
    retrieved = index.search(question, top_k=3)

    # Step 2: generate (or fall back to) an answer using only those chunks.
    answer = generate_answer(question, retrieved)

    print(answer)
    
'''
