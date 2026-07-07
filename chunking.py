import json
from typing import List, Dict, Any


def chunk_text(text: str, chunk_size: int = 120) -> List[str]:

    # Flat list of individual words
    words = text.split()
    chunks = []

    for start in range(0, len(words), chunk_size):
        chunk_words = words[start:start + chunk_size]
        chunks.append(" ".join(chunk_words))

    return chunks


def chunk_documents(docs: List[Dict[str, Any]], chunk_size: int = 120) -> List[Dict[str, Any]]:
    chunk_records = []

    for doc in docs:
        pieces = chunk_text(doc["text"], chunk_size=chunk_size)
        for i, piece in enumerate(pieces):
            chunk_records.append({
                # e.g. "call_001_0" for the first chunk of document call_001
                "chunk_id": f"{doc['id']}_{i}",
                # Keep track of the original document, so if I retrieve this chunk later, I know which source it came from.
                "doc_id": doc["id"],
                "text": piece,
            })

    return chunk_records



"""
if __name__ == "__main__":
    with open("data/sales_calls.json") as f:
        docs = json.load(f)

    chunks = chunk_documents(docs)
    print(f"{len(docs)} documents -> {len(chunks)} chunks")
    print(json.dumps(chunks[0], indent=2))
"""
