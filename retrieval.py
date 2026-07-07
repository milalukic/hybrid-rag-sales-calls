from typing import List, Dict, Any
import numpy as np
from sentence_transformers import SentenceTransformer


class SimpleIndex:
    def __init__(self, chunks: List[Dict[str, Any]], embedding_model: str = "all-MiniLM-L6-v2"):
        """
        chunks: output of chunking.chunk_documents() -- a list of
            {"chunk_id", "doc_id", "text"} dicts.
        """
        # Keep the original chunk records so that I can look up their text/doc_id later
        self.chunks = chunks

        # Pull out just the raw text of every chunk
        self.texts = [c["text"] for c in chunks]

        # Load a small pretrained neural network that turns any sentence into a fixed-length vector of numbers capturing its meaning. 
        # This downloads a small(~90MB) model file from Hugging Face the first time it's ran, then reuses the cached copy after that.
        self.model = SentenceTransformer(embedding_model)

        # Run EVERY chunk's text through the model once, up front, and store
        # the resulting vectors. We do this once here (not per-search) so
        # that searching later is fast -- we only have to embed the (much
        # shorter) query at search time.
        #
        # normalize_embeddings=True rescales every vector to have length 1, so I can use dot product instead of a more complex similarity score

        self.embeddings = self.model.encode(self.texts, normalize_embeddings=True)

    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Return the top_k chunks whose meaning is closest to `query`."""

        # Turn the question into a vector using the ****exact same model**** 
        query_embedding = self.model.encode([query], normalize_embeddings=True)[0]

        scores = self.embeddings @ query_embedding

     
        top_indices = np.argsort(-scores)[:top_k]

        # Convert the winning positions back into full chunk records (with their text and doc_id), and attach each one's similarity score so
        # the caller can see how confident the match was.
        results = []
        for i in top_indices:
            record = dict(self.chunks[i])  # shallow copy
            record["score"] = float(scores[i])
            results.append(record)

        return results


'''
if __name__ == "__main__":
    import json
    from chunking import chunk_documents

    with open("data/sales_calls.json") as f:
        docs = json.load(f)
    chunks = chunk_documents(docs)


    index = SimpleIndex(chunks)

    query = "why did customers cancel due to cost"
    print(f"=== Query: '{query}' ===")
    for r in index.search(query, top_k=3):
        print(f"[{r['score']:.4f}] {r['doc_id']}: {r['text'][:100]}...")
'''
