import sys
import json

from chunking import chunk_documents
from retrieval import HybridIndex
from generate import generate_answer


def main():
    # Usage: python main.py "Question" 
    if len(sys.argv) < 2:
        print('Wrong format!!! Usage: python main.py "Question"')
        sys.exit(1)

    question = sys.argv[1]

    # Step 1: load .json file and split the data into chunks (chunking.py)
    with open("data/sales_calls.json") as f:
        docs = json.load(f)
    chunks = chunk_documents(docs)

    # Step 2: build the search index and retrieve relevant chunks (retrieval.py)
    index = HybridIndex(chunks)
    retrieved = index.hybrid_search(question, top_k=3)
    print(f"\nRetrieved {len(retrieved)} chunks:")
    
    for r in retrieved:
    	display_score = r.get("rerank_score", r["score"])
	print(f"  [{display_score:.4f}] {r['doc_id']} -- {r['text'][:80]}...")

    # Step 3: generate an answer using only the retrieved chunks (generate.py)
    answer = generate_answer(question, retrieved)
    print(f"\nAnswer:\n{answer}")

if __name__ == "__main__":
    main()
