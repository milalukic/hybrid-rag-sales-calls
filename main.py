import sys
import json
import argparse

from chunking import chunk_documents
from retrieval import HybridIndex
from generate import generate_answer
	
def main():
	# Usage: python main.py "Question" 
	parser = argparse.ArgumentParser(description="Hybrid RAG demo over sales call transcripts")
	parser.add_argument("question", type=str, help='The question to ask, e.g. "why did halberg churn?"')
	
	parser.add_argument("--stage", type=str, default=None, help="Filter to a deal_stage, e.g. in_negotiation, closed_won, churned")
	parser.add_argument("--customer", type=str, default=None, help="Filter to a specific customer name (partial match, case-insensitive)")
	parser.add_argument("--top_k", type=int, default=3, help="How many chunks to retrieve (default 3)")
 
	args = parser.parse_args()

	# Step 1: load .json file and split the data into chunks (chunking.py)
	with open("data/sales_calls.json") as f:
		docs = json.load(f)
	chunks = chunk_documents(docs)

	# Step 2: build the search index and retrieve relevant chunks (retrieval.py)
	index = HybridIndex(chunks)
	# if the user doesn't ask for a filter every chunk will be a candidate
	metadata_filter = None
    
	if args.stage or args.customer:
		def metadata_filter(meta, stage=args.stage, customer=args.customer):
			if stage and meta.get("deal_stage") != stage:
				return False
			if customer and customer.lower() not in meta.get("customer", "").lower():
				return False
			return True
    
	retrieved = index.hybrid_search(args.question, top_k=args.top_k, metadata_filter=metadata_filter)
	print(f"\nRetrieved {len(retrieved)} chunks:")
    
	for r in retrieved:
		display_score = r.get("rerank_score", r["score"])
		customer = r["metadata"].get("customer", "?")
		stage = r["metadata"].get("deal_stage", "?")
		print(f"  [{display_score:.4f}] {customer} ({stage}) -- {r['text'][:80]}...")

	# Step 3: generate an answer using only the retrieved chunks (generate.py)
	answer = generate_answer(args.question, retrieved)
	print(f"\nAnswer:\n{answer}")

if __name__ == "__main__":
    main()
