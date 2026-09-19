from typing import List, Dict, Any, Optional, Callable
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder


class HybridIndex:
	def __init__(self, chunks: List[Dict[str, Any]], embedding_model: str = "all-MiniLM-L6-v2",  reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
		"""
		chunks: output of chunking.chunk_documents() -- a list of{"chunk_id", "doc_id", "text"} dicts.
		"""

		self.chunks = chunks
		self.texts = [c["text"] for c in chunks]

		# sparse index (BM25) - good at direct word search 
		tokenized = [t.lower().split() for t in self.texts]
		self.bm25 = BM25Okapi(tokenized)


		# dense index (good at synonyms)
		self.model = SentenceTransformer(embedding_model)
		self.embeddings = self.model.encode(self.texts, normalize_embeddings=True)

		# re-ranking - structurally different from sentence transformer 
		# takes query + chunk and combines them at input 
		self.reranker = CrossEncoder(reranker_model)
	
	def _apply_filter(self, metadata_filter: Optional[Callable[[Dict], bool]]) -> List[int]:
		if metadata_filter is None:
			return list(range(len(self.chunks)))
		return [i for i, c in enumerate(self.chunks) if metadata_filter(c["metadata"])]
        
	def dense_search(self, query: str, indices: List[int], top_k: int) -> List[int]:
		query_emb = self.model.encode([query], normalize_embeddings=True)[0]
		sub_embeddings = self.embeddings[indices]
		scores = sub_embeddings @ query_emb
		ranked_local = np.argsort(-scores)[:top_k]
		return [indices[i] for i in ranked_local]
 
	def sparse_search(self, query: str, indices: List[int], top_k: int) -> List[int]:
		scores = self.bm25.get_scores(query.lower().split())
		sub_scores = [(i, scores[i]) for i in indices]
		sub_scores.sort(key=lambda x: -x[1])
		return [i for i, _ in sub_scores[:top_k]]

	
	def rerank(self, query: str, candidates: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
		"""
		Re-score a small candidate list with the cross-encoder and return the
		top_k, re-sorted by the cross-encoder's score instead of the RRF score.
		"""
		if not candidates:
			return []
 
        	# The CrossEncoder expects a list of (query, passage) PAIRSa nd returns one relevance score per pair in the same order.
		pairs = [(query, c["text"]) for c in candidates]
		rerank_scores = self.reranker.predict(pairs)
 
        	# Attach each candidate's new cross-encoder score, then sort descending (higher = more relevant) and keep only the final top_k.
		for c, score in zip(candidates, rerank_scores):
			c["rerank_score"] = float(score)
 
		reranked = sorted(candidates, key=lambda c: -c["rerank_score"])
		return reranked[:top_k]

	
	def hybrid_search(	
		self,
		query: str,
		top_k: int = 5,
		metadata_filter: Optional[Callable[[Dict], bool]] = None,
		rrf_k: int = 60,
		candidate_pool: int = 20,
		use_reranker: bool = True,
		) -> List[Dict[str, Any]]:
			"""
			Run dense + sparse search, merge with RRF, then (by default) re-rank
			the merged candidates with a cross-encoder for final precision.
	       		"""
	       		
			indices = self._apply_filter(metadata_filter)
	
			if not indices:
				return []
	 
			pool = min(candidate_pool, len(indices))
			dense_ranked = self.dense_search(query, indices, pool)
			sparse_ranked = self.sparse_search(query, indices, pool)
	 
			rrf_scores: Dict[int, float] = {}
			for rank, idx in enumerate(dense_ranked):
				rrf_scores[idx] = rrf_scores.get(idx, 0.0) + 1.0 / (rrf_k + rank)
			for rank, idx in enumerate(sparse_ranked):
				rrf_scores[idx] = rrf_scores.get(idx, 0.0) + 1.0 / (rrf_k + rank)
	 
		
			rrf_ranked = sorted(rrf_scores.items(), key=lambda x: -x[1])
			pool_size = candidate_pool if use_reranker else top_k
			rrf_ranked = rrf_ranked[:pool_size]
	 
			results = []
			for idx, score in rrf_ranked:
				record = dict(self.chunks[idx])
				record["score"] = score
				results.append(record)
	 
			if use_reranker:
				return self.rerank(query, results, top_k)
	 
			return results[:top_k]




if __name__ == "__main__":
    import json
    from chunking import chunk_documents
 
    with open("data/sales_calls.json") as f:
        docs = json.load(f)
    chunks = chunk_documents(docs)
 
    index = HybridIndex(chunks)  # FIX 3: was SimpleIndex, now matches the class name above
 
    query = "why did customers cancel due to cost"
    print(f"=== Query: '{query}' ===")
    for r in index.hybrid_search(query, top_k=3):
        print(f"[{r.get('rerank_score', r['score']):.4f}] {r['doc_id']}: {r['text'][:100]}...")
 


