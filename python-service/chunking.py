import json
from typing import List, Dict, Any


def chunk_text(text: str, chunk_size: int = 120, overlap: int = 30) -> List[str]:

	# Flat list of individual words
	words = text.split()
	chunks = []
    
	# In case of very small documents
	if len(words) <= chunk_size:
    		return [text]

	step = chunk_size - overlap
	if step <= 0:
		raise ValueError("overlap cant be bigger than chunk size")

	start = 0
	while start < len(words):
		end = start + chunk_size
		
		# Appending this chunk to chunks but this time with blank space instead of new line or similar 
		chunk_words = words[start:end]
		chunks.append(" ".join(chunk_words))
		
		if end >= len(words):
			break
			
		# To ensure overlap!
		start += step
		
	return chunks


def chunk_documents(docs: List[Dict[str, Any]], chunk_size: int = 120, overlap: int = 30) -> List[Dict[str, Any]]:
	chunk_records = []

	for doc in docs:
		metadata = {k: v for k, v in doc.items() if k not in ("text", "id")}
		pieces = chunk_text(doc["text"], chunk_size=chunk_size, overlap = overlap)
		for i, piece in enumerate(pieces):
			chunk_records.append({
			# e.g. "call_001_0" for the first chunk of document call_001
			"chunk_id": f"{doc['id']}_{i}",
			# Keep track of the original document, so if I retrieve this chunk later, I know which source it came from.
			"doc_id": doc["id"],
			"text": piece,
			"metadata": metadata
			})

	return chunk_records




if __name__ == "__main__":
	with open("data/sales_calls.json") as f:
		docs = json.load(f)

	chunks = chunk_documents(docs)
	print(f"{len(docs)} documents -> {len(chunks)} chunks")
	print(json.dumps(chunks[0], indent=2))

