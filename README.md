# Hybrid RAG (Retrieval-Augmented Generation) Sales Calls

# Folder structure

hybrid-rag-sales-calls/
├── python-service/   
├── java-gateway/


## Python Service

### Chunking
Used sliding window overlap to reduce the chance of cutting important piece of information in half across a chunk boundary (overlap - 30, step = chunk_size - overlap)
 
Every chunk carries the chunk id, doc id, text, metadata (everything but id/text) - used for hard filters later

### Dense Retrieval
Used all-MiniLM-L6-v2 for embedding - a list of numbers represent what a text means (meaning, the synonyms will have similar numbers) , good for cases for example when someone asks “why did the customer cancel due to cost” and the transcript actually says “the price was too high”, it’s different words, same meaning.

### Sparse Retrieval
Used rank_bm25.BM25Okapi over whitespace lowercase tokens, regular keyword search.

### RRF (Fusion) - combining both searches
Each method has its blind spots, and each has a ranked list of most relevant candidates to answer the question, and we want to consider them both.
Reciprocal Rank Fusion - for each ranked list (dense and sparse) chunk at rank r contributes 1/(rrf_k + r) to a running score summed across lists.

No score normalization or learned weighting between dense/sparse signals — RRF was chosen specifically to avoid needing to calibrate BM25 scores (unbounded, corpus-dependent) against cosine similarities (bounded [-1,1]) on a common scale.

### Structure Filtering
Before both search methods, we call for hard filters: every call transcript has a customer name, a
date, and a deal stage (like “in negotiation” or “churned”) attached. If someone asks for discount
negotiations, we don't want the system loosely guessing based on wording, we can filter directly to
chunks tagged “in negotiation,” before any fuzzy searching happens at all. 

metadata_filter: Callable[[Dict], bool] applied via _apply_filter() before either dense or sparse search executes:  implemented as index-set restriction (list comprehension over chunk positions), not a post-hoc result filter. This guarantees filtered-out chunks never enter either ranking, which matters specifically because a chunk from an excluded customer could otherwise score arbitrarily high on pure text similarity. CLI exposes this as --stage (exact match) and --customer (case-insensitive substring match); both conditions are conjunctive (AND) when combined.

### Cross-Encoder Reranking
cross-encoder/ms-marco-MiniLM-L-6-v2, applied to the RRF-merged candidate pool as a second-stage precision pass (retrieve-then-rerank). 

Unlike the bi-encoder used for dense retrieval, the cross-encoder concatenates query and passage into a single forward pass, allowing cross-attention between them; this is more accurate but has no practical way to precompute or cache passage representations, hence restricting it to a small post-fusion shortlist.

### Generation
Model: claude-sonnet-4-6, max_tokens=500

Retrieved chunks are concatenated with source headers (doc_id/customer/date/deal_stage) and delimited, then passed as the user turn alongside a system prompt constraining the model to answer only from provided context and to explicitly flag insufficient context rather than extrapolate.

### Confidence-Gated Abstention
Pre-generation gate: if chunks is empty, or the top result's rerank_score (falling back to RRF score if
reranking was skipped) is below DEFAULT_CONFIDENCE_THRESHOLD = -2.0, the pipeline returns a fixed abstention message without invoking the LLM.

### Eval/Tests
Separate for chunks, retrieval

## Setup

```bash
cd path/to/rag_project

# Recommended - virtual environment
python3 -m venv venv
source venv/bin/activate 

pip install -r requirements.txt
export ANTHROPIC_API_KEY=your_key_here   # optional; omit to use offline fallback
```

## Usage

```bash
python3 main.py "why did halberg retail group churn?"
python3 main.py "discount negotiation" --stage in_negotiation
python3 main.py "onboarding issues" --customer Nordwind
python3 main.py "pricing pushback" --stage in_negotiation --customer Birkenstadt --top_k 5
```

## API (FastAPI)

[#api-fastapi](#api-fastapi)

Wraps the same pipeline (`chunk_documents`, `HybridIndex.hybrid_search`, `generate_answer`) behind a single endpoint. The index is built once at startup via a FastAPI lifespan event, not per-request.

### Setup

```
pip install fastapi uvicorn
```

(add `fastapi` and `uvicorn` to `requirements.txt`)

### Run

```
uvicorn api:app --reload
```

### Endpoint

`POST /query`

Request body:

```json
{
  "question": "why did halberg retail group churn?",
  "stage": "in_negotiation",
  "customer": "Birkenstadt",
  "top_k": 3
}
```

`stage` and `customer` are optional filters (same semantics as the CLI's `--stage`/`--customer`); `top_k` defaults to 3.

Response:

```json
{
  "answer": "...",
  "retrieved": [
    {
      "text": "...",
      "customer": "...",
      "deal_stage": "...",
      "score": 0.0123
    }
  ]
}
```

## Java Gateway

### Running locally

Requires JDK 17+ and Maven. Run from this folder (java-gateway/), with python-service/ already running in another terminal.

```bash
# from repo root:
cd java-gateway
export PYTHON_SERVICE_URL=http://localhost:8000
export GATEWAY_API_KEY=dev-local-key
mvn spring-boot:run
```

Test it:

```bash
curl -X POST http://localhost:8080/api/query \
  -H "Content-Type: application/json" \
  -H "X-API-Key: dev-local-key" \
  -d '{"question": "What objections came up in the Acme deal?", "top_k": 5}'
```

Health check (no API key needed): `GET /api/health`

## Config (env vars)

| Variable                 | Default                 | Purpose                                   |
|---------------------------|--------------------------|--------------------------------------------|
| `PORT`                    | `8080`                   | Port the gateway listens on                |
| `PYTHON_SERVICE_URL`      | `http://localhost:8000` | Base URL of the deployed Python service    |
| `PYTHON_SERVICE_TIMEOUT_MS` | `20000`                | Timeout waiting for Python's response      |
| `GATEWAY_API_KEY`         | `dev-local-key`         | Shared secret Android must send            |
| `CACHE_TTL_SECONDS`       | `300`                    | How long a cached answer is served         |
| `CACHE_MAX_SIZE`          | `500`                    | Max number of cached query results         |

## Endpoint contract (matched to the real Day 1 FastAPI service)

**Request** `POST {PYTHON_SERVICE_URL}/query`
```json
{"question": "...", "stage": "optional", "customer": "optional", "top_k": 3}
```

**Response**
```json
{
  "answer": "...",
  "retrieved": [
    {"text": "...", "customer": "...", "deal_stage": "...", "score": 0.83}
  ]
}
```

Java fields stay camelCase (`topK`, `dealStage`) internally but are mapped to
the Python service's snake_case JSON via `@JsonProperty("top_k")` /
`@JsonProperty("deal_stage")` in `QueryRequest`/`RetrievedChunk`, so no
manual renaming is needed on either side. `top_k` defaults to `3`, matching
the Python `Field(default=3, ge=1)`.

## Deploying

Package with `mvn clean package`, run the resulting jar from `target/`, or
point Render/Fly/Railway at this repo with a Java 17 buildpack. Set
`PYTHON_SERVICE_URL` to your deployed Python service's public URL and pick a
real `GATEWAY_API_KEY` (not the dev default) before deploying.

