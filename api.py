import json
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from chunking import chunk_documents
from retrieval import HybridIndex
from generate import generate_answer

DATA_PATH = "data/sales_calls.json"


@asynccontextmanager
async def lifespan(app: FastAPI):
    with open(DATA_PATH) as f:
        docs = json.load(f)
    chunks = chunk_documents(docs)
    app.state.index = HybridIndex(chunks)
    yield
    app.state.index = None


app = FastAPI(title="Hybrid RAG Sales Calls API", lifespan=lifespan)


class QueryRequest(BaseModel):
    question: str
    stage: Optional[str] = None
    customer: Optional[str] = None
    top_k: int = Field(default=3, ge=1)


class RetrievedChunk(BaseModel):
    text: str
    customer: str
    deal_stage: str
    score: float


class QueryResponse(BaseModel):
    answer: str
    retrieved: List[RetrievedChunk]


def _build_metadata_filter(stage: Optional[str], customer: Optional[str]):
    if not stage and not customer:
        return None

    def metadata_filter(meta: Dict[str, Any], stage=stage, customer=customer) -> bool:
        if stage and meta.get("deal_stage") != stage:
            return False
        if customer and customer.lower() not in meta.get("customer", "").lower():
            return False
        return True

    return metadata_filter


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest) -> QueryResponse:
    index: HybridIndex = app.state.index
    if index is None:
        raise HTTPException(status_code=503, detail="Index is not ready")

    metadata_filter = _build_metadata_filter(req.stage, req.customer)

    retrieved = index.hybrid_search(
        req.question,
        top_k=req.top_k,
        metadata_filter=metadata_filter,
    )

    answer = generate_answer(req.question, retrieved)

    retrieved_out = [
        RetrievedChunk(
            text=r["text"],
            customer=r["metadata"].get("customer", "?"),
            deal_stage=r["metadata"].get("deal_stage", "?"),
            score=r.get("rerank_score", r["score"]),
        )
        for r in retrieved
    ]

    return QueryResponse(answer=answer, retrieved=retrieved_out)
