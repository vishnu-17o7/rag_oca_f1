import asyncio
import time
from fastapi import APIRouter, Request

print(f"[{time.strftime('%H:%M:%S')}] [MODULE] api/routes_chat.py loaded")
from pydantic import BaseModel
from src.rag_pipeline import RAGPipeline
from api.state import rag_cache, rag_init_lock, limiter

router = APIRouter()

class QueryRequest(BaseModel):
    question: str
    chunk_size: int = 700
    chunk_overlap: int = 100
    temperature: float = 0.0
    top_k: int = 4

class QueryResponse(BaseModel):
    answer: str
    params_used: dict
    chunks_retrieved: list  # [{text: str, score: float, source: str}]

@router.post("/query", response_model=QueryResponse)
@limiter.limit("5/minute")
async def query_rag(request: Request, req: QueryRequest):
    """
    Instantiate (or retrieve cached) RAGPipeline with given params.
    Call .query(question), return answer + retrieved chunks.
    
    For chunks_retrieved: modify RAGPipeline if needed to also return
    the source chunks alongside the answer. If RAGPipeline only returns
    a string, wrap the retrieval step to also capture the docs.
    Return retrieved chunks from the latest query for UI rendering.
    """
    cache_key = (req.chunk_size, req.chunk_overlap, req.temperature, req.top_k)
    
    if cache_key not in rag_cache:
        async with rag_init_lock:
            if cache_key not in rag_cache:
                print(f"[CACHE MISS] Building RAGPipeline for key={cache_key} ...")
                rag_cache[cache_key] = await asyncio.to_thread(
                    RAGPipeline,
                    req.chunk_size,
                    req.chunk_overlap,
                    req.top_k,
                    req.temperature,
                )
    
    pipeline = rag_cache[cache_key]
    
    # Query and return
    answer = pipeline.query(req.question)
    
    # Get retrieved chunks — adapt to actual RAGPipeline implementation
    chunks = pipeline.get_last_chunks() if hasattr(pipeline, 'get_last_chunks') else []
    
    return QueryResponse(
        answer=answer,
        params_used={
            "chunk_size": req.chunk_size,
            "chunk_overlap": req.chunk_overlap,
            "temperature": req.temperature,
            "top_k": req.top_k
        },
        chunks_retrieved=chunks
    )

@router.get("/health")
@limiter.exempt
async def health(request: Request):
    """Returns server status, PDF count, and pipeline stats."""
    import glob, os
    pdfs = glob.glob("data/*.pdf")

    # Grab chunk count from the default cached pipeline if available
    default_key = (700, 100, 0.0, 4)
    pipeline = rag_cache.get(default_key)
    chunk_count = len(pipeline.chunks) if pipeline and hasattr(pipeline, "chunks") and pipeline.chunks else 0

    return {
        "status": "ok",
        "pdfs_loaded": len(pdfs),
        "pdf_names": sorted(os.path.basename(p) for p in pdfs),
        "chunk_count": chunk_count,
    }
