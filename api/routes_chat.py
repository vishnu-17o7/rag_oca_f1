from fastapi import APIRouter
from pydantic import BaseModel
from src.rag_pipeline import RAGPipeline
from api.state import rag_cache

router = APIRouter()

class QueryRequest(BaseModel):
    question: str
    chunk_size: int = 600
    chunk_overlap: int = 80
    temperature: float = 0.2
    top_k: int = 4

class QueryResponse(BaseModel):
    answer: str
    params_used: dict
    chunks_retrieved: list  # [{text: str, score: float, source: str}]

@router.post("/query", response_model=QueryResponse)
async def query_rag(req: QueryRequest):
    """
    Instantiate (or retrieve cached) RAGPipeline with given params.
    Call .query(question), return answer + retrieved chunks.
    
    For chunks_retrieved: modify RAGPipeline if needed to also return
    the source chunks alongside the answer. If RAGPipeline only returns
    a string, wrap the retrieval step to also capture the docs.
    Return max 3 chunks, each truncated to 200 characters.
    """
    cache_key = (req.chunk_size, req.chunk_overlap, req.temperature, req.top_k)
    
    if cache_key not in rag_cache:
        rag_cache[cache_key] = RAGPipeline(
            req.chunk_size, req.chunk_overlap, req.top_k, req.temperature
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
        chunks_retrieved=chunks[:3]
    )

@router.get("/health")
async def health():
    """Returns Ollama status and whether PDFs are loaded."""
    import glob, os
    pdfs = glob.glob("data/*.pdf")
    return {
        "status": "ok",
        "pdfs_loaded": len(pdfs),
        "pdf_names": [os.path.basename(p) for p in pdfs]
    }
