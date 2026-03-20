from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from api.routes_chat import router as chat_router
from api.routes_benchmark import router as benchmark_router
from api.state import rag_cache
from src.rag_pipeline import RAGPipeline
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Initializing Default RAGPipeline on startup...")
    # Initialize the default pipeline configuration so vector DB chunks are pre-generated
    rag_cache[(600, 80, 0.2, 4)] = RAGPipeline(
        chunk_size=600, chunk_overlap=80, top_k=4, temperature=0.2
    )
    print("RAGPipeline initialized.")
    yield
    # Cleanup on shutdown if needed
    rag_cache.clear()

app = FastAPI(title="F1 REG / RAG", docs_url=None, redoc_url=None, lifespan=lifespan)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Register routers
app.include_router(chat_router, prefix="/api")
app.include_router(benchmark_router, prefix="/api")

# Serve HTML pages at clean routes
@app.get("/")
async def serve_chat():
    return FileResponse("static/chat.html")

@app.get("/bench")
async def serve_benchmark():
    return FileResponse("static/benchmark.html")

# Run with: uvicorn server:app --reload --port 8000
