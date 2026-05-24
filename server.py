import asyncio
import os
from dotenv import load_dotenv
from fastapi import FastAPI

load_dotenv()
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from api.routes_chat import router as chat_router
from api.routes_benchmark import router as benchmark_router
from api.state import rag_cache, rag_init_lock
from src.rag_pipeline import RAGPipeline
from contextlib import asynccontextmanager

DEFAULT_RAG_KEY = (700, 100, 0.0, 4)
DEFAULT_RAG_PARAMS = {
    "chunk_size": 700,
    "chunk_overlap": 100,
    "top_k": 4,
    "temperature": 0.0,
}


async def ensure_default_pipeline():
    """Create default pipeline once, off the event loop."""
    if DEFAULT_RAG_KEY in rag_cache:
        return

    async with rag_init_lock:
        if DEFAULT_RAG_KEY in rag_cache:
            return

        print("Initializing Default RAGPipeline...")
        rag_cache[DEFAULT_RAG_KEY] = await asyncio.to_thread(
            RAGPipeline, **DEFAULT_RAG_PARAMS
        )
        print("RAGPipeline initialized.")

@asynccontextmanager
async def lifespan(app: FastAPI):
    startup_mode = os.getenv("RAG_STARTUP_MODE", "blocking").strip().lower()

    if startup_mode in {"blocking", "sync", "eager"}:
        print("RAG startup mode: blocking preload")
        await ensure_default_pipeline()
    elif startup_mode in {"background", "bg"}:
        print("RAG startup mode: background warmup")
        app.state.rag_warmup_task = asyncio.create_task(ensure_default_pipeline())
    else:
        print("RAG startup mode: lazy (skip preload for fast server startup)")

    yield

    warmup_task = getattr(app.state, "rag_warmup_task", None)
    if warmup_task and not warmup_task.done():
        warmup_task.cancel()
        try:
            await warmup_task
        except asyncio.CancelledError:
            pass

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

# Run with: uvicorn server:app --port 8000
# Optional: RAG_STARTUP_MODE=background|blocking|lazy (default: lazy)
