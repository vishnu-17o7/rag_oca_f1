import asyncio
import os
import time
from dotenv import load_dotenv
from fastapi import FastAPI

load_dotenv()
print(f"[{time.strftime('%H:%M:%S')}] [SERVER] .env loaded")

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

        print(f"[{time.strftime('%H:%M:%S')}] [PIPELINE] Initializing Default RAGPipeline (blocking)...")
        t0 = time.time()
        rag_cache[DEFAULT_RAG_KEY] = await asyncio.to_thread(
            RAGPipeline, **DEFAULT_RAG_PARAMS
        )
        print(f"[{time.strftime('%H:%M:%S')}] [PIPELINE] RAGPipeline initialized in {time.time() - t0:.0f}s")

@asynccontextmanager
async def lifespan(app: FastAPI):
    startup_mode = os.getenv("RAG_STARTUP_MODE", "blocking").strip().lower()
    print(f"[{time.strftime('%H:%M:%S')}] [LIFESPAN] Startup mode: {startup_mode}")

    if startup_mode in {"blocking", "sync", "eager"}:
        print(f"[{time.strftime('%H:%M:%S')}] [LIFESPAN] BLOCKING: will not accept requests until pipeline is ready")
        await ensure_default_pipeline()
    elif startup_mode in {"background", "bg"}:
        print(f"[{time.strftime('%H:%M:%S')}] [LIFESPAN] BACKGROUND: server accepts requests immediately, pipeline warms up in background")
        app.state.rag_warmup_task = asyncio.create_task(ensure_default_pipeline())
    else:
        print(f"[{time.strftime('%H:%M:%S')}] [LIFESPAN] LAZY: server accepts requests immediately, pipeline builds on first query")

    print(f"[{time.strftime('%H:%M:%S')}] [LIFESPAN] Startup complete — server is now accepting requests")
    yield

    warmup_task = getattr(app.state, "rag_warmup_task", None)
    if warmup_task and not warmup_task.done():
        warmup_task.cancel()
        try:
            await warmup_task
        except asyncio.CancelledError:
            pass

    rag_cache.clear()
    print(f"[{time.strftime('%H:%M:%S')}] [LIFESPAN] Shutdown complete")

app = FastAPI(title="F1 REG / RAG", docs_url=None, redoc_url=None, lifespan=lifespan)

# Quick health endpoint (always responds immediately, even before pipeline is ready)
@app.get("/health")
async def health():
    return {"status": "ok", "pipeline_ready": DEFAULT_RAG_KEY in rag_cache}

# Mount static files
print(f"[{time.strftime('%H:%M:%S')}] [ROUTE] Mounting /static -> static/")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Register routers
print(f"[{time.strftime('%H:%M:%S')}] [ROUTE] Registering /api/* (chat)")
app.include_router(chat_router, prefix="/api")
print(f"[{time.strftime('%H:%M:%S')}] [ROUTE] Registering /api/* (benchmark)")
app.include_router(benchmark_router, prefix="/api")

# Serve HTML pages at clean routes
@app.get("/")
async def serve_chat():
    return FileResponse("static/chat.html")
print(f"[{time.strftime('%H:%M:%S')}] [ROUTE] GET / -> static/chat.html")

@app.get("/bench")
async def serve_benchmark():
    return FileResponse("static/benchmark.html")
print(f"[{time.strftime('%H:%M:%S')}] [ROUTE] GET /bench -> static/benchmark.html")

@app.get("/healthz")
def healthz():
    return {"ok": True}
print(f"[{time.strftime('%H:%M:%S')}] [ROUTE] GET /healthz -> 200 (liveness probe)")

print(f"[{time.strftime('%H:%M:%S')}] [SERVER] All routes registered. Handing over to uvicorn...")

# Run with: uvicorn server:app --port 8000
# Optional: RAG_STARTUP_MODE=background|blocking|lazy (default: lazy)
