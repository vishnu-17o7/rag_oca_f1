import asyncio
import os
import time
from dotenv import load_dotenv
from fastapi import FastAPI, Request

load_dotenv()
print(f"[{time.strftime('%H:%M:%S')}] [SERVER] .env loaded")

from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from api.routes_chat import router as chat_router
from api.routes_benchmark import router as benchmark_router
from api.state import rag_cache, rag_init_lock, limiter
from src.rag_pipeline import RAGPipeline
from contextlib import asynccontextmanager
from slowapi.errors import RateLimitExceeded
from slowapi import _rate_limit_exceeded_handler
from slowapi.middleware import LimiterMiddleware

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

# Rate limiter
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(LimiterMiddleware, limiter=limiter)

# Quick health endpoint (always responds immediately, even before pipeline is ready)
@app.get("/health")
@limiter.exempt
async def health(request: Request):
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
@limiter.exempt
async def serve_chat(request: Request):
    return FileResponse("static/chat.html")
print(f"[{time.strftime('%H:%M:%S')}] [ROUTE] GET / -> static/chat.html")

@app.get("/bench")
@limiter.exempt
async def serve_benchmark(request: Request):
    return FileResponse("static/benchmark.html")
print(f"[{time.strftime('%H:%M:%S')}] [ROUTE] GET /bench -> static/benchmark.html")

@app.get("/healthz")
@limiter.exempt
async def healthz(request: Request):
    return {"ok": True}
print(f"[{time.strftime('%H:%M:%S')}] [ROUTE] GET /healthz -> 200 (liveness probe)")

print(f"[{time.strftime('%H:%M:%S')}] [SERVER] All routes registered. Handing over to uvicorn...")

# Run with: uvicorn server:app --port 8000
# Optional: RAG_STARTUP_MODE=background|blocking|lazy (default: lazy)
