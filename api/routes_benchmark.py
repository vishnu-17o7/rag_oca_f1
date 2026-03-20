from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
import asyncio, time, json
from api.state import benchmark_state, active_connections
from src.fitness import fitness_function

router = APIRouter()

# ── WebSocket Manager ──────────────────────────────────────────

class ConnectionManager:
    def __init__(self):
        self.connections: list[WebSocket] = []

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.connections.append(ws)

    def disconnect(self, ws: WebSocket):
        self.connections.remove(ws)

    async def broadcast(self, data: dict):
        """Send JSON to all connected benchmark clients."""
        dead = []
        for ws in self.connections:
            try:
                await ws.send_json(data)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.connections.remove(ws)

manager = ConnectionManager()

# ── WebSocket Endpoint ─────────────────────────────────────────

@router.websocket("/ws/benchmark")
async def benchmark_websocket(ws: WebSocket):
    """
    Persistent connection for the benchmark dashboard.
    Receives commands: { "action": "start", "budget": 50 }
    Sends updates:     see WEBSOCKET MESSAGE SPEC below
    """
    await manager.connect(ws)
    try:
        while True:
            data = await ws.receive_json()
            if data.get("action") == "start":
                if not benchmark_state.is_running:
                    asyncio.create_task(run_optimization(data.get("budget", 50)))
            elif data.get("action") == "stop":
                benchmark_state.is_running = False
    except WebSocketDisconnect:
        manager.disconnect(ws)

# ── Optimization Runner ────────────────────────────────────────

async def run_optimization(budget: int = 50):
    """
    Runs OCA optimization in a background async task.
    After each fitness evaluation, broadcasts a state update to all
    connected WebSocket clients.
    """
    from research.oca import OverclockingAlgorithm
    import numpy as np
    
    benchmark_state.is_running = True
    benchmark_state.current_iter = 0
    benchmark_state.history = []
    benchmark_state.best_fitness = 0.0
    benchmark_state.invalid_count = 0
    benchmark_state.cache_hits = 0
    start_time = time.time()

    BOUNDS = [
        (100, 1000),  # chunk_size
        (0,   200),   # chunk_overlap
        (0.0, 1.0),   # temperature
        (1,   10),    # top_k
    ]

    call_count = [0]  # mutable container for closure

    def instrumented_fitness(params):
        """
        Wraps fitness_function with:
        1. Call count tracking
        2. Cache hit detection
        3. State update + broadcast after each eval
        """
        from src.fitness import _cache
        
        chunk_size    = int(round(params[0]))
        chunk_overlap = int(round(params[1]))
        temperature   = round(float(params[2]), 2)
        top_k         = int(round(params[3]))
        
        cache_key = (chunk_size, chunk_overlap, temperature, top_k)
        was_cached = getattr(_cache, 'get', lambda k: None)(cache_key) is not None if hasattr(_cache, 'get') else cache_key in _cache
        if type(_cache) == dict and cache_key in _cache:
            was_cached = True
            
        score = fitness_function(params)
        call_count[0] += 1
        
        if was_cached:
            benchmark_state.cache_hits += 1
        
        if chunk_overlap >= chunk_size:
            benchmark_state.invalid_count += 1
            status = "PENALTY"
        else:
            status = "OK"
        
        # Update state
        benchmark_state.current_iter = call_count[0]
        benchmark_state.elapsed_seconds = int(time.time() - start_time)
        
        if score > benchmark_state.best_fitness:
            benchmark_state.best_fitness = score
            benchmark_state.best_params = {
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap,
                "temperature": temperature,
                "top_k": top_k
            }
        
        all_scores = [h["fitness"] for h in benchmark_state.history if h["status"] == "OK"]
        benchmark_state.avg_fitness = sum(all_scores) / len(all_scores) if all_scores else 0.0
        
        entry = {
            "iter": call_count[0],
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "temperature": temperature,
            "top_k": top_k,
            "fitness": round(score, 4),
            "status": status,
            "cached": was_cached
        }
        benchmark_state.history.append(entry)
        
        # Broadcast to all WebSocket clients
        loop = asyncio.get_event_loop()
        loop.call_soon_threadsafe(
            asyncio.ensure_future,
            manager.broadcast(build_update_payload())
        )
        
        # Handle early stop if user pressed stop button
        if not benchmark_state.is_running:
            raise StopIteration("Optimization stopped by user")
            
        return score

    # Wrapper logic from main.py
    def _clip_to_bounds(position: np.ndarray, dim: int, bounds: list) -> np.ndarray:
        clipped = np.zeros(dim)
        for i, (lower, upper) in enumerate(bounds):
            clipped[i] = np.clip(position[i], lower, upper)
        return clipped

    def _objective_wrapper(position: np.ndarray) -> float:
        clipped = _clip_to_bounds(position, len(BOUNDS), BOUNDS)
        return instrumented_fitness(clipped.tolist())

    def run_oca_sync():
        dim = len(BOUNDS)
        max_evals = budget
        pop_size = 10
        num_p_cores = 3
        max_iterations = max(1, max_evals // pop_size)
        
        oca = OverclockingAlgorithm(
            pop_size=pop_size,
            num_p_cores=num_p_cores,
            initial_voltage=2.0,
            final_voltage=0.0,
            aggressive_voltage=True,
        )
        
        all_lower = min(b[0] for b in BOUNDS)
        all_upper = max(b[1] for b in BOUNDS)
        
        oca.optimize(
            objective_fn=_objective_wrapper,
            bounds=(all_lower, all_upper),
            dim=dim,
            max_iterations=max_iterations,
        )
        
        # Fill remaining budget
        remaining = max_evals - call_count[0]
        while remaining > 0 and call_count[0] < max_evals and benchmark_state.is_running:
            for _ in range(min(remaining, pop_size)):
                random_pos = np.array([np.random.uniform(b[0], b[1]) for b in BOUNDS])
                _objective_wrapper(random_pos)
                remaining = max_evals - call_count[0]
                if remaining <= 0 or not benchmark_state.is_running:
                    break

    # Run OCA in executor to avoid blocking the event loop
    loop = asyncio.get_event_loop()
    try:
        await loop.run_in_executor(
            None,
            run_oca_sync
        )
    except StopIteration:
        pass  # Expected if user stops
    except Exception as e:
        print(f"OCA execution error: {e}")
    finally:
        benchmark_state.is_running = False
        await manager.broadcast({"type": "complete", **build_update_payload()})


def build_update_payload() -> dict:
    """Builds the full state snapshot sent to the browser after each iteration."""
    s = benchmark_state
    return {
        "type": "update",
        "is_running": s.is_running,
        "current_iter": s.current_iter,
        "total_budget": s.total_budget,
        "best_fitness": round(s.best_fitness, 4),
        "best_params": s.best_params,
        "avg_fitness": round(s.avg_fitness, 4),
        "invalid_count": s.invalid_count,
        "cache_hits": s.cache_hits,
        "elapsed_seconds": s.elapsed_seconds,
        "history": s.history,   # full list — browser diffs this
    }


# ── REST Fallback ──────────────────────────────────────────────

@router.get("/benchmark/state")
async def get_benchmark_state():
    """HTTP fallback — returns current benchmark state as JSON."""
    return build_update_payload()

@router.post("/benchmark/start")
async def start_benchmark(budget: int = 50):
    """HTTP trigger for starting a run (fallback if WS not available)."""
    if not benchmark_state.is_running:
        asyncio.create_task(run_optimization(budget))
    return {"started": True}
