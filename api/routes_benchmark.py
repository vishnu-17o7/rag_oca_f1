from fastapi import APIRouter, WebSocket, WebSocketDisconnect
import asyncio, time, json, os
from datetime import datetime
from api.state import benchmark_state

router = APIRouter()

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")


def save_benchmark_results():
    """Save benchmark results to a timestamped JSON file."""
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"benchmark_{timestamp}.json"
    filepath = os.path.join(RESULTS_DIR, filename)

    result_data = {
        "timestamp": timestamp,
        "best_fitness": benchmark_state.best_fitness,
        "best_params": benchmark_state.best_params,
        "avg_fitness": benchmark_state.avg_fitness,
        "total_iterations": benchmark_state.current_iter,
        "invalid_count": benchmark_state.invalid_count,
        "cache_hits": benchmark_state.cache_hits,
        "elapsed_seconds": benchmark_state.elapsed_seconds,
        "history": benchmark_state.history,
    }

    with open(filepath, "w") as f:
        json.dump(result_data, f, indent=2)

    return filename


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


class OptimizationTerminated(Exception):
    """Raised for expected benchmark termination paths."""


def normalize_budget(value: object, default: int = 100) -> int:
    """Clamp requested iteration budget to a safe range."""
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    return max(1, min(parsed, 1000))


# ── WebSocket Endpoint ─────────────────────────────────────────

@router.websocket("/ws/benchmark")
async def benchmark_websocket(ws: WebSocket):
    """
    Persistent connection for the benchmark dashboard.
    Receives commands: { "action": "start", "budget": <n> }
    Sends updates:     see WEBSOCKET MESSAGE SPEC below
    """
    await manager.connect(ws)
    try:
        while True:
            data = await ws.receive_json()
            if data.get("action") == "start":
                if not benchmark_state.is_running:
                    budget = normalize_budget(data.get("budget", 100))
                    asyncio.create_task(run_optimization(budget))
            elif data.get("action") == "stop":
                benchmark_state.is_running = False
                await manager.broadcast(build_update_payload())
    except WebSocketDisconnect:
        manager.disconnect(ws)


# ── Optimization Runner ────────────────────────────────────────

async def run_optimization(budget: int = 100):
    """
    Runs OCA optimization in a background async task.
    After each fitness evaluation, broadcasts a state update to all
    connected WebSocket clients.
    """
    from research.oca import OverclockingAlgorithm
    import numpy as np
    from src.fitness import fitness_function, _evaluation_cache

    # Capture the active event loop once; fitness callbacks run in a worker
    # thread and must hand async work back to this loop thread-safely.
    loop = asyncio.get_running_loop()

    max_evals = normalize_budget(budget)

    benchmark_state.is_running = True
    benchmark_state.total_budget = max_evals
    benchmark_state.current_iter = 0
    benchmark_state.history = []
    benchmark_state.best_fitness = 0.0
    benchmark_state.best_params = None
    benchmark_state.avg_fitness = 0.0
    benchmark_state.invalid_count = 0
    benchmark_state.cache_hits = 0
    benchmark_state.elapsed_seconds = 0
    start_time = time.time()
    await manager.broadcast(build_update_payload())

    BOUNDS = [
        (100, 1000),  # chunk_size
        (0, 200),  # chunk_overlap
        (0.0, 1.0),  # temperature
        (1, 10),  # top_k
    ]

    call_count = [0]  # mutable container for closure

    def instrumented_fitness(params):
        """
        Wraps fitness_function with:
        1. Call count tracking
        2. Cache hit detection
        3. State update + broadcast after each eval
        """
        chunk_size = int(round(params[0]))
        chunk_overlap = int(round(params[1]))
        temperature = round(float(params[2]), 2)
        top_k = int(round(params[3]))

        cache_key = (chunk_size, chunk_overlap, temperature, top_k)
        was_cached = cache_key in _evaluation_cache

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

        if status == "OK" and (
            benchmark_state.best_params is None or score >= benchmark_state.best_fitness
        ):
            benchmark_state.best_fitness = score
            benchmark_state.best_params = {
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap,
                "temperature": temperature,
                "top_k": top_k,
            }

        entry = {
            "iter": call_count[0],
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "temperature": temperature,
            "top_k": top_k,
            "fitness": round(score, 4),
            "status": status,
            "cached": was_cached,
        }
        benchmark_state.history.append(entry)

        all_scores = [h["fitness"] for h in benchmark_state.history if h["status"] == "OK"]
        benchmark_state.avg_fitness = sum(all_scores) / len(all_scores) if all_scores else 0.0

        # Broadcast state updates from worker thread to main asyncio loop.
        asyncio.run_coroutine_threadsafe(
            manager.broadcast(build_update_payload()),
            loop,
        )

        if call_count[0] >= max_evals:
            raise OptimizationTerminated(f"Reached max iterations ({max_evals})")

        # Handle early stop if user pressed stop button
        if not benchmark_state.is_running:
            raise OptimizationTerminated("Optimization stopped by user")

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
    try:
        await loop.run_in_executor(
            None,
            run_oca_sync,
        )
    except OptimizationTerminated:
        pass  # Expected if user stops
    except Exception as e:
        print(f"OCA execution error: {e}")
    finally:
        benchmark_state.is_running = False
        benchmark_state.elapsed_seconds = int(time.time() - start_time)
        saved_file = save_benchmark_results()
        await manager.broadcast(
            {"type": "complete", **build_update_payload(), "saved_file": saved_file}
        )


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
        "history": s.history,  # full list — browser diffs this
    }


# ── REST Fallback ──────────────────────────────────────────────

@router.get("/benchmark/state")
async def get_benchmark_state():
    """HTTP fallback — returns current benchmark state as JSON."""
    return build_update_payload()


@router.post("/benchmark/start")
async def start_benchmark(budget: int = 100):
    """HTTP trigger for starting a run (fallback if WS not available)."""
    if not benchmark_state.is_running:
        asyncio.create_task(run_optimization(normalize_budget(budget)))
    return {"started": True}


@router.post("/benchmark/stop")
async def stop_benchmark():
    """HTTP trigger for stopping an active run (fallback if WS not available)."""
    benchmark_state.is_running = False
    await manager.broadcast(build_update_payload())
    return {"stopped": True}
