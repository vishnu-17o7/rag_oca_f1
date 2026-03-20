from dataclasses import dataclass, field
from typing import Optional
import asyncio

@dataclass
class BenchmarkState:
    is_running: bool = False
    current_iter: int = 0
    total_budget: int = 50
    best_fitness: float = 0.0
    best_params: dict = field(default_factory=lambda: {
        "chunk_size": 600,
        "chunk_overlap": 80,
        "temperature": 0.2,
        "top_k": 4
    })
    avg_fitness: float = 0.0
    invalid_count: int = 0
    cache_hits: int = 0
    history: list = field(default_factory=list)
    # history entries: {iter, chunk_size, chunk_overlap, temperature, top_k, fitness, status}
    elapsed_seconds: int = 0
    websocket_clients: list = field(default_factory=list)

benchmark_state = BenchmarkState()

# RAG pipeline cache — keyed by (chunk_size, chunk_overlap, temperature, top_k)
rag_cache: dict = {}

# Active websocket connections for benchmark
active_connections: list = []
