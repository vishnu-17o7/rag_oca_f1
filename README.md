# RAG Hyperparameter Optimization with OCA

Optimizing RAG Pipeline Hyperparameters for F1/FIA Regulatory Documents using the Overclocking Algorithm

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-WebSocket-009688?logo=fastapi&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-ChromaDB-FF6F00?logo=langchain&logoColor=white)
![Ollama](https://img.shields.io/badge/Ollama-FF6F00?logo=ollama&logoColor=white)

---

## Table of Contents

1. [Overview](#overview)
2. [Features](#features)
3. [Architecture](#architecture)
4. [Project Structure](#project-structure)
5. [Installation](#installation)
6. [Usage](#usage)
7. [API Reference](#api-reference)
8. [Hyperparameters](#hyperparameters)
9. [Results & Output](#results--output)
10. [Technology Stack](#technology-stack)
11. [License](#license)

---

## Overview

This project implements an automated hyperparameter optimization system for Retrieval-Augmented Generation (RAG) pipelines, specifically tuned for **FIA Formula 1 and F1 regulatory documents**. It uses the **Overclocking Algorithm (OCA)** — a population-based metaheuristic inspired by CPU overclocking — to efficiently search the hyperparameter space and find optimal configurations.

The system evaluates RAG configurations by comparing generated answers against a gold standard set of Q&A pairs using **semantic similarity**, enabling automatic discovery of the best chunk size, overlap, temperature, and retrieval parameters.

### Why This Project?

- **Domain-Specific**: Built specifically for legal/regulatory RAG use cases (F1 regulations)
- **Automated**: No manual tuning — OCA discovers optimal hyperparameters automatically
- **Real-Time Monitoring**: WebSocket-powered live dashboard shows convergence as it happens
- **Efficient**: Caching eliminates redundant LLM calls, saving time and API costs

---

## Features

| Feature | Description |
|---------|------------|
| **OCA Optimizer** | Population-based metaheuristic with adaptive voltage for exploration/exploitation |
| **Semantic Fitness** | Evaluates RAG output quality using sentence transformer similarity |
| **WebSocket Dashboard** | Real-time convergence chart, parameter space visualization, iteration log |
| **Result Persistence** | Saves all benchmark runs to JSON for reproducibility |
| **Vector Store Caching** | Reuses ChromaDB vector stores to avoid redundant indexing |
| **Evaluation Caching** | Memoization of fitness calls to minimize LLM queries |
| **REST + WebSocket API** | HTTP fallback + persistent WebSocket for live updates |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Client (Browser)                       │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │ Chat UI     │  │Benchmark UI   │  │ Results UI  │  │
│  │(chat.html) │  │(benchmark)   │  │ (JSON)     │  │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  │
└─────────┼─────────────────┼─────────────────┼──────────┘
          │                 │                 │
          ▼                 ▼                 ▼
┌─────────────────────────────────────────────────────────────┐
│                     FastAPI Server                        │
│  ┌──────────────────────────────────────────────────┐    │
│  │                API Routes                      │    │
│  │  /api/query (REST)  /api/bench (REST)  /api/ws │    │
│  └──────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
          │                                                 │
          ▼                                                 ▼
┌─────────────────────────────────────────────────────────────┐
│                  Core Components                            │
│  ┌────────────────┐  ┌────────────────┐  ┌──────────┐ │
│  │ RAG Pipeline  │  │ OCA Optimizer │  │ Fitness │ │
│  │(LangChain+   │  │(Metaheuristic)│  │ (Sim)  │ │
│  │ ChromaDB+   │  │              │  │        │ │
│  │ Ollama)      │  └──────────────┘  └──────────┘ │
│  └────────────────┘                                   │
└─────────────────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────────┐
│                 External Services                        │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐          │
│  │ Ollama  │  │ ChromaDB │  │  PDFs   │          │
│  │(LLM)   │  │(Vector) │  │(Data)   │          │
│  └────────┘  └────────┘  └────────┘          │
└─────────────────────────────────────────────────────────────┘
```

### Data Flow

1. **Benchmark Start**: User sets budget (max LLM calls) via WebSocket or HTTP
2. **OCA Iteration**: OCA generates candidate hyperparameters
3. **Fitness Eval**: For each candidate:
   - RAG pipeline is instantiated with candidate params
   - Pipeline answers gold standard Q&A (8 pairs)
   - Semantic similarity computed against ground truth
4. **State Broadcast**: Results sent to all WebSocket clients in real-time
5. **Completion**: Final results saved to JSON

---

## Project Structure

```
rag_oca_f1/
├── api/                      # FastAPI route handlers
│   ├── routes_benchmark.py   # /api/benchmark/* endpoints
│   ├── routes_chat.py      # /api/query endpoint
│   └── state.py           # Shared benchmark state
├── data/                    # Input PDFs (F1/FIA regulations)
│   ├── general.pdf
│   └── sportif.pdf
├── research/                 # Algorithm implementations
│   └── oca.py             # Overclocking Algorithm (OCA)
├── results/                # Output directory
│   ├── benchmark_*.json   # Benchmark run results
│   └── convergence_plot.png # Convergence visualization
├── src/                    # Core RAG components
│   ├── rag_pipeline.py     # RAG pipeline implementation
│   ├── fitness.py        # Fitness function (semantic similarity)
│   ├── gold_standard.json # 8 Q&A gold standard pairs
│   └── visualize.py     # Convergence plotting
├── static/                 # Frontend assets
│   ├── benchmark.html    # Real-time dashboard
│   └── chat.html        # Chat interface
├── server.py             # FastAPI server entry point
├── main.py               # CLI entry point (deprecated)
└── requirements.txt      # Python dependencies
```

---

## Installation

### Prerequisites

| Requirement | Version | Notes |
|------------|---------|-------|
| **Python** | 3.10+ | Required |
| **Ollama** | Latest | Must have model pulled |
| **Disk Space** | ~2GB | For models + vector stores |

### 1. Clone and Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/rag_oca_f1.git
cd rag_oca_f1

# Create virtual environment
python -m venv .venv

# Activate (Linux/macOS)
source .venv/bin/activate

# Activate (Windows)
.venv\Scripts\activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Pull Ollama Model

```bash
# Option 1: TinyLlama (fastest, ~1GB)
ollama pull tinyllama

# Option 2: Phi-3 (better quality, ~2GB)
ollama pull phi3

# Option 3: Custom (set via OLLAMA_MODEL_PRIORITY)
ollama pull qwen3.5:0.8b
```

### 4. Add PDF Documents

Place your F1/FIA PDF regulations in the `data/` directory:

```
data/
├── general.pdf    # FIA General Regulations
└── sportif.pdf   # FIA Sporting Regulations
```

---

## Usage

### Starting the Server

```bash
# Start with default settings
uvicorn server:app --port 8000 --reload

# Or with background RAG warmup (faster startup)
RAG_STARTUP_MODE=background uvicorn server:app --port 8000 --reload
```

### Accessing the Interface

| Route | Description |
|-------|-----------|
| `http://localhost:8000/` | Chat interface |
| `http://localhost:8000/bench` | Benchmark dashboard |

### Running a Benchmark

1. Open `http://localhost:8000/bench`
2. Click the **play** button
3. Set the budget (e.g., 50 LLM calls)
4. Watch convergence in real-time

### CLI Usage (Deprecated)

```bash
# Run optimization via CLI
python main.py
```

---

## API Reference

### REST Endpoints

| Endpoint | Method | Description |
|----------|--------|------------|
| `/api/query` | POST | Query the RAG pipeline |
| `/api/benchmark/state` | GET | Get benchmark state |
| `/api/benchmark/start` | POST | Start benchmark run |
| `/api/benchmark/stop` | POST | Stop benchmark run |
| `/api/health` | GET | Check system health |

### WebSocket Endpoint

| Endpoint | Description |
|----------|------------|
| `/api/ws/benchmark` | Real-time benchmark updates |

#### WebSocket Messages

**Client → Server:**
```json
{ "action": "start", "budget": 50 }
{ "action": "stop" }
```

**Server → Client:**
```json
{
  "type": "update",
  "is_running": true,
  "current_iter": 25,
  "best_fitness": 0.8523,
  "best_params": {
    "chunk_size": 600,
    "chunk_overlap": 80,
    "temperature": 0.2,
    "top_k": 4
  },
  "history": [...]
}
```

---

## Hyperparameters

The optimizer searches the following 4-dimensional space:

| Parameter | Range | Default | Description |
|-----------|-------|---------|-------------|
| `chunk_size` | 100–1000 | 600 | Characters per text chunk |
| `chunk_overlap` | 0–200 | 80 | Overlap between consecutive chunks |
| `temperature` | 0.0–1.0 | 0.2 | LLM generation temperature |
| `top_k` | 1–10 | 4 | Number of documents to retrieve |

### Constraints

- **Penalty**: Configurations where `chunk_overlap >= chunk_size` are invalid and receive `fitness = 0.0`
- **Budget**: Default 50 LLM calls (configurable up to 1000)

### Search Space Visualization

```
chunk_size    : ████████░░░░░░░░░░░░ 100 ────────── 1000
chunk_overlap: ████░░░░░░░░░░░░░░░░ 0 ────────── 200
temperature : ████████░░░░░░░░░░░░░ 0.0 ──────── 1.0
top_k       : ████░░░░░░░░░░░░░░░░░ 1 ────────── 10
```

---

## Results & Output

### JSON Output Schema

Each benchmark run produces a file like `results/benchmark_20260404_154603.json`:

```json
{
  "timestamp": "20260404_154603",
  "best_fitness": 0.8523,
  "best_params": {
    "chunk_size": 600,
    "chunk_overlap": 80,
    "temperature": 0.2,
    "top_k": 4
  },
  "avg_fitness": 0.7845,
  "total_iterations": 50,
  "invalid_count": 3,
  "cache_hits": 12,
  "elapsed_seconds": 847,
  "history": [
    {
      "iter": 1,
      "chunk_size": 450,
      "chunk_overlap": 75,
      "temperature": 0.15,
      "top_k": 3,
      "fitness": 0.7234,
      "status": "OK",
      "cached": false
    },
    ...
  ]
}
```

### Convergence Plot

![Convergence Plot](results/convergence_plot.png)

---

## Gold Standard Q&A

The fitness function evaluates against 8 domain-specific Q&A pairs:

| # | Question | Answer (Gold Standard) |
|----|---------|-------------------|
| 1 | What is the minimum weight of the car including the driver? | 798 kg |
| 2 | How many engines is each driver allowed per season? | 4 engines |
| 3 | What is the penalty for using an additional power unit element? | 10-place grid penalty |
| 4 | What is the maximum race distance? | 305 km (260 km at Monaco) |
| 5 | How long before the race must the starting grid be published? | 4 hours |
| 6 | Under what conditions can a Safety Car be deployed? | When competitors/officials are in danger |
| 7 | What is the minimum tyre pressure requirement? | Set by Pirelli per event |
| 8 | How many DRS zones are typically present? | 1–3 zones (circuit-dependent) |

---

## Technology Stack

| Layer | Technology |
|-------|-----------|
| **Server** | FastAPI + Uvicorn + WebSocket |
| **RAG Framework** | LangChain + LangChain Community |
| **Vector Store** | ChromaDB |
| **Embeddings** | BAAI/bge-small-en-v1.5 |
| **LLM** | Ollama (phi3, tinyllama, qwen) |
| **Fitness** | SentenceTransformers (all-MiniLM-L6-v2) |
| **Optimizer** | Overclocking Algorithm (OCA) |
| **Frontend** | TailwindCSS + Vanilla JS |
| **Visualization** | SVG Charts |

---

## Benchmark Dashboard

The real-time dashboard (`/bench`) provides:

- **KPI Row**: Best fitness, current iteration, avg fitness, best parameters
- **Convergence Chart**: Live line chart of best vs. raw fitness
- **Parameter Space**: Scatter plot of explored configurations
- **Iteration Log**: Table of all evaluation results

---

## Performance

| Metric | Value |
|--------|-------|
| **Default Budget** | 50 LLM calls |
| **Avg Time per Call** | ~15–20 seconds |
| **Total Runtime** | ~15 minutes |
| **Cache Efficiency** | ~20–30% hit rate |

---

## Troubleshooting

### Common Issues

| Issue | Solution |
|-------|---------|
| Ollama model not found | Pull model: `ollama pull phi3` |
| No PDF files found | Place PDFs in `data/` directory |
| WebSocket connection failed | Use HTTP fallback endpoint |
| Empty vector store | Delete `chroma_db/` and restart |

### Enabling Debug Logs

```bash
# Set debug mode
export PYTHONUNBUFFERED=1
uvicorn server:app --log-level debug
```

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

## Citation

If you use this code in your research, please cite:

```bibtex
@software{rag_oca_f1,
  title = {RAG Hyperparameter Optimization with OCA},
  author = {Your Name},
  year = {2024},
  url = {https://github.com/yourusername/rag_oca_f1},
  license = {MIT}
}
```

---

## References

- [LangChain Documentation](https://python.langchain.com/docs/)
- [ChromaDB Documentation](https://docs.trychroma.com/)
- [Ollama Documentation](https://github.com/ollama/ollama)
- [SentenceTransformers](https://sbert.net/)

---

<p align="center">
  <sub>Built for F1 Regulations</sub>
</p>