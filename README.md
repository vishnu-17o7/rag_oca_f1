# RAG Hyperparameter Optimization using OCA

Optimizes RAG pipeline hyperparameters for F1/FIA regulatory documents using the Overclocking Algorithm (OCA).

## Project Structure

```
rag_oca_f1/
├── data/                    # Place F1 PDF regulations here
├── research/
│   └── oca.py               # OCA wrapper (imports from research/src/oca/)
├── src/
│   ├── rag_pipeline.py      # RAG pipeline implementation
│   ├── fitness.py           # Fitness function (semantic similarity)
│   ├── gold_standard.json   # 8 Q&A gold standard pairs
│   └── visualize.py         # Convergence table & plot
├── main.py                  # Entry point
├── requirements.txt         # Dependencies
└── README.md
```

## Prerequisites

1. **Python 3.10+**
2. **Ollama** running locally with model pulled:
   ```bash
   ollama pull phi3
   # or
   ollama pull tinyllama
   ```

## Installation

```bash
# Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Place F1/FIA PDF regulations in the data/ folder
```

## Usage

Run the optimization:

```bash
python main.py
```

## Hyperparameters Optimized

| Parameter | Range | Description |
|-----------|-------|-------------|
| chunk_size | 100-1000 | Characters per text chunk |
| chunk_overlap | 0-200 | Overlap between chunks |
| temperature | 0.0-1.0 | LLM generation temperature |
| top_k | 1-10 | Number of documents to retrieve |

## Budget

- Maximum 50 LLM calls (Ollama queries)
- Results are cached to avoid redundant evaluations

## Output

- Convergence table printed to console
- Plot saved to `results/convergence_plot.png`

## Credits

- OCA algorithm sourced from `research/src/oca/algorithm.py`
- RAG built with LangChain + ChromaDB + Ollama
