"""
Fitness Function for RAG Hyperparameter Optimization
Evaluates configurations using Semantic Similarity against Gold Standard Q&A
"""

import json
import os
import shutil
from typing import Dict, Optional, Tuple

from sentence_transformers import SentenceTransformer, util


# Global sentence transformer model (loaded once)
_model = SentenceTransformer("all-MiniLM-L6-v2")

# Load gold standard Q&A pairs
_project_root = os.path.dirname(os.path.dirname(__file__))
_gold_path = os.path.join(_project_root, "src", "gold_standard.json")
_chroma_root = os.path.join(_project_root, "chroma_db")

with open(_gold_path, "r") as f:
    GOLD_STANDARD = json.load(f)

# Cache for evaluated configurations (saves LLM calls)
_evaluation_cache: Dict[Tuple, float] = {}
_last_persist_dir: Optional[str] = None


def _get_persist_dir(chunk_size: int, chunk_overlap: int) -> str:
    return os.path.join(_chroma_root, f"db_{chunk_size}_{chunk_overlap}")


def _cleanup_previous_persist_dir(current_persist_dir: str) -> None:
    """Keep only the active iteration's persisted vector store directory."""
    global _last_persist_dir

    current_abs = os.path.abspath(current_persist_dir)
    chroma_root_abs = os.path.abspath(_chroma_root)

    if _last_persist_dir is None:
        if os.path.isdir(chroma_root_abs):
            for entry in os.listdir(chroma_root_abs):
                candidate = os.path.join(chroma_root_abs, entry)
                candidate_abs = os.path.abspath(candidate)
                if (
                    entry.startswith("db_")
                    and os.path.isdir(candidate_abs)
                    and candidate_abs != current_abs
                ):
                    try:
                        shutil.rmtree(candidate_abs)
                        print(f"[CLEANUP] Removed stale vector store: {candidate_abs}")
                    except Exception as exc:
                        print(f"[WARN] Could not remove stale vector store '{candidate_abs}': {exc}")
        _last_persist_dir = current_abs
        return

    previous_abs = os.path.abspath(_last_persist_dir)

    if (
        previous_abs != current_abs
        and previous_abs.startswith(chroma_root_abs)
        and os.path.isdir(previous_abs)
    ):
        try:
            shutil.rmtree(previous_abs)
            print(f"[CLEANUP] Removed previous vector store: {previous_abs}")
        except Exception as exc:
            print(f"[WARN] Could not remove previous vector store '{previous_abs}': {exc}")

    _last_persist_dir = current_abs


def fitness_function(params: list) -> float:
    """
    Evaluate a RAG configuration using semantic similarity.

    Args:
        params: List [chunk_size, chunk_overlap, temperature, top_k]

    Returns:
        Fitness score between 0.0 and 1.0 (higher is better)
    """
    # Extract hyperparameters
    chunk_size = int(round(params[0]))  # Range: 100-1000
    chunk_overlap = int(round(params[1]))  # Range: 0-200
    temperature = float(params[2])  # Range: 0.0-1.0
    top_k = int(round(params[3]))  # Range: 1-10

    # Penalty: chunk_overlap must be less than chunk_size
    if chunk_overlap >= chunk_size:
        print(f"[PENALTY] chunk_overlap ({chunk_overlap}) >= chunk_size ({chunk_size})")
        return 0.0

    persist_dir = _get_persist_dir(chunk_size, chunk_overlap)
    _cleanup_previous_persist_dir(persist_dir)

    # Create cache key (round temperature to 2 decimals for caching)
    cache_key = (chunk_size, chunk_overlap, round(temperature, 2), top_k)

    # Return cached result if available
    if cache_key in _evaluation_cache:
        print(f"[CACHE] Using cached result for {cache_key}")
        return _evaluation_cache[cache_key]

    print(
        f"[EVAL] Testing config: chunk_size={chunk_size}, chunk_overlap={chunk_overlap}, "
        f"temperature={temperature:.2f}, top_k={top_k}"
    )

    try:
        # Import here to avoid circular imports
        from src.rag_pipeline import RAGPipeline

        print(
            f"  >> Creating RAG pipeline with chunk_size={chunk_size}, chunk_overlap={chunk_overlap}, top_k={top_k}, temperature={temperature:.2f}"
        )

        # Create RAG pipeline with given hyperparameters
        rag = RAGPipeline(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            top_k=top_k,
            temperature=temperature,
        )

        print(
            f"  >> RAG pipeline created. Evaluating {len(GOLD_STANDARD)} Q&A pairs..."
        )

        # Evaluate on each gold standard Q&A pair
        similarity_scores = []

        for idx, item in enumerate(GOLD_STANDARD):
            question = item["question"]
            gold_answer = item["answer"]

            print(f"  >> [{idx + 1}/{len(GOLD_STANDARD)}] Querying: {question[:50]}...")

            # Get RAG prediction
            predicted_answer = rag.query(question)

            # Compute semantic similarity
            gold_embedding = _model.encode(gold_answer, convert_to_tensor=True)
            pred_embedding = _model.encode(predicted_answer, convert_to_tensor=True)

            similarity = float(util.cos_sim(gold_embedding, pred_embedding))
            similarity_scores.append(similarity)
            print(f"       Similarity: {similarity:.4f}")

        # Calculate average fitness score
        fitness = sum(similarity_scores) / len(similarity_scores)
        print(f"  >> Average fitness: {fitness:.4f}")

    except Exception as e:
        print(f"[ERROR] Fitness evaluation failed: {e}")
        fitness = 0.0

    # Cache the result
    _evaluation_cache[cache_key] = fitness

    print(f"[RESULT] Fitness = {fitness:.4f}")

    return fitness


def clear_cache():
    """Clear the evaluation cache."""
    global _evaluation_cache
    _evaluation_cache.clear()


def get_cache_size() -> int:
    """Get the number of cached evaluations."""
    return len(_evaluation_cache)
