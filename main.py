"""
DEPRECATED: Use `uvicorn server:app --reload` instead
Main Entry Point for RAG Hyperparameter Optimization
Uses OCA (Overclocking Algorithm) to find optimal RAG configuration
"""

import os
import sys
import numpy as np

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from research.oca import OverclockingAlgorithm
from src.fitness import fitness_function, get_cache_size
from src.visualize import plot_convergence, print_table


# ============================================================================
# Configuration
# ============================================================================

# Search space bounds: [chunk_size, chunk_overlap, temperature, top_k]
BOUNDS = [
    (100, 1000),  # chunk_size: 100-1000 characters
    (0, 200),  # chunk_overlap: 0-200 characters
    (0.0, 1.0),  # temperature: 0.0-1.0
    (1, 10),  # top_k: 1-10 documents
]

# Hard budget limit for LLM calls
MAX_EVALUATIONS = 50

# OCA hyperparameters
POP_SIZE = 10
NUM_P_CORES = 3
INITIAL_VOLTAGE = 2.0
FINAL_VOLTAGE = 0.0


# ============================================================================
# Wrapper for OCA (handles multi-dimensional bounds)
# ============================================================================


class OCAOptimizer:
    """
    Wrapper around OverclockingAlgorithm to handle multiple dimensions.
    OCA's optimize() expects a single (lower, upper) tuple for all dimensions.
    """

    def __init__(self, bounds, max_evals, pop_size, num_p_cores):
        self.bounds = bounds
        self.max_evals = max_evals
        self.dim = len(bounds)

        # Create OCA instance
        self.oca = OverclockingAlgorithm(
            pop_size=pop_size,
            num_p_cores=num_p_cores,
            initial_voltage=INITIAL_VOLTAGE,
            final_voltage=FINAL_VOLTAGE,
            aggressive_voltage=True,
        )

        self.evaluation_count = 0
        self.history = []
        self.best_fitness = -np.inf
        self.best_params = None

    def _clip_to_bounds(self, position: np.ndarray) -> np.ndarray:
        """Clip position to bounds for each dimension."""
        clipped = np.zeros(self.dim)
        for i, (lower, upper) in enumerate(self.bounds):
            clipped[i] = np.clip(position[i], lower, upper)
        return clipped

    def _objective_wrapper(self, position: np.ndarray) -> float:
        """Wrapper that clips position and tracks history."""
        # Clip to bounds
        clipped = self._clip_to_bounds(position)

        # Evaluate fitness
        fitness = fitness_function(clipped.tolist())

        # Track history
        self.evaluation_count += 1

        # Update best
        if fitness > self.best_fitness:
            self.best_fitness = fitness
            self.best_params = clipped.copy()

        # Record history (iteration is 0-indexed internally)
        iteration = self.evaluation_count - 1
        self.history.append((iteration, self.best_fitness, self.best_params.copy()))

        print(
            f"[{self.evaluation_count}/{self.max_evals}] Best so far: {self.best_fitness:.4f}"
        )

        return fitness

    def run(self):
        """Run the optimizer."""
        print("\n" + "-" * 60)
        print("Starting OCA Optimizer")
        print("-" * 60)
        print(f"  Dimensions: {self.dim}")
        print(f"  Max evaluations: {self.max_evals}")
        print(f"  Population size: {POP_SIZE}")
        print(f"  P-Cores: {NUM_P_CORES}")
        print(f"  Bounds: {self.bounds}")
        print("-" * 60 + "\n")

        # Calculate iterations based on budget
        # Each iteration evaluates pop_size positions
        max_iterations = max(1, self.max_evals // POP_SIZE)

        # Use same bounds for all dimensions (OCA requirement)
        # We'll handle per-dimension bounds in the wrapper
        all_lower = min(b[0] for b in self.bounds)
        all_upper = max(b[1] for b in self.bounds)

        # Run optimization
        best_pos, best_fit, convergence_curve = self.oca.optimize(
            objective_fn=self._objective_wrapper,
            bounds=(all_lower, all_upper),
            dim=self.dim,
            max_iterations=max_iterations,
        )

        # Clip final result to bounds
        best_pos = self._clip_to_bounds(best_pos)

        # If we haven't reached max_evals, continue with remaining budget
        remaining = self.max_evals - self.evaluation_count

        while remaining > 0 and self.evaluation_count < self.max_evals:
            # Generate random positions to explore remaining space
            for _ in range(min(remaining, POP_SIZE)):
                random_pos = np.array(
                    [np.random.uniform(b[0], b[1]) for b in self.bounds]
                )
                self._objective_wrapper(random_pos)
                remaining = self.max_evals - self.evaluation_count
                if remaining <= 0:
                    break

        return self.best_params, self.best_fitness, self.history


# ============================================================================
# Main Execution
# ============================================================================


def main():
    """Main entry point."""
    print("\n" + "=" * 70)
    print("  RAG HYPERPARAMETER OPTIMIZATION USING OCA")
    print("  Domain: F1 / FIA Legal & Regulatory Documents")
    print("=" * 70 + "\n")

    print("[INFO] Starting optimization process...")
    print(f"[INFO] Budget: {MAX_EVALUATIONS} LLM calls maximum")
    print(f"[INFO] Hyperparameters to optimize:")
    print(f"       - chunk_size: {BOUNDS[0]}")
    print(f"       - chunk_overlap: {BOUNDS[1]}")
    print(f"       - temperature: {BOUNDS[2]}")
    print(f"       - top_k: {BOUNDS[3]}")
    print()

    # Create optimizer
    optimizer = OCAOptimizer(
        bounds=BOUNDS,
        max_evals=MAX_EVALUATIONS,
        pop_size=POP_SIZE,
        num_p_cores=NUM_P_CORES,
    )

    # Run optimization
    best_params, best_score, history = optimizer.run()

    # Print results
    print("\n" + "=" * 60)
    print("OPTIMIZATION COMPLETE")
    print("=" * 60)
    print(f"Best Fitness Score: {best_score:.4f}")
    print(f"\nOptimal Hyperparameters:")
    print(f"  chunk_size    : {int(round(best_params[0]))}")
    print(f"  chunk_overlap : {int(round(best_params[1]))}")
    print(f"  temperature   : {best_params[2]:.2f}")
    print(f"  top_k         : {int(round(best_params[3]))}")
    print(f"\nTotal evaluations: {get_cache_size()}")
    print("=" * 60)

    # Visualize results
    print_table(history)
    plot_convergence(history)

    return best_params, best_score


if __name__ == "__main__":
    main()
