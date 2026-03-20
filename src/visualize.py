"""
Visualization Module for OCA Convergence
Generates console table and matplotlib plot
"""

import os
import pandas as pd
import matplotlib.pyplot as plt


def print_table(history: list):
    """
    Print convergence table to console.

    Args:
        history: List of (iteration, best_fitness, params) tuples
    """
    if not history:
        print("No history to display.")
        return

    print("\n" + "-" * 60)
    print("Generating convergence table...")
    print("-" * 60)

    # Build DataFrame
    data = []
    for iteration, best_fitness, params in history:
        data.append(
            {
                "Iteration": iteration + 1,  # 1-indexed for display
                "Best Fitness": f"{best_fitness:.4f}",
                "chunk_size": int(round(params[0])),
                "chunk_overlap": int(round(params[1])),
                "temperature": f"{params[2]:.2f}",
                "top_k": int(round(params[3])),
            }
        )

    df = pd.DataFrame(data)

    # Print table
    print("\n" + "=" * 85)
    print("OCA CONVERGENCE TABLE")
    print("=" * 85)
    print(df.to_string(index=False))
    print("=" * 85 + "\n")


def plot_convergence(history: list, save_path: str = None):
    """
    Generate and save convergence plot.

    Args:
        history: List of (iteration, best_fitness, params) tuples
        save_path: Path to save plot (optional)
    """
    if not history:
        print("No history to plot.")
        return

    print("-" * 60)
    print("Generating convergence plot...")
    print("-" * 60)

    # Extract data
    iterations = [h[0] + 1 for h in history]  # 1-indexed
    fitness_scores = [h[1] for h in history]

    # Create plot
    plt.figure(figsize=(10, 6))

    # Plot convergence curve
    plt.plot(iterations, fitness_scores, "b-", linewidth=2, label="Best Fitness")

    # Plot target threshold
    plt.axhline(
        y=0.8, color="r", linestyle="--", linewidth=1.5, label="Target Threshold (0.8)"
    )

    # Fill area above threshold
    plt.fill_between(
        iterations,
        fitness_scores,
        0.8,
        where=[f >= 0.8 for f in fitness_scores],
        alpha=0.3,
        color="green",
        label="Above Target",
    )

    # Labels and title
    plt.xlabel("Iteration", fontsize=12)
    plt.ylabel("Best Fitness Score", fontsize=12)
    plt.title("OCA Convergence on F1 Regulations RAG", fontsize=14)

    # Legend and grid
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)

    # Set axis limits
    plt.xlim(1, max(iterations))
    plt.ylim(0, 1)

    plt.tight_layout()

    # Save plot
    if save_path is None:
        project_root = os.path.dirname(os.path.dirname(__file__))
        results_dir = os.path.join(project_root, "results")
        os.makedirs(results_dir, exist_ok=True)
        save_path = os.path.join(results_dir, "convergence_plot.png")

    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"[INFO] Plot saved to: {save_path}")

    # Show plot
    plt.show()
    plt.close()
