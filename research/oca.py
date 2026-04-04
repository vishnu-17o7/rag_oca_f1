"""
OCA (Overclocking Algorithm)

The original project referenced an external module at `research/src/oca/algorithm.py`.
That source is not present in this repository, so this file provides a compatible
in-repo implementation with the same constructor and optimize API used by the app.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Tuple

import numpy as np


@dataclass
class OverclockingAlgorithm:
    """
    Lightweight population-based optimizer for maximizing an objective function.

    API compatibility:
    - __init__(pop_size, num_p_cores, initial_voltage, final_voltage, aggressive_voltage)
    - optimize(objective_fn, bounds, dim, max_iterations)
    """

    pop_size: int = 10
    num_p_cores: int = 3
    initial_voltage: float = 2.0
    final_voltage: float = 0.0
    aggressive_voltage: bool = True

    def __post_init__(self) -> None:
        self.pop_size = max(2, int(self.pop_size))
        self.num_p_cores = max(1, int(self.num_p_cores))
        self.initial_voltage = float(self.initial_voltage)
        self.final_voltage = float(self.final_voltage)

    def _evaluate_population(
        self, population: np.ndarray, objective_fn: Callable[[np.ndarray], float]
    ) -> np.ndarray:
        scores = np.zeros(population.shape[0], dtype=float)
        for idx, position in enumerate(population):
            scores[idx] = float(objective_fn(position))
        return scores

    def optimize(
        self,
        objective_fn: Callable[[np.ndarray], float],
        bounds: Tuple[float, float],
        dim: int,
        max_iterations: int,
    ) -> Tuple[np.ndarray, float, List[float]]:
        """
        Maximize objective_fn over a shared bound range for all dimensions.
        Returns (best_position, best_fitness, convergence_curve).
        """
        dim = int(dim)
        max_iterations = max(1, int(max_iterations))
        lower, upper = float(bounds[0]), float(bounds[1])

        if dim <= 0:
            raise ValueError("dim must be greater than 0")
        if lower >= upper:
            raise ValueError("bounds must be (lower, upper) with lower < upper")

        population = np.random.uniform(lower, upper, size=(self.pop_size, dim))
        velocities = np.zeros_like(population)

        fitness = self._evaluate_population(population, objective_fn)
        personal_best_positions = population.copy()
        personal_best_fitness = fitness.copy()

        best_idx = int(np.argmax(fitness))
        global_best_position = population[best_idx].copy()
        global_best_fitness = float(fitness[best_idx])
        convergence_curve = [global_best_fitness]

        for iteration in range(max_iterations):
            progress = iteration / max(1, max_iterations - 1)
            voltage = self.initial_voltage + (
                self.final_voltage - self.initial_voltage
            ) * progress

            voltage_span = self.initial_voltage - self.final_voltage
            if abs(voltage_span) < 1e-9:
                voltage_norm = 0.5
            else:
                voltage_norm = (voltage - self.final_voltage) / voltage_span
            voltage_norm = float(np.clip(voltage_norm, 0.05, 1.0))

            inertia = 0.35 + (0.45 * voltage_norm)
            cognitive = 1.10 + ((0.80 if self.aggressive_voltage else 0.40) * voltage_norm)
            social = 1.40 + ((0.50 if self.aggressive_voltage else 0.20) * (1.0 - voltage_norm))

            rand_cognitive = np.random.rand(self.pop_size, dim)
            rand_social = np.random.rand(self.pop_size, dim)

            velocities = (
                (inertia * velocities)
                + (cognitive * rand_cognitive * (personal_best_positions - population))
                + (social * rand_social * (global_best_position - population))
            )

            noise_scale = (upper - lower) * (
                0.015 * voltage_norm if self.aggressive_voltage else 0.005 * voltage_norm
            )
            exploration_noise = np.random.normal(0.0, noise_scale, size=population.shape)

            population = np.clip(population + velocities + exploration_noise, lower, upper)

            fitness = self._evaluate_population(population, objective_fn)
            improved = fitness > personal_best_fitness

            personal_best_positions[improved] = population[improved]
            personal_best_fitness[improved] = fitness[improved]

            candidate_idx = int(np.argmax(fitness))
            candidate_fitness = float(fitness[candidate_idx])
            if candidate_fitness > global_best_fitness:
                global_best_fitness = candidate_fitness
                global_best_position = population[candidate_idx].copy()

            convergence_curve.append(global_best_fitness)

        return global_best_position, global_best_fitness, convergence_curve


__all__ = ["OverclockingAlgorithm"]
