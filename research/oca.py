"""
OCA (Overclocking Algorithm) Wrapper
Wraps the OverclockingAlgorithm from research/src/oca/algorithm.py
"""

import sys
import os

# Add parent directory to path to import OCA from original research location
parent_dir = os.path.dirname(
    os.path.dirname(os.path.dirname(__file__))
)  # Go up to mho lab
sys.path.insert(0, parent_dir)

from research.src.oca.algorithm import OverclockingAlgorithm

__all__ = ["OverclockingAlgorithm"]
