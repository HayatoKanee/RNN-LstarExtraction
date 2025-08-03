"""
RNN Extraction Benchmark

A comprehensive benchmarking framework for comparing different equivalence oracle
approaches in DFA extraction from RNNs using the L* algorithm.
"""

from .core.lstar import LStarAlgorithm, run_lstar
from .teacher.teacher import Teacher

__version__ = "0.1.0"
__all__ = ["LStarAlgorithm", "run_lstar", "Teacher"]