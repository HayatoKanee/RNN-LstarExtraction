"""Equivalence oracle implementations for DFA extraction."""

from .base_oracle import EquivalenceOracle
from .whitebox_oracle import WhiteboxEquivalenceOracle
from .pac_oracle import PACEquivalenceOracle, WeightedPACOracle
from .w_method_oracle import WMethodOracle
from .bfs_oracle import BFSOracle
from .random_wp_oracle import RandomWpOracle
from .sliding_window_oracle import SlidingWindowOracle

__all__ = [
    'EquivalenceOracle',
    'WhiteboxEquivalenceOracle',
    'PACEquivalenceOracle',
    'WeightedPACOracle',
    'WMethodOracle',
    'BFSOracle',
    'RandomWpOracle',
    'SlidingWindowOracle'
]