"""
Configuration and factory for different equivalence oracle implementations.

This module provides a unified interface for creating and configuring different
equivalence oracle strategies for benchmarking.
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional, Callable, List
from enum import Enum

from counterexample.base_oracle import EquivalenceOracle
from counterexample.whitebox_oracle import WhiteboxEquivalenceOracle
from counterexample.pac_oracle import PACEquivalenceOracle
from counterexample.w_method_oracle import WMethodOracle
from counterexample.bfs_oracle import BFSOracle
from counterexample.sliding_window_oracle import SlidingWindowOracle
from partitioning.svm_partitioner import SVMPartitioner


class OracleType(Enum):
    """Available oracle types for benchmarking."""
    WHITEBOX = "whitebox"
    PAC = "pac"
    W_METHOD = "w_method"
    BFS = "bfs"
    RANDOM_WP = "random_wp"
    SLIDING_WINDOW = "sliding_window"


@dataclass
class OracleConfig:
    """Configuration for a specific oracle implementation."""
    
    oracle_type: OracleType
    
    # Common parameters
    time_limit: float = 60.0
    
    # PAC oracle parameters
    epsilon: float = 0.01  # Error tolerance
    delta: float = 0.01    # Confidence parameter
    max_length: int = 50   # Maximum string length for sampling
    distribution: str = 'uniform'  # Sampling distribution
    
    # W-method oracle parameters
    max_target_states: int = 10  # Upper bound on target automaton states
    
    # BFS oracle parameters
    max_depth: int = 15
    breadth_limit: int = 1000  # Max nodes per level
    
    # Whitebox oracle parameters
    split_depth: int = 10  # Initial dimension split depth for partitioning
    
    # Random W_p oracle parameters (Note: defaults differ from config below)
    min_length: int = 3        # Minimum infix length
    expected_length: int = 11  # Expected infix length (geometric distribution)
    num_tests: int = 10000     # Number of random tests per equivalence query
    max_total_length: int = 10  # Maximum total string length
    
    # Sliding window oracle parameters
    window_size: int = 4       # Size of the sliding window
    
    # Bounded L* parameters
    use_bounded_lstar: bool = False  # Whether to use bounded L* with negotiation
    max_query_length: int = 20       # Maximum query length for bounded L*
    
    # Starting examples override
    use_starting_examples_override: Optional[bool] = None  # Override starting examples behavior
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for serialization."""
        # Base parameters
        result = {
            'oracle_type': self.oracle_type.value,
            'time_limit': self.time_limit,
        }
        
        # Oracle-specific parameters
        if self.oracle_type == OracleType.PAC:
            result.update({
                'epsilon': self.epsilon,
                'delta': self.delta,
                'max_length': self.max_length,
                'distribution': self.distribution,
            })
        elif self.oracle_type == OracleType.W_METHOD:
            result.update({
                'max_target_states': self.max_target_states,
            })
        elif self.oracle_type == OracleType.BFS:
            result.update({
                'max_depth': self.max_depth,
                'breadth_limit': self.breadth_limit,
            })
        elif self.oracle_type == OracleType.WHITEBOX:
            result.update({
                'split_depth': self.split_depth,
            })
        elif self.oracle_type == OracleType.RANDOM_WP:
            result.update({
                'min_length': self.min_length,
                'expected_length': self.expected_length,
                'num_tests': self.num_tests,
                'max_total_length': self.max_total_length,
            })
        elif self.oracle_type == OracleType.SLIDING_WINDOW:
            result.update({
                'window_size': self.window_size,
            })
        
        # Always include bounded L* parameters if they're not default
        if self.use_bounded_lstar:
            result.update({
                'use_bounded_lstar': self.use_bounded_lstar,
                'max_query_length': self.max_query_length,
            })
            
        return result




def get_sliding_window_configs(window_sizes: List[int] = None) -> Dict[str, OracleConfig]:
    """Get sliding window configurations with different window sizes."""
    if window_sizes is None:
        window_sizes = [2, 4, 6, 8]  # Default window sizes to test
    
    configs = {}
    for window_size in window_sizes:
        configs[f"sliding_window_w{window_size}"] = OracleConfig(
            oracle_type=OracleType.SLIDING_WINDOW,
            window_size=window_size,
            split_depth=10,
            time_limit=60.0
        )
    
    return configs


def get_bounded_configs(max_query_length: int = 20) -> Dict[str, OracleConfig]:
    """Get configurations with bounded L* enabled for each oracle type."""
    configs = {}
    
    # Get base configs
    base_configs = get_default_configs()
    
    # Create bounded versions
    for name, config in base_configs.items():
        bounded_config = OracleConfig(
            oracle_type=config.oracle_type,
            time_limit=config.time_limit,
            use_bounded_lstar=True,
            max_query_length=max_query_length
        )
        
        # Copy oracle-specific parameters
        if config.oracle_type == OracleType.PAC:
            bounded_config.epsilon = config.epsilon
            bounded_config.delta = config.delta
            bounded_config.max_length = config.max_length
            bounded_config.distribution = config.distribution
        elif config.oracle_type == OracleType.W_METHOD:
            bounded_config.max_target_states = config.max_target_states
        elif config.oracle_type == OracleType.BFS:
            bounded_config.max_depth = config.max_depth
            bounded_config.breadth_limit = config.breadth_limit
        elif config.oracle_type == OracleType.WHITEBOX:
            bounded_config.split_depth = config.split_depth
        elif config.oracle_type == OracleType.RANDOM_WP:
            bounded_config.min_length = config.min_length
            bounded_config.expected_length = config.expected_length
            bounded_config.num_tests = config.num_tests
            bounded_config.max_total_length = config.max_total_length
        elif config.oracle_type == OracleType.SLIDING_WINDOW:
            bounded_config.window_size = config.window_size
            bounded_config.split_depth = config.split_depth
            
        configs[f"{name}_bounded"] = bounded_config
    
    return configs


def get_default_configs() -> Dict[str, OracleConfig]:
    """Get default configurations for each oracle type."""
    return {
        "whitebox": OracleConfig(
            oracle_type=OracleType.WHITEBOX,
            split_depth=10,
            time_limit=60.0
        ),
        "pac": OracleConfig(
            oracle_type=OracleType.PAC,
            epsilon=0.001,
            delta=0.001,
            distribution='uniform',  # As specified in Table 4.1
            max_length=15,  # Limit to avoid querying beyond RNN training range
            time_limit=60.0
        ),
        "w_method": OracleConfig(
            oracle_type=OracleType.W_METHOD,
            max_target_states=10,
            time_limit=60.0
        ),
        "bfs": OracleConfig(
            oracle_type=OracleType.BFS,
            max_depth=10,  # Restricted to depth 10
            breadth_limit=1000,
            time_limit=60.0
        ),
        "random_wp": OracleConfig(
            oracle_type=OracleType.RANDOM_WP,
            min_length=3,           # Min infix length from Table 4.1
            expected_length=5,      # Expected infix length from Table 4.1
            num_tests=10000,        # Number of tests from Table 4.1
            max_total_length=10,    # Max total length from Table 4.1
            time_limit=60.0
        ),
        "sliding_window": OracleConfig(
            oracle_type=OracleType.SLIDING_WINDOW,
            window_size=4,          # Default window size (can be overridden)
            split_depth=10,         # Also uses SVM partitioning like whitebox
            time_limit=60.0
        )
    }