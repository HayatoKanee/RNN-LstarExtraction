"""
Complete Teacher implementation for UPCA-L*.

This orchestrates the RNN oracle, partitioner, and whitebox
counterexample generator to provide a complete teacher for L*.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Optional, Tuple, Dict
import time

from .rnn_oracle import RNNOracle
from partitioning.svm_partitioner import SVMPartitioner
from counterexample.whitebox_oracle import WhiteboxEquivalenceOracle
from counterexample.sliding_window_oracle import SlidingWindowOracle
from counterexample.pac_oracle import PACEquivalenceOracle, WeightedPACOracle
from counterexample.w_method_oracle import WMethodOracle
from counterexample.bfs_oracle import BFSOracle
from counterexample.random_wp_oracle import RandomWpOracle
from core.dfa import DFA


class Teacher:
    """
    Teacher for L* algorithm - orchestrates membership and equivalence queries.
    
    Following Weiss et al. (2018), this class coordinates between the RNN oracle
    (for membership queries) and equivalence oracles (whitebox, PAC, etc.) without
    creating circular dependencies. The Teacher acts as the sole interface for L*.
    """
    
    def __init__(self, rnn_model: nn.Module, alphabet: List[str],
                 initial_split_depth: int = 10,
                 starting_examples: List[Tuple[str, bool]] = None,
                 device: Optional[str] = None,
                 use_adaptive: bool = True,
                 oracle_type: str = "whitebox",
                 oracle_params: Optional[Dict] = None,
                 use_starting_examples: bool = True):
        """
        Initialize teacher.
        
        Args:
            rnn_model: Trained RNN model
            alphabet: Input alphabet
            initial_split_depth: Initial partitioning depth (for whitebox)
            starting_examples: Examples to check first
            device: Computation device
            use_adaptive: Whether to use UPCA-L* enhancements
            oracle_type: "whitebox", "pac", "w_method", or "bfs"
            oracle_params: Oracle-specific parameters
            use_starting_examples: Whether to generate starting examples (default True)
        """
        self.alphabet = alphabet
        self.oracle_type = oracle_type
        
        # Initialize RNN oracle (only handles membership queries)
        self.rnn_oracle = RNNOracle(
            rnn_model=rnn_model,
            alphabet=alphabet,
            device=device,
            threshold=0.5  # Default threshold
        )
        
        # Find starting examples if not provided and if enabled
        if starting_examples is None and use_starting_examples:
            starting_examples = self._find_starting_examples()
        elif starting_examples is None:
            starting_examples = []  # Empty list if disabled
            self.starting_example_queries = 0
            
        # Initialize equivalence oracle
        self.equivalence_oracle = self._create_equivalence_oracle(
            oracle_type, oracle_params or {}, initial_split_depth, starting_examples
        )
        
        
        # Statistics
        self.dfas_proposed = []
        self.counterexamples = []
        self.iteration = 0
        
    def membership_queries(self, words: List[str]) -> List[bool]:
        """Batch membership queries."""
        return self.rnn_oracle.membership_queries(words)
        
    def classify_word(self, word: str) -> bool:
        """Single membership query."""
        return self.rnn_oracle.classify_word(word)
        
    def set_counterexample_blacklist(self, blacklist: set):
        """
        Set the blacklist of counterexamples for the equivalence oracle.
        
        Args:
            blacklist: Set of strings that should not be returned as counterexamples
        """
        if hasattr(self.equivalence_oracle, 'set_blacklist'):
            self.equivalence_oracle.set_blacklist(blacklist)
    
    def equivalence_query(self, hypothesis_dfa: DFA, 
                         iteration: Optional[int] = None) -> Optional[str]:
        """
        Equivalence query delegated to the configured oracle.
        
        Args:
            hypothesis_dfa: L* proposed DFA
            iteration: Current L* iteration
            
        Returns:
            Counterexample or None if equivalent
        """
        self.iteration = iteration or self.iteration + 1
        self.dfas_proposed.append(hypothesis_dfa)
        
        print(f"\nEquivalence query (iteration {self.iteration})")
        print(f"  Oracle type: {self.oracle_type}")
        print(f"  DFA states: {len(hypothesis_dfa.states)}")
        
        # Get time limit if set
        remaining_time = None
        if hasattr(self.rnn_oracle, 'time_limit') and self.rnn_oracle.time_limit:
            elapsed = time.time() - self.rnn_oracle.start_time
            remaining_time = self.rnn_oracle.time_limit - elapsed
            
        # Delegate to equivalence oracle
        start_time = time.time()
        counterexample = self.equivalence_oracle.find_counterexample(
            hypothesis_dfa, self.iteration, time_limit=remaining_time
        )
            
        if counterexample:
            # Found counterexample
            ce_time = time.time() - start_time
            print(f"  Counterexample found: '{counterexample}' "
                  f"(length {len(counterexample)}, time: {ce_time:.2f}s)")
            self.counterexamples.append((counterexample, ce_time))
            return counterexample
        else:
            # No counterexample found
            print(f"  No counterexample found (time: {time.time() - start_time:.2f}s)")
            return None
                
    def set_time_limit(self, time_limit: float, start_time: float):
        """Set time limit for operations."""
        self.rnn_oracle.set_time_limit(time_limit, start_time)
        
    def _find_starting_examples(self) -> List[Tuple[str, bool]]:
        """
        Find starting examples using Weiss et al. approach.
        
        Generate a comprehensive set of strings and find the shortest
        positive and negative examples. This ensures we have proper
        starting examples even for complex grammars.
        """
        # Following Weiss: generate many samples across different lengths
        train_set = {}
        lengths = list(range(8))  # Check lengths 0-7
        samples_per_length = 200
        self.starting_example_queries = 0  # Track queries used 
        
        for length in lengths:
            if length == 0:
                candidates = ['']
            else:
                # Generate all possible strings for small lengths
                if length <= 3:
                    # Exhaustive enumeration for small lengths
                    candidates = []
                    for i in range(len(self.alphabet) ** length):
                        word = ''
                        num = i
                        for _ in range(length):
                            word = self.alphabet[num % len(self.alphabet)] + word
                            num //= len(self.alphabet)
                        candidates.append(word)
                else:
                    # Random sampling for longer lengths
                    candidates = []
                    for _ in range(samples_per_length):
                        word = ''.join(np.random.choice(self.alphabet, size=length))
                        candidates.append(word)
            
            # Classify all candidates
            for word in candidates:
                if word not in train_set:  # Avoid duplicates
                    train_set[word] = self.classify_word(word)
                    self.starting_example_queries += 1
        
        # Find shortest positive and negative examples
        sorted_words = sorted(train_set.keys(), key=len)
        shortest_pos = next((w for w in sorted_words if train_set[w]), None)
        shortest_neg = next((w for w in sorted_words if not train_set[w]), None)
        
        starting_examples = []
        if shortest_pos is not None:
            starting_examples.append((shortest_pos, True))
        if shortest_neg is not None:
            starting_examples.append((shortest_neg, False))
            
        # Log what we found
        print(f"Starting example generation: checked {len(train_set)} strings ({self.starting_example_queries} queries)")
        print(f"  Positive examples: {sum(1 for v in train_set.values() if v)}")
        print(f"  Negative examples: {sum(1 for v in train_set.values() if not v)}")
        if starting_examples:
            print(f"  Selected starting examples: {starting_examples}")
        else:
            print("  WARNING: Could not find both positive and negative examples!")
            
        return starting_examples
        
    def get_statistics(self) -> Dict:
        """Get teacher statistics."""
        rnn_stats = self.rnn_oracle.get_statistics()
        
        # Include starting example queries in the total count
        total_membership_queries = rnn_stats.get('total_queries', 0)
        if hasattr(self, 'starting_example_queries'):
            total_membership_queries += self.starting_example_queries
            
        stats = {
            'iterations': self.iteration,
            'counterexamples': len(self.counterexamples),
            'oracle_type': self.oracle_type,
            'membership_queries': total_membership_queries,
            'starting_example_queries': getattr(self, 'starting_example_queries', 0)
        }
        
        # Add partitioner stats for whitebox and sliding window oracles
        if self.oracle_type in ["whitebox", "sliding_window"] and hasattr(self, 'partitioner'):
            stats['partitions'] = self.partitioner.get_num_partitions()
        
        # Add equivalence oracle statistics
        if hasattr(self.equivalence_oracle, 'get_statistics'):
            oracle_stats = self.equivalence_oracle.get_statistics()
            stats['equivalence_queries'] = oracle_stats.get('total_queries', 0)
            stats['oracle_specific'] = oracle_stats
        
        # Add counterexample statistics
        if self.counterexamples:
            ce_lengths = [len(ce) for ce, _ in self.counterexamples]
            ce_times = [t for _, t in self.counterexamples]
            stats['avg_ce_length'] = np.mean(ce_lengths)
            stats['avg_ce_time'] = np.mean(ce_times)
            stats['min_ce_length'] = min(ce_lengths)
            stats['max_ce_length'] = max(ce_lengths)
            
        return stats
    
    def _create_equivalence_oracle(self, oracle_type: str, params: Dict,
                                   initial_split_depth: int,
                                   starting_examples: List[Tuple[str, bool]]):
        """
        Factory method for creating equivalence oracles.
        
        Args:
            oracle_type: Type of oracle to create
            params: Oracle-specific parameters
            initial_split_depth: For whitebox oracle partitioning
            starting_examples: Initial examples for whitebox
            
        Returns:
            Initialized equivalence oracle
            
        Raises:
            ValueError: If oracle_type is unknown
        """
        oracle_constructors = {
            "whitebox": self._create_whitebox_oracle,
            "sliding_window": self._create_sliding_window_oracle,
            "pac": self._create_pac_oracle,
            "w_method": self._create_w_method_oracle,
            "bfs": self._create_bfs_oracle,
            "random_wp": self._create_random_wp_oracle
        }
        
        if oracle_type not in oracle_constructors:
            raise ValueError(
                f"Unknown oracle type: {oracle_type}. "
                f"Available types: {list(oracle_constructors.keys())}"
            )
            
        return oracle_constructors[oracle_type](params, initial_split_depth, starting_examples)
    
    def _create_whitebox_oracle(self, params: Dict, initial_split_depth: int,
                               starting_examples: List[Tuple[str, bool]]):
        """Create whitebox oracle with SVM partitioner."""
        # Initialize partitioner (stored for statistics)
        self.partitioner = SVMPartitioner(
            params.get('split_depth', initial_split_depth)
        )
        
        return WhiteboxEquivalenceOracle(
            self.rnn_oracle,
            self.alphabet,
            self.partitioner,
            starting_examples
        )
    
    def _create_sliding_window_oracle(self, params: Dict, initial_split_depth: int,
                                    starting_examples: List[Tuple[str, bool]]):
        """Create sliding window oracle that uses context-aware partitioning."""
        # Initialize partitioner for sliding window states
        self.partitioner = SVMPartitioner(
            params.get('split_depth', initial_split_depth)
        )
        
        return SlidingWindowOracle(
            self.rnn_oracle,
            self.alphabet,
            self.partitioner,
            window_size=params.get('window_size', 4),
            starting_examples=starting_examples
        )
    
    
    def _create_pac_oracle(self, params: Dict, *args):
        """Create PAC oracle with statistical guarantees."""
        return PACEquivalenceOracle(
            self.rnn_oracle,
            self.alphabet,
            epsilon=params.get('epsilon', 0.1),
            delta=params.get('delta', 0.1),
            max_length=params.get('max_length', 50),
            distribution=params.get('distribution', 'uniform')
        )
    
    def _create_w_method_oracle(self, params: Dict, *args):
        """Create W-method oracle for systematic testing."""
        return WMethodOracle(
            self.rnn_oracle,
            self.alphabet,
            max_target_states=params.get('max_target_states', 10)
        )
    
    def _create_bfs_oracle(self, params: Dict, *args):
        """Create BFS oracle for breadth-first exploration."""
        return BFSOracle(
            self.rnn_oracle,
            self.alphabet,
            max_depth=params.get('max_depth', 20),
            breadth_limit=params.get('breadth_limit', 10000)
        )
    
    def _create_random_wp_oracle(self, params: Dict, *args):
        """Create Random W_p oracle for probabilistic testing."""
        return RandomWpOracle(
            self.rnn_oracle,
            self.alphabet,
            min_length=params.get('min_length', 3),
            expected_length=params.get('expected_length', 11),
            num_tests=params.get('num_tests', 10000),
            max_total_length=params.get('max_total_length', 50)
        )