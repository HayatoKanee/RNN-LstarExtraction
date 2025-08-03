"""
Sliding window counterexample generation - an enhancement of the whitebox approach.

Instead of mapping hidden states directly to partitions, we map (hidden_state, last_k_symbols)
to partitions. This allows the same hidden state to map to different DFA states based on
recent input history, potentially avoiding clustering conflicts for non-regular languages.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Set, Any
from collections import deque
import time
from copy import deepcopy

from .base_oracle import EquivalenceOracle
from core.dfa import DFA
from partitioning.svm_partitioner import SVMPartitioner


class WindowedUnrollingInfo:
    """State information with window context."""
    
    def __init__(self, dfa_state: str, path: str, RState: np.ndarray, accepting: bool, window: str):
        self.explored = False
        self.dfa_state = dfa_state
        self.paths = [path]
        self.RStates = [RState]
        self.accepting = accepting
        self.windows = [window]  # Last k symbols
        
    def __add__(self, other):
        """Combine information from multiple paths to same partition."""
        res = deepcopy(self)
        res.paths += other.paths
        res.RStates += other.RStates
        res.windows += other.windows
        return res


class SplitInfo:
    """Information about a required partitioning split."""
    
    def __init__(self, agreeing_states=None, conflicted_state=None):
        self.agreeing_states = agreeing_states  # List of (RState, window) tuples
        self.conflicted_state = conflicted_state  # Single (RState, window) tuple
        self.has_info = conflicted_state is not None


class RestartException(Exception):
    """Signal that partitioning has changed and we need to restart."""
    pass


class SlidingWindowOracle(EquivalenceOracle):
    """
    Sliding window counterexample generator using parallel exploration.
    
    Key innovation: Instead of clustering based only on hidden states, we cluster
    based on (hidden_state, last_k_symbols). This allows the same hidden state to
    map to different DFA states depending on recent input history.
    
    This helps with non-regular languages where the same hidden state needs different
    behavior based on context (e.g., balanced parentheses where "((" and "()" might
    have similar hidden states but need different DFA states).
    """
    
    def __init__(self, rnn_oracle, alphabet: List[str],
                 partitioner: SVMPartitioner,
                 window_size: int = 4,
                 starting_examples: List[Tuple[str, bool]] = None,
                 **kwargs):
        """
        Initialize sliding window oracle.
        
        Args:
            rnn_oracle: RNN oracle for membership queries
            alphabet: Input alphabet
            partitioner: State space partitioner (will be adapted for windowed states)
            window_size: Size of the sliding window (k)
            starting_examples: Known examples to check first
        """
        super().__init__(rnn_oracle, alphabet, **kwargs)
        
        self.whiteboxrnn = rnn_oracle
        self.partitioning = partitioner
        self.window_size = window_size
        self.starting_dict = {ex[0]: ex[1] for ex in (starting_examples or [])}
        
        # Statistics
        self.refinement_count = 0
        self.explored_states = set()
        
        # We'll use a modified partitioner that works on concatenated vectors
        # [hidden_state; window_encoding]
        self.window_dim = len(alphabet) * window_size  # One-hot encoding of window
        
    def _encode_window(self, window: str) -> np.ndarray:
        """Encode window as a vector for partitioning."""
        # One-hot encode each symbol in the window
        encoding = np.zeros(self.window_dim)
        char_to_idx = {char: i for i, char in enumerate(self.alphabet)}
        
        for i, char in enumerate(window):
            if i < self.window_size and char in char_to_idx:
                idx = i * len(self.alphabet) + char_to_idx[char]
                encoding[idx] = 1.0
                
        return encoding
    
    def _create_windowed_state(self, RState: np.ndarray, window: str) -> np.ndarray:
        """Combine hidden state with window encoding for partitioning."""
        window_encoding = self._encode_window(window)
        # Concatenate hidden state with window encoding
        return np.concatenate([RState, window_encoding])
    
    def _get_window(self, path: str) -> str:
        """Get the last k symbols from the path."""
        return path[-self.window_size:] if len(path) >= self.window_size else path
        
    def find_counterexample(self, hypothesis_dfa: DFA, iteration: int,
                          time_limit: Optional[float] = None) -> Optional[str]:
        """
        Find counterexample by parallel exploration with sliding window.
        
        Args:
            hypothesis_dfa: L* proposed DFA
            iteration: L* iteration number
            time_limit: Optional time limit
            
        Returns:
            Counterexample string or None
        """
        print(f"\nSliding Window Equivalence Query (iteration {iteration}, window_size={self.window_size})")
        print(f"  Current partitions: {self.partitioning.get_num_partitions()}")
        
        start_time = time.time()
        self.total_queries += 1
        self.proposed_dfa = hypothesis_dfa
        
        # First check starting examples
        cex = self._cex_from_starting_dict(hypothesis_dfa)
        if cex:
            self.counterexamples_found += 1
            self.total_time += time.time() - start_time
            print(f"  Counterexample found in starting examples: '{cex}'")
            return cex
            
        # Main loop: restarts when partitioning is refined
        while True:
            try:
                # Initialize unrolling
                self._initialize_unrolling()
                
                # Inner loop: explore abstraction
                while True:
                    # Check time limit
                    if time_limit and (time.time() - start_time) > time_limit:
                        self.total_time += time.time() - start_time
                        print(f"  Time limit reached")
                        return None
                        
                    # Check if exploration complete
                    if not self.new_RStates:
                        self.total_time += time.time() - start_time
                        print(f"  No counterexample found")
                        return None
                        
                    # Process next state
                    counterexample, split = self._process_top_pair()
                    
                    if counterexample:
                        self.counterexamples_found += 1
                        self.total_time += time.time() - start_time
                        print(f"  Counterexample found: '{counterexample}' (length {len(counterexample)})")
                        return counterexample
                    elif split.has_info:
                        # Handle partitioning refinement
                        # Get windowed states for partitioning
                        agreeing_windowed = [self._create_windowed_state(rs, w) 
                                           for rs, w in split.agreeing_states]
                        conflicted_windowed = self._create_windowed_state(
                            split.conflicted_state[0], split.conflicted_state[1])
                        
                        cluster_being_split = self.partitioning.get_partition(agreeing_windowed[0])
                        self.partitioning.refine(agreeing_windowed, conflicted_windowed)
                        self.refinement_count += 1
                            
                        if self._split_was_clean(cluster_being_split, split):
                            # Clean split - reprocess the state
                            self.new_RStates = [self.new_RStates_backup] + self.new_RStates
                        else:
                            # Need to restart unrolling
                            raise RestartException()
                            
            except RestartException:
                # Restart with refined partitioning
                continue
                
    def _initialize_unrolling(self):
        """Initialize BFS exploration."""
        self.cluster_information = {}
        initial_RState, pos = self.whiteboxrnn.get_first_RState()
        initial_window = ""  # Empty window for initial state
        self.new_RStates = [WindowedUnrollingInfo(
            self.proposed_dfa.q0, "", initial_RState, pos, initial_window)]
        
    def _cex_from_starting_dict(self, dfa: DFA) -> Optional[str]:
        """Check starting examples for counterexamples."""
        for cex, label in self.starting_dict.items():
            if dfa.accepts(list(cex)) != label:
                return cex
        return None
        
    def _process_top_pair(self) -> Tuple[Optional[str], SplitInfo]:
        """Process next state in exploration."""
        # Get next state to process
        new_info = self.new_RStates.pop(0)
        self.new_RStates_backup = new_info
        
        # Get partition for this windowed state
        windowed_state = self._create_windowed_state(new_info.RStates[0], new_info.windows[0])
        new_cluster = self.partitioning.get_partition(windowed_state)
        
        # Process the state
        counterexample, split = self._process_new_state_except_children(
            new_cluster, new_info)
            
        # If no conflicts, expand children
        if not counterexample and not split.has_info:
            self._add_children_states(new_cluster)
            
        return counterexample, split
        
    def _process_new_state_except_children(self, new_cluster: int, 
                                         new_info: WindowedUnrollingInfo) -> Tuple[Optional[str], SplitInfo]:
        """Process state and check for conflicts."""
        counterexample = None
        split = SplitInfo()
        
        # Track explored state
        self.explored_states.add((new_cluster, new_info.dfa_state))
        
        # Get existing info for this cluster
        old_info = self.cluster_information.get(new_cluster)
        full_info = old_info + new_info if old_info else new_info
        
        # Check for classification conflict
        if new_info.accepting != (new_info.dfa_state in self.proposed_dfa.F):
            counterexample = self._counterexample_from_classification_conflict(new_info)
        # Check for clustering conflict
        elif old_info and new_info.dfa_state != old_info.dfa_state:
            counterexample = self._counterexample_from_cluster_conflict(old_info, new_info)
            if not counterexample:
                # Create split info with windowed states
                split = SplitInfo(
                    agreeing_states=[(rs, w) for rs, w in zip(old_info.RStates, old_info.windows)],
                    conflicted_state=(new_info.RStates[0], new_info.windows[0]))
        else:
            # No conflicts - store the information
            self.cluster_information[new_cluster] = full_info
            
        return counterexample, split
        
    def _counterexample_from_classification_conflict(self, state_info: WindowedUnrollingInfo) -> Optional[str]:
        """Extract counterexample from classification conflict."""
        path = min(state_info.paths, key=len)
        
        # Verify this is a real counterexample
        rnn_label = self.whiteboxrnn.classify_word(path)
        dfa_accepts = state_info.dfa_state in self.proposed_dfa.F
        
        if rnn_label != dfa_accepts:
            return path
        else:
            return None
        
    def _counterexample_from_cluster_conflict(self, old_info: WindowedUnrollingInfo, 
                                            new_info: WindowedUnrollingInfo) -> Optional[str]:
        """Extract counterexample from clustering conflict."""
        q1 = old_info.dfa_state
        q2 = new_info.dfa_state
        
        # Find distinguishing suffix
        suffix = self.proposed_dfa.minimal_diverging_suffix(q1, q2)
        if not suffix:
            return None
            
        # Check all paths with suffix
        prefixes = old_info.paths + new_info.paths
        candidates = [p + suffix for p in prefixes]
        
        # Find counterexample
        for candidate in sorted(candidates, key=len):
            if self.whiteboxrnn.classify_word(candidate) != self.proposed_dfa.accepts(list(candidate)):
                return candidate
                
        return None
        
    def _add_children_states(self, cluster: int):
        """Add children states to exploration queue."""
        if cluster not in self.cluster_information:
            return
            
        state_info = self.cluster_information[cluster]
        
        if not state_info.explored:
            state_info.explored = True
            
            # Process each state in the cluster
            for i, (RState, window) in enumerate(zip(state_info.RStates, state_info.windows)):
                path = state_info.paths[i]
                
                for char in self.proposed_dfa.alphabet:
                    # Get next RNN state
                    next_path = path + char
                    next_RState, accepting = self.whiteboxrnn.get_next_RState(RState, char)
                    next_dfa_state = self.proposed_dfa.delta[state_info.dfa_state][char]
                    
                    # Update window
                    next_window = self._get_window(next_path)
                    
                    # Add to exploration queue
                    self.new_RStates.append(
                        WindowedUnrollingInfo(next_dfa_state, next_path, next_RState, 
                                            accepting, next_window))
                    
    def _split_was_clean(self, old_cluster: int, split: SplitInfo) -> bool:
        """Check if partitioning split was clean."""
        # Check that agreeing states stayed in same cluster
        agreeing_windowed = [self._create_windowed_state(rs, w) 
                           for rs, w in split.agreeing_states]
        new_states_given_to_agreeing = list(set(
            [self.partitioning.get_partition(vec) for vec in agreeing_windowed]))
            
        conflicted_windowed = self._create_windowed_state(
            split.conflicted_state[0], split.conflicted_state[1])
            
        return (self.partitioning.refinement_doesnt_hurt_other_clusters and
                new_states_given_to_agreeing == [old_cluster] and
                self.partitioning.get_partition(conflicted_windowed) != old_cluster)
                
    def get_statistics(self) -> Dict[str, Any]:
        """Return oracle statistics."""
        stats = super().get_statistics()
        
        stats.update({
            'partitions': self.partitioning.get_num_partitions(),
            'refinements': self.refinement_count,
            'states_explored': len(self.explored_states),
            'window_size': self.window_size,
            'strategy': f'sliding_window_w{self.window_size}'
        })
        
        return stats


# Helper functions for specific grammars
def balanced_parentheses_depth(window: str) -> int:
    """Calculate the depth/balance of a parentheses string."""
    depth = 0
    for char in window:
        if char == '(':
            depth += 1
        elif char == ')':
            depth -= 1
    return depth


def anbn_count(window: str) -> Tuple[int, int]:
    """Count a's and b's in the window."""
    a_count = window.count('a')
    b_count = window.count('b')
    return (a_count, b_count)