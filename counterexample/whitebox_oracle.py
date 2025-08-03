"""
Whitebox counterexample generation using the Weiss et al. (2018) algorithm.

This implements counterexample finding through parallel exploration of 
the hypothesis DFA and the abstracted RNN state space.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Set, Any
from collections import deque
import time
from copy import deepcopy

from .base_oracle import EquivalenceOracle
from core.dfa import DFA
from partitioning.svm_partitioner import SVMPartitioner


class UnrollingInfo:
    """State information during unrolling (matches Weiss's design)."""
    
    def __init__(self, dfa_state: str, path: str, RState: np.ndarray, accepting: bool):
        self.explored = False
        self.dfa_state = dfa_state
        self.paths = [path]
        self.RStates = [RState]
        self.accepting = accepting
        
    def __add__(self, other):
        """Combine information from multiple paths to same partition."""
        res = deepcopy(self)
        res.paths += other.paths
        res.RStates += other.RStates
        return res


class SplitInfo:
    """Information about a required partitioning split."""
    
    def __init__(self, agreeing_RStates=None, conflicted_RState=None):
        self.agreeing_RStates = agreeing_RStates
        self.conflicted_RState = conflicted_RState
        self.has_info = conflicted_RState is not None


class RestartException(Exception):
    """Signal that partitioning has changed and we need to restart."""
    pass


class WhiteboxEquivalenceOracle(EquivalenceOracle):
    """
    Whitebox counterexample generator using parallel exploration.
    
    Core algorithm: Simultaneously explore the hypothesis DFA and the RNN's
    abstracted state space. Counterexamples arise from two sources:
    1. Classification conflicts: RNN and DFA disagree on accepting/rejecting
    2. Clustering conflicts: States that should be equivalent (same partition)
       lead to different DFA states
    
    The partitioning is refined dynamically when conflicts are discovered.
    """
    
    def __init__(self, rnn_oracle, alphabet: List[str],
                 partitioner: SVMPartitioner,
                 starting_examples: List[Tuple[str, bool]] = None,
                 **kwargs):
        """
        Initialize whitebox generator.
        
        Args:
            rnn_oracle: RNN oracle for membership queries
            alphabet: Input alphabet
            partitioner: State space partitioner
            starting_examples: Known examples to check first
        """
        super().__init__(rnn_oracle, alphabet, **kwargs)
        
        self.whiteboxrnn = rnn_oracle
        self.partitioning = partitioner
        self.starting_dict = {ex[0]: ex[1] for ex in (starting_examples or [])}
        
        # Statistics
        self.refinement_count = 0
        self.explored_states = set()
        
    def find_counterexample(self, hypothesis_dfa: DFA, iteration: int,
                          time_limit: Optional[float] = None) -> Optional[str]:
        """
        Find counterexample by parallel exploration.
        
        Args:
            hypothesis_dfa: L* proposed DFA
            iteration: L* iteration number
            time_limit: Optional time limit
            
        Returns:
            Counterexample string or None
        """
        print(f"\nWhitebox Equivalence Query (iteration {iteration})")
        print(f"  Current partitions: {self.partitioning.get_num_partitions()}")
        print(f"  DFA states: {len(hypothesis_dfa.states)}")
        
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
                        return None  # No counterexample found
                        
                    # Process next state
                    counterexample, split = self._process_top_pair()
                    
                    if counterexample:
                        self.counterexamples_found += 1
                        self.total_time += time.time() - start_time
                        print(f"  Counterexample found: '{counterexample}' (length {len(counterexample)})")
                        return counterexample
                    elif split.has_info:
                        # Handle partitioning refinement
                        cluster_being_split = self.partitioning.get_partition(
                            split.agreeing_RStates[0])
                        self.partitioning.refine(
                            split.agreeing_RStates, split.conflicted_RState)
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
        self.new_RStates = [UnrollingInfo(self.proposed_dfa.q0, "", initial_RState, pos)]
        
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
        self.new_RStates_backup = new_info  # For potential reprocessing
        
        # Get partition for this RNN state
        new_cluster = self.partitioning.get_partition(new_info.RStates[0])
        
        # Process the state
        counterexample, split = self._process_new_state_except_children(
            new_cluster, new_info)
            
        # If no conflicts, we should add children
        if not counterexample and not split.has_info:
            self._add_children_states(new_cluster)
            
        return counterexample, split
        
    def _process_new_state_except_children(self, new_cluster: int, 
                                         new_info: UnrollingInfo) -> Tuple[Optional[str], SplitInfo]:
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
                split = SplitInfo(
                    agreeing_RStates=old_info.RStates,
                    conflicted_RState=new_info.RStates[0])
        else:
            # No conflicts - store the information
            self.cluster_information[new_cluster] = full_info
            
        return counterexample, split
        
    def _counterexample_from_classification_conflict(self, state_info: UnrollingInfo) -> Optional[str]:
        """Extract counterexample from classification conflict."""
        # Return shortest path that demonstrates the conflict
        path = min(state_info.paths, key=len)
        
        # Verify this is a real counterexample
        # Sometimes RNN state labels can be inconsistent with actual classification
        rnn_label = self.whiteboxrnn.classify_word(path)
        dfa_accepts = state_info.dfa_state in self.proposed_dfa.F
        
        if rnn_label != dfa_accepts:
            return path
        else:
            # False conflict - RNN state label was wrong
            return None
        
    def _counterexample_from_cluster_conflict(self, old_info: UnrollingInfo, 
                                            new_info: UnrollingInfo) -> Optional[str]:
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
        """Add children states from a cluster."""
        state_info = self.cluster_information[cluster]
        if not state_info.explored:
            # we explore a state only the first time we successfully visit and process it
            RState = state_info.RStates[0]
            state_info.explored = True
            for char in self.proposed_dfa.alphabet:
                next_RState, accepting = self.whiteboxrnn.get_next_RState(RState, char)
                path = state_info.paths[0] + char
                # we only ever explore a state the first
                # time we find it, so, with the first path in its list of reaching paths
                next_dfa_state = self.proposed_dfa.delta[state_info.dfa_state][char]
                self.new_RStates.append(
                    UnrollingInfo(next_dfa_state, path, next_RState, accepting))
                    
    def _split_was_clean(self, old_cluster: int, split: SplitInfo) -> bool:
        """Check if partitioning split was clean."""
        # Check that agreeing states stayed in same cluster
        new_states_given_to_agreeing = list(set(
            [self.partitioning.get_partition(vec) for vec in split.agreeing_RStates]))
            
        return (self.partitioning.refinement_doesnt_hurt_other_clusters and
                new_states_given_to_agreeing == [old_cluster] and
                self.partitioning.get_partition(split.conflicted_RState) != old_cluster)
                
    def get_statistics(self) -> Dict[str, Any]:
        """Return oracle statistics."""
        # Get base statistics
        stats = super().get_statistics()
        
        # Add whitebox-specific statistics
        stats.update({
            'partitions': self.partitioning.get_num_partitions(),
            'refinements': self.refinement_count,
            'states_explored': len(self.explored_states),
            'strategy': 'svm_partitioning'
        })
        
        return stats