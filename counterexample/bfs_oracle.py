"""
BFS (Breadth-First Search) Equivalence Oracle

Systematically explores strings in breadth-first order to find counterexamples.
This provides a simple baseline that guarantees finding the shortest counterexample.
"""

import time
from collections import deque
from typing import Optional, List, Dict, Any, Set

from .base_oracle import EquivalenceOracle
from core.dfa import DFA


class BFSOracle(EquivalenceOracle):
    """
    BFS equivalence oracle for systematic exploration.
    
    Explores strings in order of increasing length, guaranteeing that
    the shortest counterexample will be found first.
    """
    
    def __init__(self, rnn_oracle, alphabet: List[str],
                 max_depth: int = 20,
                 breadth_limit: int = 10000,
                 **kwargs):
        """
        Initialize BFS oracle.
        
        Args:
            rnn_oracle: RNN oracle for membership queries
            alphabet: Input alphabet
            max_depth: Maximum string length to explore
            breadth_limit: Maximum strings to check at each depth
        """
        super().__init__(rnn_oracle, alphabet, **kwargs)
        self.max_depth = max_depth
        self.breadth_limit = breadth_limit
        
        # BFS-specific statistics
        self.total_strings_checked = 0
        self.max_depth_reached = 0
        
    def find_counterexample(self, hypothesis_dfa: DFA, iteration: int,
                          time_limit: Optional[float] = None) -> Optional[str]:
        """
        Find counterexample using breadth-first search.
        
        Args:
            hypothesis_dfa: Current hypothesis DFA
            iteration: L* iteration number
            time_limit: Optional time limit
            
        Returns:
            Counterexample string or None
        """
        print(f"\nBFS Equivalence Query (iteration {iteration})")
        print(f"  Max depth: {self.max_depth}")
        print(f"  Breadth limit: {self.breadth_limit}")
        
        start_time = time.time()
        self.total_queries += 1
        
        # Check empty string first
        strings_checked = 0
        if self._check_string("", hypothesis_dfa):
            # Check if it's blacklisted
            if "" not in self.blacklist:
                self.counterexamples_found += 1
                self.total_time += time.time() - start_time
                self.total_strings_checked += 1
                print(f"  Counterexample found: '' (empty string)")
                return ""
            else:
                print(f"  Skipping blacklisted counterexample: '' (empty string)")
        strings_checked += 1
        
        # BFS exploration
        current_depth = 0
        queue = deque([""])  # Start with empty string
        visited = {""}  # Track visited strings to avoid duplicates
        
        while queue and current_depth < self.max_depth:
            # Check time limit
            if time_limit and (time.time() - start_time) > time_limit:
                print(f"  Time limit reached at depth {current_depth}")
                break
            
            # Process all strings at current depth
            level_size = len(queue)
            depth_strings_checked = 0
            
            for _ in range(level_size):
                if depth_strings_checked >= self.breadth_limit:
                    print(f"  Breadth limit reached at depth {current_depth}")
                    break
                    
                current = queue.popleft()
                
                # Generate children (extend by one symbol)
                for symbol in self.alphabet:
                    child = current + symbol
                    
                    if child not in visited:
                        visited.add(child)
                        queue.append(child)
                        
                        # Check if it's a counterexample
                        if self._check_string(child, hypothesis_dfa):
                            # Check if it's blacklisted
                            if child in self.blacklist:
                                print(f"  Skipping blacklisted counterexample: '{child}'")
                                continue
                                
                            self.counterexamples_found += 1
                            self.total_time += time.time() - start_time
                            self.total_strings_checked += strings_checked + 1
                            self.max_depth_reached = max(self.max_depth_reached, current_depth + 1)
                            
                            print(f"  Counterexample found: '{child}' (length {len(child)})")
                            print(f"  Strings checked: {strings_checked + 1}")
                            print(f"  Depth reached: {current_depth + 1}")
                            return child
                        
                        strings_checked += 1
                        depth_strings_checked += 1
            
            current_depth += 1
            if current_depth <= self.max_depth:
                print(f"  Completed depth {current_depth - 1}, checked {strings_checked} strings total")
        
        # No counterexample found
        self.total_strings_checked += strings_checked
        self.total_time += time.time() - start_time
        self.max_depth_reached = max(self.max_depth_reached, current_depth)
        
        print(f"  No counterexample found after checking {strings_checked} strings")
        print(f"  Maximum depth explored: {current_depth}")
        return None
    
    def _check_string(self, string: str, hypothesis_dfa: DFA) -> bool:
        """
        Check if a string is a counterexample.
        
        Args:
            string: String to check
            hypothesis_dfa: Hypothesis DFA
            
        Returns:
            True if string is a counterexample
        """
        # Convert to list for DFA
        word_list = list(string)
        
        # Check for disagreement
        dfa_accepts = hypothesis_dfa.accepts(word_list)
        rnn_accepts = self.rnn_oracle.classify_word(string)
        
        return dfa_accepts != rnn_accepts
    
    def get_statistics(self) -> Dict[str, Any]:
        """Return oracle statistics."""
        stats = super().get_statistics()
        
        # Add BFS-specific statistics
        stats.update({
            'total_strings_checked': self.total_strings_checked,
            'max_depth': self.max_depth,
            'max_depth_reached': self.max_depth_reached,
            'breadth_limit': self.breadth_limit,
            'avg_strings_per_query': (
                self.total_strings_checked / max(1, self.total_queries)
            )
        })
        
        return stats


class OptimizedBFSOracle(BFSOracle):
    """
    Optimized BFS oracle with pruning based on DFA structure.
    
    Avoids exploring strings that lead to previously visited DFA states
    with the same suffix exploration potential.
    """
    
    def find_counterexample(self, hypothesis_dfa: DFA, iteration: int,
                          time_limit: Optional[float] = None) -> Optional[str]:
        """
        Find counterexample using optimized BFS with state-based pruning.
        """
        print(f"\nOptimized BFS Equivalence Query (iteration {iteration})")
        print(f"  Max depth: {self.max_depth}")
        print(f"  Using DFA state pruning")
        
        start_time = time.time()
        self.total_queries += 1
        strings_checked = 0
        
        # Check empty string
        if self._check_string("", hypothesis_dfa):
            self.counterexamples_found += 1
            self.total_time += time.time() - start_time
            self.total_strings_checked += 1
            print(f"  Counterexample found: '' (empty string)")
            return ""
        strings_checked += 1
        
        # Track (dfa_state, remaining_depth) pairs we've seen
        state_depth_pairs: Set[tuple] = set()
        
        # BFS with state tracking
        queue = deque([("", hypothesis_dfa.q0, 0)])  # (string, dfa_state, depth)
        
        while queue:
            if time_limit and (time.time() - start_time) > time_limit:
                print(f"  Time limit reached")
                break
                
            current_string, current_state, current_depth = queue.popleft()
            
            if current_depth >= self.max_depth:
                continue
            
            # Try each symbol
            for symbol in self.alphabet:
                child_string = current_string + symbol
                
                # Get next DFA state
                if current_state in hypothesis_dfa.delta and symbol in hypothesis_dfa.delta[current_state]:
                    next_state = hypothesis_dfa.delta[current_state][symbol]
                    
                    # Check pruning condition
                    remaining_depth = self.max_depth - (current_depth + 1)
                    state_depth_key = (next_state, remaining_depth)
                    
                    if state_depth_key not in state_depth_pairs:
                        state_depth_pairs.add(state_depth_key)
                        
                        # Check if it's a counterexample
                        if self._check_string(child_string, hypothesis_dfa):
                            # Check if it's blacklisted
                            if child_string in self.blacklist:
                                print(f"  Skipping blacklisted counterexample: '{child_string}'")
                                strings_checked += 1
                                queue.append((child_string, next_state, current_depth + 1))
                                continue
                                
                            self.counterexamples_found += 1
                            self.total_time += time.time() - start_time
                            self.total_strings_checked += strings_checked + 1
                            self.max_depth_reached = max(self.max_depth_reached, current_depth + 1)
                            
                            print(f"  Counterexample found: '{child_string}' (length {len(child_string)})")
                            print(f"  Strings checked: {strings_checked + 1}")
                            return child_string
                        
                        strings_checked += 1
                        queue.append((child_string, next_state, current_depth + 1))
        
        # No counterexample found
        self.total_strings_checked += strings_checked
        self.total_time += time.time() - start_time
        print(f"  No counterexample found after checking {strings_checked} strings")
        return None