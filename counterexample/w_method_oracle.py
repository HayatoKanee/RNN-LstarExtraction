"""
W-method Equivalence Oracle

Implements Chow's W-method for equivalence testing, which systematically
tests the hypothesis DFA against the target using a characterization set.
"""

import itertools
import time
from typing import Optional, List, Set, Dict, Any

from .base_oracle import EquivalenceOracle
from core.dfa import DFA


class WMethodOracle(EquivalenceOracle):
    """
    W-method equivalence oracle for systematic testing.
    
    The W-method assumes an upper bound on the number of states in the
    target automaton and performs exhaustive testing up to a certain depth.
    """
    
    def __init__(self, rnn_oracle, alphabet: List[str],
                 max_target_states: int = 10,
                 **kwargs):
        """
        Initialize W-method oracle.
        
        Args:
            rnn_oracle: RNN oracle for membership queries
            alphabet: Input alphabet
            max_target_states: Upper bound on number of states in target
        """
        super().__init__(rnn_oracle, alphabet, **kwargs)
        self.max_target_states = max_target_states
        self.cache = set()  # Cache tested sequences
        
        # Statistics specific to W-method
        self.total_test_strings = 0
        
    def find_counterexample(self, hypothesis_dfa: DFA, iteration: int,
                          time_limit: Optional[float] = None) -> Optional[str]:
        """
        Find counterexample using W-method.
        
        Args:
            hypothesis_dfa: Current hypothesis DFA
            iteration: L* iteration number
            time_limit: Optional time limit
            
        Returns:
            Counterexample string or None
            
        Raises:
            RuntimeError: If hypothesis exceeds max_states limit
        """
        print(f"\nW-Method Equivalence Query (iteration {iteration})")
        print(f"  Hypothesis states: {len(hypothesis_dfa.states)}")
        
        
        # Calculate depth based on hypothesis size
        depth = self.max_target_states + 1 - len(hypothesis_dfa.states)
        depth = max(0, depth)  # Ensure non-negative
        print(f"  Test depth: {depth} (max_target={self.max_target_states})")
        
        start_time = time.time()
        self.total_queries += 1
        
        # Step 1: Compute transition cover
        # Covers every transition of the hypothesis at least once
        transition_cover = self._compute_transition_cover(hypothesis_dfa)
        print(f"  Transition cover size: {len(transition_cover)}")
        
        # Step 2: Compute characterization set W
        # W distinguishes between all pairs of states
        W = self._compute_characterization_set(hypothesis_dfa)
        print(f"  Characterization set size: {len(W)}")
        
        # Step 3: Generate test set
        # Test set = transition_cover · Σ^[0..depth] · W
        test_strings = self._generate_test_set(transition_cover, W, depth)
        print(f"  Test set size: {len(test_strings)}")
        
        # Step 4: Check each test string
        strings_checked = 0
        for test_string in sorted(test_strings, key=len):  # Check shorter strings first
            if time_limit and (time.time() - start_time) > time_limit:
                print(f"  Time limit reached after {strings_checked} strings")
                break
                
            strings_checked += 1
            
            # Convert to list for DFA
            word_list = list(test_string)
            
            # Check for disagreement
            dfa_accepts = hypothesis_dfa.accepts(word_list)
            rnn_accepts = self.rnn_oracle.classify_word(test_string)
            
            if dfa_accepts != rnn_accepts:
                self.counterexamples_found += 1
                self.total_time += time.time() - start_time
                self.total_test_strings += strings_checked
                
                print(f"  Counterexample found: '{test_string}' (length {len(test_string)})")
                print(f"  Strings checked: {strings_checked}/{len(test_strings)}")
                return test_string
        
        # No counterexample found
        self.total_test_strings += strings_checked
        self.total_time += time.time() - start_time
        print(f"  No counterexample found after checking {strings_checked} strings")
        return None
    
    def _compute_state_cover(self, dfa: DFA) -> Dict[str, str]:
        """
        Compute access sequences for all states.
        
        Returns a mapping from state to shortest string that reaches it.
        """
        access_sequences = {dfa.q0: ""}  # Initial state reached by empty string
        
        # BFS to find shortest paths to all states
        queue = [(dfa.q0, "")]
        
        while queue and len(access_sequences) < len(dfa.states):
            current_state, current_path = queue.pop(0)
            
            # Try each symbol
            for symbol in self.alphabet:
                if current_state in dfa.delta and symbol in dfa.delta[current_state]:
                    next_state = dfa.delta[current_state][symbol]
                    next_path = current_path + symbol
                    
                    if next_state not in access_sequences:
                        access_sequences[next_state] = next_path
                        queue.append((next_state, next_path))
        
        return access_sequences
    
    def _compute_transition_cover(self, dfa: DFA) -> Set[str]:
        """
        Compute a transition cover.
        
        Returns a set of strings that covers every transition at least once.
        """
        # First get access sequences for all states
        access_sequences = self._compute_state_cover(dfa)
        
        # Transition cover includes access to each state + each symbol
        transition_cover = set()
        for state in dfa.states:
            access_seq = access_sequences.get(state, "")
            for symbol in self.alphabet:
                transition_cover.add(access_seq + symbol)
        
        return transition_cover
    
    def _compute_characterization_set(self, dfa: DFA) -> Set[str]:
        """
        Compute a characterization set W.
        
        W is a set of strings that distinguish between every pair of
        inequivalent states in the DFA.
        """
        W = {""}  # Always include empty string
        
        # For each pair of distinct states
        state_list = list(dfa.states)
        for i in range(len(state_list)):
            for j in range(i + 1, len(state_list)):
                state1, state2 = state_list[i], state_list[j]
                
                # Find a string that distinguishes them
                distinguisher = self._find_distinguishing_string(dfa, state1, state2)
                if distinguisher is not None:
                    W.add(distinguisher)
        
        return W
    
    def _find_distinguishing_string(self, dfa: DFA, state1: str, state2: str) -> Optional[str]:
        """
        Find a string that distinguishes between two states.
        
        Returns shortest string w such that exactly one of state1·w and state2·w
        is accepting.
        """
        # Check if states have different acceptance
        if (state1 in dfa.F) != (state2 in dfa.F):
            return ""
        
        # BFS to find shortest distinguishing string
        queue = [("", state1, state2)]
        visited = {(state1, state2)}
        
        while queue:
            path, s1, s2 = queue.pop(0)
            
            # Try each symbol
            for symbol in self.alphabet:
                if s1 in dfa.delta and symbol in dfa.delta[s1] and \
                   s2 in dfa.delta and symbol in dfa.delta[s2]:
                    next_s1 = dfa.delta[s1][symbol]
                    next_s2 = dfa.delta[s2][symbol]
                    next_path = path + symbol
                    
                    # Check if states differ in acceptance
                    if (next_s1 in dfa.F) != (next_s2 in dfa.F):
                        return next_path
                    
                    # Continue search if not visited
                    if (next_s1, next_s2) not in visited:
                        visited.add((next_s1, next_s2))
                        queue.append((next_path, next_s1, next_s2))
        
        return None  # States are equivalent
    
    def _generate_test_set(self, P: Set[str], W: Set[str], depth: int) -> Set[str]:
        """
        Generate the W-method test set.
        
        Test set = P · Σ^[0..depth] · W
        """
        test_set = set()
        
        # Generate all strings up to length depth
        sigma_star_m = {""}
        for length in range(1, depth + 1):
            for string in itertools.product(self.alphabet, repeat=length):
                sigma_star_m.add(''.join(string))
        
        # Combine P · Σ* · W
        for p in P:
            for s in sigma_star_m:
                for w in W:
                    test_set.add(p + s + w)
        
        return test_set
    
    def get_statistics(self) -> Dict[str, Any]:
        """Return oracle statistics."""
        stats = super().get_statistics()
        
        # Add W-method specific statistics
        stats.update({
            'total_test_strings': self.total_test_strings,
            'max_target_states': self.max_target_states,
            'avg_strings_per_query': (
                self.total_test_strings / max(1, self.total_queries)
            )
        })
        
        return stats