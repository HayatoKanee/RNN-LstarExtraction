"""
Random W_p Method Equivalence Oracle.

Implements the Random W_p Method as described in "Complementing Model
Learning with Mutation-Based Fuzzing" by Rick Smetsers et al.

The key idea is to randomly sample test sequences instead of exhaustively
testing all possibilities
"""

import random
import time
from typing import List, Optional, Set, Tuple
from .base_oracle import EquivalenceOracle
from core.dfa import DFA


class RandomWpOracle(EquivalenceOracle):
    """
    Random W_p Method oracle for probabilistic equivalence testing.
    
    Instead of exhaustive testing like the standard W_p method, this approach
    randomly samples test sequences, making it more scalable for larger systems
    while still providing good coverage.
    """
    
    def __init__(self, rnn_oracle, alphabet: List[str],
                 min_length: int = 3,
                 expected_length: int = 11,
                 num_tests: int = 10000,
                 max_total_length: int = 50):
        """
        Initialize Random W_p oracle.
        
        Args:
            rnn_oracle: Oracle for membership queries
            alphabet: Input alphabet
            min_length: Minimum length of the middle part (infix)
            expected_length: Expected length of the middle part (geometric distribution)
            num_tests: Number of random tests to perform per equivalence query
            max_total_length: Maximum total string length to test
        """
        super().__init__(rnn_oracle, alphabet)
        self.min_length = min_length
        self.expected_length = expected_length
        self.num_tests = num_tests
        self.max_total_length = max_total_length
        self.test_count = 0
        
    def find_counterexample(self, hypothesis: DFA, 
                           iteration: int,
                           time_limit: Optional[float] = None) -> Optional[str]:
        """
        Find counterexample using random sampling of test sequences.
        
        The algorithm:
        1. Compute state-specific discriminators for each state
        2. For each test:
           a. Sample a prefix uniformly from state cover
           b. Sample a middle part length from geometric distribution
           c. Generate random middle part of that length
           d. Sample a suffix from state-specific discriminators
           e. Test if this sequence is a counterexample
           
        Args:
            hypothesis: Current hypothesis DFA
            iteration: Current iteration number
            time_limit: Optional time limit
            
        Returns:
            Counterexample string or None
        """
        print(f"\nRandom W_p Equivalence Query (iteration {iteration})")
        print(f"  Number of tests: {self.num_tests}")
        print(f"  Min infix length: {self.min_length}")
        print(f"  Expected infix length: {self.expected_length}")
        print(f"  Max total length: {self.max_total_length}")
        
        start_time = time.time()
        
        # Note: state cover computation is kept for completeness but not directly used
        # since we get access sequences dynamically for each sampled state
        _ = self._get_state_cover(hypothesis)
        
        # Compute state-specific discriminators
        state_discriminators = self._compute_state_discriminators(hypothesis)
        
        # Perform random tests
        tests_performed = 0
        for test_num in range(self.num_tests):
            # Check time limit
            if time_limit and (time.time() - start_time) > time_limit:
                print(f"  Time limit reached after {tests_performed} tests")
                break
                
            # 1. Sample prefix uniformly from state cover
            prefix_state = random.choice(list(hypothesis.states))
            prefix = self._get_access_sequence(hypothesis, prefix_state)
            
            # 2. Generate random middle part (infix)
            # Use geometric distribution for length
            infix_length = self.min_length
            while random.random() < (self.expected_length - self.min_length) / (self.expected_length + 1):
                infix_length += 1
            
            # Generate random infix of that length
            infix = [random.choice(self.alphabet) for _ in range(infix_length)]
            
            # 3. Determine which state we reach after prefix + infix
            current_state = hypothesis.q0
            for symbol in prefix + infix:
                if symbol in hypothesis.delta.get(current_state, {}):
                    current_state = hypothesis.delta[current_state][symbol]
                else:
                    # Invalid transition, skip this test
                    continue
            
            # 4. Sample suffix from state-specific discriminators
            if current_state in state_discriminators and state_discriminators[current_state]:
                suffix = random.choice(state_discriminators[current_state])
            else:
                # No discriminators for this state, use empty suffix
                suffix = []
            
            # Construct full test sequence
            test_sequence = prefix + infix + suffix
            test_string = ''.join(test_sequence)
            
            # Skip if total length exceeds max_total_length
            if len(test_string) > self.max_total_length:
                continue
            
            # Test if it's a counterexample
            self.test_count += 1
            tests_performed += 1
            
            # Get hypothesis output
            hyp_output = hypothesis.accepts(test_sequence)
            
            # Get RNN output
            rnn_output = self.rnn_oracle.classify_word(test_string)
            
            if hyp_output != rnn_output:
                print(f"  Counterexample found: '{test_string}' (length {len(test_string)})")
                print(f"  Test number: {test_num + 1}")
                print(f"  Prefix: '{prefix}', Infix: '{infix}', Suffix: '{suffix}'")
                self.counterexamples_found += 1
                return test_string
        
        print(f"  No counterexample found after {tests_performed} tests")
        return None
    
    def _get_state_cover(self, hypothesis: DFA) -> Set[Tuple[str, ...]]:
        """Get state cover - access sequences for all states."""
        state_cover = set()
        
        # Use BFS to find shortest access sequences
        from collections import deque
        visited = {hypothesis.q0}
        queue = deque([(hypothesis.q0, [])])
        state_cover.add(())  # Empty sequence for initial state
        
        while queue:
            state, path = queue.popleft()
            
            for symbol in self.alphabet:
                if symbol in hypothesis.delta.get(state, {}):
                    next_state = hypothesis.delta[state][symbol]
                    if next_state not in visited:
                        visited.add(next_state)
                        access_seq = path + [symbol]
                        state_cover.add(tuple(access_seq))
                        queue.append((next_state, access_seq))
        
        return state_cover
    
    def _get_access_sequence(self, hypothesis: DFA, target_state: str) -> List[str]:
        """Get access sequence for a specific state."""
        # Use BFS to find shortest path
        from collections import deque
        
        if target_state == hypothesis.q0:
            return []
        
        visited = {hypothesis.q0}
        queue = deque([(hypothesis.q0, [])])
        
        while queue:
            state, path = queue.popleft()
            
            for symbol in self.alphabet:
                if symbol in hypothesis.delta.get(state, {}):
                    next_state = hypothesis.delta[state][symbol]
                    if next_state == target_state:
                        return path + [symbol]
                    if next_state not in visited:
                        visited.add(next_state)
                        queue.append((next_state, path + [symbol]))
        
        # Should not reach here for a valid DFA
        return []
    
    def _compute_state_discriminators(self, hypothesis: DFA) -> dict:
        """
        Compute state-specific discriminators.
        
        For each state, find sequences that distinguish it from all other states.
        """
        discriminators = {}
        
        for state in hypothesis.states:
            state_discs = []
            
            # Find discriminators that distinguish this state from all others
            for other_state in hypothesis.states:
                if state != other_state:
                    # Find a sequence that distinguishes state from other_state
                    disc = self._find_discriminator(hypothesis, state, other_state)
                    if disc:
                        state_discs.append(disc)
            
            discriminators[state] = state_discs
        
        return discriminators
    
    def _find_discriminator(self, hypothesis: DFA, state1: str, state2: str, 
                           max_length: int = 10) -> Optional[List[str]]:
        """
        Find a sequence that distinguishes two states.
        
        Uses BFS to find the shortest distinguishing sequence.
        """
        from collections import deque
        
        # Check if states have different acceptance
        if (state1 in hypothesis.F) != (state2 in hypothesis.F):
            return []  # Empty sequence distinguishes them
        
        # BFS to find distinguishing sequence
        queue = deque([((state1, state2), [])])
        visited = {(state1, state2)}
        
        while queue:
            (s1, s2), path = queue.popleft()
            
            if len(path) >= max_length:
                break
            
            for symbol in self.alphabet:
                next_s1 = hypothesis.delta.get(s1, {}).get(symbol)
                next_s2 = hypothesis.delta.get(s2, {}).get(symbol)
                
                if next_s1 is None or next_s2 is None:
                    continue
                
                new_path = path + [symbol]
                
                # Check if this path distinguishes the original states
                acc1 = next_s1 in hypothesis.F
                acc2 = next_s2 in hypothesis.F
                
                if acc1 != acc2:
                    return new_path
                
                state_pair = (next_s1, next_s2)
                if state_pair not in visited:
                    visited.add(state_pair)
                    queue.append((state_pair, new_path))
        
        return None
    
    def get_statistics(self) -> dict:
        """Get oracle statistics."""
        stats = super().get_statistics()
        stats['test_count'] = self.test_count
        stats['min_length'] = self.min_length
        stats['expected_length'] = self.expected_length
        stats['num_tests'] = self.num_tests
        stats['max_total_length'] = self.max_total_length
        return stats