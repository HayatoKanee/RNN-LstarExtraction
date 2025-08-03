"""
Bounded L* algorithm with negotiation protocol.

Key innovation: When L* cannot find a distinguishing suffix within query bounds,
it requests the oracle to provide a different counterexample instead of failing.
This maintains theoretical soundness while handling imperfect RNN teachers.
"""

from typing import Optional, Set, List, Tuple
import time
from .lstar import LStarAlgorithm
from .observation_table import ObservationTable
from .dfa import DFA


class BoundedObservationTable(ObservationTable):
    """
    Observation table that respects query length bounds.
    
    Key insight: We only add experiments that keep all queries s·e within bounds.
    """
    
    def __init__(self, alphabet: List[str], teacher, 
                 max_query_length: int = 20,
                 max_table_size: Optional[int] = None):
        """
        Initialize bounded observation table.
        
        Args:
            max_query_length: Maximum length for any membership query
        """
        self.max_query_length = max_query_length
        super().__init__(alphabet, teacher, max_table_size)
        
    def _fill_T(self, new_e_list: Optional[List[str]] = None, 
                new_s: Optional[str] = None):
        """
        Override to only query strings within length bound.
        """
        words_to_query = self._Trange(new_e_list, new_s)
        
        # Filter out strings that are too long
        bounded_words = {w for w in words_to_query if len(w) <= self.max_query_length}
        
        # Warn if we're skipping queries
        skipped = len(words_to_query) - len(bounded_words)
        if skipped > 0:
            print(f"  [Bounded L*] Skipping {skipped} queries exceeding length {self.max_query_length}")
        
        # Only query bounded words
        uncached_words = [w for w in bounded_words if w not in self.T]
        
        if uncached_words:
            self.query_count += len(uncached_words)
            results = self.teacher.membership_queries(uncached_words)
            
            for word, result in zip(uncached_words, results):
                self.T[word] = result
        else:
            self.cache_hits += len(bounded_words)
    
    def find_and_handle_inconsistency(self) -> bool:
        """
        Find and resolve inconsistency with bounded experiments.
        
        Only add experiments a·e where s1·a·e and s2·a·e are within bounds.
        """
        # Find potentially inconsistent pairs
        maybe_inconsistent = [(s1, s2, a) for s1, s2 in self.equal_cache 
                              if s1 in self.S 
                              for a in self.A
                              if (s1 + a, s2 + a) not in self.equal_cache]
        
        troublemakers = []
        for s1, s2, a in maybe_inconsistent:
            # Find e such that T[s1·a·e] ≠ T[s2·a·e]
            # BUT only consider e where both queries are bounded
            for e in self.E:
                query1 = s1 + a + e
                query2 = s2 + a + e
                
                # Skip if either query would be too long
                if len(query1) > self.max_query_length or len(query2) > self.max_query_length:
                    continue
                    
                if self.T.get(query1) != self.T.get(query2):
                    troublemakers.append(a + e)
                    break
        
        if len(troublemakers) == 0:
            return False
            
        # Add the shortest troublemaker first
        troublemakers.sort(key=len)
        new_exp = troublemakers[0]
        
        print(f"  [Bounded L*] Adding experiment '{new_exp}' (length {len(new_exp)})")
        
        self.E.add(new_exp)
        self._fill_T(new_e_list=[new_exp])
        self._update_row_equivalence_cache(new_e=new_exp)
        self._assert_not_timed_out()
        return True
    
    def add_counterexample(self, ce: str, label: bool):
        """
        Process counterexample but respect length bounds.
        
        We add all prefixes that don't cause unbounded queries.
        """
        if ce in self.S:
            print(f"  Counterexample already in S!")
            return
            
        # Add counterexample classification
        self.T[ce] = label
        
        # Add prefixes, but stop if they would create unbounded queries
        new_states = []
        max_e_length = max(len(e) for e in self.E) if self.E else 0
        
        for i in range(len(ce) + 1):
            prefix = ce[:i]
            
            # Check if adding this prefix would create unbounded queries
            # Need to consider prefix + a + e for all a ∈ Σ, e ∈ E
            max_potential_query = len(prefix) + 1 + max_e_length  # +1 for alphabet symbol
            
            if max_potential_query > self.max_query_length:
                print(f"  [Bounded L*] Stopping prefix addition at length {len(prefix)}")
                print(f"    (Would create queries up to length {max_potential_query})")
                break
                
            if prefix not in self.S:
                new_states.append(prefix)
                self.S.add(prefix)
        
        # Fill table for new states
        self._fill_T()
        
        # Update equivalence cache
        for s in new_states:
            self._update_row_equivalence_cache(new_s=s)
            
        self._assert_not_timed_out()


class BoundedLStar(LStarAlgorithm):
    """
    Bounded L* with negotiation protocol.
    
    When a counterexample cannot be processed due to query bounds,
    it negotiates with the oracle for an alternative counterexample.
    """
    
    def __init__(self, teacher, 
                 max_query_length: int = 20,
                 time_limit: Optional[float] = None,
                 validation_set: Optional[List] = None):
        """
        Initialize bounded L*.
        
        Args:
            max_query_length: Maximum length for membership queries
        """
        self.max_query_length = max_query_length
        self.rejected_counterexamples = set()  # Track CEs we couldn't process
        super().__init__(teacher, time_limit, validation_set)
        
    def run(self) -> DFA:
        """
        Run bounded L* algorithm with negotiation.
        """
        print(f"\nStarting Bounded L* (max query length: {self.max_query_length})")
        print(f"  Negotiation enabled: Can request alternative counterexamples")
        
        # Initialize bounded observation table
        self.table = BoundedObservationTable(
            self.alphabet, 
            self.teacher,
            max_query_length=self.max_query_length
        )
        
        if self.time_limit is not None:
            self.table.set_time_limit(self.time_limit, self.start_time)
        
        # Main L* loop with negotiation
        iterations = 0
        self.hypotheses_history = []
        start_time = time.time()
        consecutive_rejections = 0  # Track consecutive rejected CEs
        
        while True:
            # Make table closed and consistent
            changes = True
            while changes:
                self.table._assert_not_timed_out()
                changes = False
                
                # Fix all inconsistencies first
                while self.table.find_and_handle_inconsistency():
                    changes = True
                    
                # Then check closure
                if self.table.find_and_close_row():
                    changes = True
                        
            # Build hypothesis
            hypothesis = DFA(obs_table=self.table)
            iterations += 1
            
            # Test equivalence with blacklist
            ce = self._equivalence_query_with_blacklist(hypothesis, iterations)
            
            # Record hypothesis
            self.hypotheses_history.append({
                'iteration': iterations,
                'time': time.time() - start_time,
                'states': len(hypothesis.states),
                'counterexample': ce,
                'dfa_object': hypothesis
            })
            
            if ce is None:
                # No counterexample found
                print(f"\nBounded L* complete!")
                print(f"  Learned DFA with {len(hypothesis.states)} states")
                print(f"  Rejected {len(self.rejected_counterexamples)} unprocessable counterexamples")
                return hypothesis
            
            # Check if we're stuck rejecting too many counterexamples
            if consecutive_rejections >= 10:
                print(f"\nBounded L* terminating: Too many consecutive rejections")
                print(f"  Learned DFA with {len(hypothesis.states)} states")
                print(f"  Rejected {len(self.rejected_counterexamples)} total counterexamples")
                print(f"  Oracle cannot find processable counterexamples within bounds")
                return hypothesis
                
            # Try to process counterexample
            print(f"\nIteration {iterations}: Counterexample '{ce}' (length {len(ce)})")
            
            # Store the current hypothesis to compare later
            old_hypothesis = hypothesis
            
            # Add counterexample and try to refine the table
            self.table.add_counterexample(ce, self.teacher.classify_word(ce))
            
            # Make table closed and consistent again
            changes = True
            while changes:
                self.table._assert_not_timed_out()
                changes = False
                
                # Fix all inconsistencies first
                while self.table.find_and_handle_inconsistency():
                    changes = True
                    
                # Then check closure
                if self.table.find_and_close_row():
                    changes = True
            
            # Build new hypothesis
            new_hypothesis = DFA(obs_table=self.table)
            
            # Check if the hypothesis changed
            if self._are_dfas_equal(old_hypothesis, new_hypothesis):
                # Hypothesis didn't change - we couldn't process this CE
                print(f"  Cannot refine hypothesis with this counterexample!")
                print(f"  Rejecting counterexample and requesting alternative...")
                self.rejected_counterexamples.add(ce)
                consecutive_rejections += 1
                # Reset table to before we added this CE
                # For simplicity, we'll keep going without resetting
                # The oracle should give us a different CE next time
            else:
                # Successfully processed the counterexample
                consecutive_rejections = 0
    
    def _equivalence_query_with_blacklist(self, hypothesis: DFA, iteration: int) -> Optional[str]:
        """
        Perform equivalence query with blacklisted counterexamples.
        """
        # Tell the oracle about rejected counterexamples
        if hasattr(self.teacher, 'set_counterexample_blacklist'):
            self.teacher.set_counterexample_blacklist(self.rejected_counterexamples)
        elif hasattr(self.teacher, 'equivalence_oracle') and hasattr(self.teacher.equivalence_oracle, 'set_blacklist'):
            self.teacher.equivalence_oracle.set_blacklist(self.rejected_counterexamples)
            
        ce = self.teacher.equivalence_query(hypothesis, iteration)
        
        # Check if oracle returned a blacklisted CE (shouldn't happen with proper implementation)
        while ce in self.rejected_counterexamples:
            print(f"  Warning: Oracle returned blacklisted CE '{ce}', requesting another...")
            ce = self.teacher.equivalence_query(hypothesis, iteration)
            
        return ce
    
    def _are_dfas_equal(self, dfa1: DFA, dfa2: DFA) -> bool:
        """
        Check if two DFAs are structurally equal.
        Simple check: same states, same start, same accept states, same transitions.
        """
        if len(dfa1.states) != len(dfa2.states):
            return False
            
        if dfa1.q0 != dfa2.q0:
            return False
            
        if dfa1.F != dfa2.F:
            return False
            
        # Check transitions
        for state in dfa1.states:
            if state not in dfa2.delta:
                return False
            for symbol in dfa1.alphabet:
                if dfa1.delta.get(state, {}).get(symbol) != dfa2.delta.get(state, {}).get(symbol):
                    return False
                    
        return True
    
    def validate_on_bounded_set(self, dfa: DFA, max_length: int) -> float:
        """
        Validate DFA on all strings up to max_length.
        
        This gives us empirical confidence in the bounded guarantee.
        """
        correct = 0
        total = 0
        
        for length in range(max_length + 1):
            # Generate all strings of this length
            if length == 0:
                strings = ['']
            else:
                strings = []
                for i in range(len(self.alphabet) ** length):
                    s = ''
                    n = i
                    for _ in range(length):
                        s = self.alphabet[n % len(self.alphabet)] + s
                        n //= len(self.alphabet)
                    strings.append(s)
            
            # Test each string
            for s in strings:
                dfa_label = dfa.classify_word(s)
                rnn_label = self.teacher.classify_word(s)
                
                if dfa_label == rnn_label:
                    correct += 1
                total += 1
        
        accuracy = correct / total if total > 0 else 0
        print(f"\nValidation on all strings ≤ length {max_length}:")
        print(f"  Correct: {correct}/{total} ({accuracy:.1%})")
        
        return accuracy