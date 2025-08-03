"""
L* Algorithm implementation based on Angluin (1987).

Algorithm learns a minimal DFA for an unknown regular language using membership and
equivalence queries with polynomial complexity in the number of states and the length
of counterexamples.
"""

from typing import Optional, Dict, Any, List, Tuple
import time
from .dfa import DFA
from .observation_table import ObservationTable, TableTimedOut


class LStarAlgorithm:
    """L* learning algorithm implementation."""
    
    def __init__(self, teacher, time_limit: Optional[float] = None,
                 validation_set: Optional[List[Tuple[str, bool]]] = None):
        """
        Initialize L* learner.
        
        Args:
            teacher: Oracle providing membership/equivalence queries
            time_limit: Optional time bound
            validation_set: Optional validation examples for tracking accuracy
        """
        self.teacher = teacher
        self.alphabet = teacher.alphabet
        self.time_limit = time_limit
        self.start_time = time.time()
        self.validation_set = validation_set or []
        
        # Initialize observation table
        self.table = ObservationTable(self.alphabet, teacher)
        
        # Track best hypothesis found so far
        self.best_dfa = None
        self.best_accuracy = 0.0
        self.best_size = float('inf')
        self.hypotheses_history = []
        
        # Statistics
        self.iterations = 0
        self.counterexamples = []
        self.refinement_times = []
        self.equivalence_times = []
        
    def run(self) -> DFA:
        """
        Execute L* learning algorithm.
        
        Returns:
            Learned DFA (exact if no timeout, best approximation otherwise)
            
        Raises:
            TableTimedOut: If time limit exceeded
        """
        # Set time limits
        if self.time_limit:
            self.table.set_time_limit(self.time_limit, self.start_time)
            self.teacher.set_time_limit(self.time_limit, self.start_time)
            
        try:
            while True:
                self.iterations += 1
                refinement_start = time.time()
                
                # Phase 1: Make table closed and consistent
                self._refine_table()
                
                refinement_time = time.time() - refinement_start
                self.refinement_times.append(refinement_time)
                
                # Phase 2: Construct hypothesis DFA
                hypothesis = DFA(obs_table=self.table)
                
                print(f"Iteration {self.iterations}: "
                      f"Constructed DFA with {len(hypothesis.states)} states "
                      f"(refinement: {refinement_time:.2f}s)")
                
                # Update best hypothesis tracking
                self._update_best_hypothesis(hypothesis)
                
                # Phase 3: Equivalence query
                equiv_start = time.time()
                try:
                    counterexample = self.teacher.equivalence_query(
                        hypothesis, 
                        iteration=self.iterations
                    )
                except RuntimeError as e:
                    # Handle state limit exceeded or other runtime errors
                    print(f" Equivalence oracle error: {e}")
                    print(f"  Stopping with current hypothesis (states: {len(hypothesis.states)})")
                    return hypothesis
                    
                equiv_time = time.time() - equiv_start
                self.equivalence_times.append(equiv_time)
                
                if counterexample is None:
                    # Exact DFA found!
                    print(f"Exact DFA learned in {self.iterations} iterations")
                    return hypothesis
                    
                # Phase 4: Process counterexample
                print(f"  Counterexample found: '{counterexample}' "
                      f"(length {len(counterexample)}, search: {equiv_time:.2f}s)")
                
                self.counterexamples.append(counterexample)
                ce_label = self.teacher.classify_word(counterexample)
                
                # Update the last hypothesis with counterexample info
                if self.hypotheses_history:
                    self.hypotheses_history[-1]['counterexample'] = {
                        'string': counterexample,
                        'length': len(counterexample),
                        'rnn_label': ce_label,
                        'dfa_label': hypothesis.accepts(list(counterexample)),
                        'found_by': self.teacher.oracle_type
                    }
                
                self.table.add_counterexample(counterexample, ce_label)
                
                # DEBUG: Print table state after adding counterexample
                # Uncomment below for debugging observation table growth
                # if len(self.counterexamples) <= 3 or len(counterexample) > 15:
                #     self.table.debug_table_state()
                
                # Check timeout
                if self._check_timeout():
                    print(f" Time limit reached, returning best hypothesis")
                    return self.best_dfa
                    
        except TableTimedOut:
            print(f" Table operations timed out, returning best approximation")
            return self.best_dfa
            
    def _refine_table(self):
        """
        Make observation table closed and consistent.
        
        Follows standard L* refinement loop with optimizations:
        - Handle all inconsistencies before checking closure
        - Batch operations where possible
        """
        changes = True
        while changes:
            changes = False
            
            # Fix all inconsistencies first
            while self.table.find_and_handle_inconsistency():
                changes = True
                
            # Then check closure
            if self.table.find_and_close_row():
                changes = True
                
    def _update_best_hypothesis(self, hypothesis: DFA):
        """
        Update best hypothesis found so far.
        
        Selection criteria:
        1. Validation accuracy (if validation set provided)
        2. DFA size (prefer smaller when accuracy is equal)
        """
        current_accuracy = 0.0
        current_size = len(hypothesis.states)
        
        # Compute validation accuracy if we have data
        if self.validation_set:
            correct = 0
            for word, label in self.validation_set:
                if hypothesis.accepts(list(word)) == label:
                    correct += 1
            current_accuracy = correct / len(self.validation_set)
            
        # Update best if this is better
        update_best = False
        if self.best_dfa is None:
            update_best = True
        elif current_accuracy > self.best_accuracy:
            update_best = True
        elif current_accuracy == self.best_accuracy and current_size < self.best_size:
            update_best = True
            
        if update_best:
            self.best_dfa = hypothesis
            self.best_accuracy = current_accuracy
            self.best_size = current_size
            
            if self.validation_set:
                print(f"  New best: accuracy={self.best_accuracy:.1%}, size={self.best_size}")
                
        # Track all hypotheses for analysis
        self.hypotheses_history.append({
            'iteration': self.iterations,
            'time': time.time() - self.start_time,
            'size': current_size,
            'accuracy': current_accuracy,
            'is_best': update_best,
            'states': len(hypothesis.states),
            'dfa_summary': {
                'states': list(hypothesis.states),
                'accept_states': list(hypothesis.F),
                'start_state': hypothesis.q0
            },
            'dfa_object': hypothesis,  # Store the actual DFA object
            'counterexample': None  # Will be filled in if this hypothesis is rejected
        })
        
    def _check_timeout(self) -> bool:
        """Check if time limit exceeded."""
        if self.time_limit is None:
            return False
        return time.time() - self.start_time > self.time_limit
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Return learning statistics.
        
        Returns:
            Dictionary with performance metrics
        """
        total_time = time.time() - self.start_time
        
        stats = {
            "iterations": self.iterations,
            "total_time": total_time,
            "final_states": len(self.best_dfa.states) if self.best_dfa else 0,
            "best_accuracy": self.best_accuracy,
            "hypotheses_tested": len(self.hypotheses_history),
            "counterexamples": len(self.counterexamples),
            "avg_ce_length": sum(len(ce) for ce in self.counterexamples) / max(1, len(self.counterexamples)),
            "table_stats": self.table.get_statistics(),
        }
        
        if self.refinement_times:
            stats["avg_refinement_time"] = sum(self.refinement_times) / len(self.refinement_times)
            
        if self.equivalence_times:
            stats["avg_equivalence_time"] = sum(self.equivalence_times) / len(self.equivalence_times)
            
        # Breakdown of time spent
        refinement_total = sum(self.refinement_times)
        equivalence_total = sum(self.equivalence_times)
        stats["time_breakdown"] = {
            "refinement": refinement_total,
            "equivalence": equivalence_total,
            "other": total_time - refinement_total - equivalence_total
        }
        
        return stats
    
    def print_summary(self):
        """Print learning summary."""
        stats = self.get_statistics()
        
        print("\n" + "="*50)
        print("L* Learning Summary")
        print("="*50)
        
        print(f"Iterations: {stats['iterations']}")
        print(f"Total time: {stats['total_time']:.2f}s")
        print(f"Final DFA states: {stats['final_states']}")
        
        print(f"\nCounterexamples: {stats['counterexamples']}")
        print(f"Average CE length: {stats['avg_ce_length']:.1f}")
        
        print(f"\nTime breakdown:")
        breakdown = stats['time_breakdown']
        print(f"  Refinement: {breakdown['refinement']:.2f}s "
              f"({breakdown['refinement']/stats['total_time']*100:.1f}%)")
        print(f"  Equivalence: {breakdown['equivalence']:.2f}s "
              f"({breakdown['equivalence']/stats['total_time']*100:.1f}%)")
        
        print(f"\nTable statistics:")
        table_stats = stats['table_stats']
        print(f"  States (|S|): {table_stats['states']}")
        print(f"  Experiments (|E|): {table_stats['experiments']}")
        print(f"  Total queries: {table_stats['total_queries']}")
        print(f"  Cache hit rate: {table_stats['cache_hit_rate']:.1%}")
        
        print("="*50)


def run_lstar(teacher, time_limit: Optional[float] = None,
              validation_set: Optional[List[Tuple[str, bool]]] = None) -> DFA:
    """
    Convenience function matching Weiss interface.
    
    Args:
        teacher: Oracle providing queries
        time_limit: Optional time bound
        validation_set: Optional validation examples for tracking accuracy
        
    Returns:
        Learned DFA (best found if validation set provided)
    """
    learner = LStarAlgorithm(teacher, time_limit, validation_set=validation_set)
    dfa = learner.run()
    learner.print_summary()
    
    # Attach learning history to result
    if hasattr(learner, 'hypotheses_history'):
        dfa.learning_history = learner.hypotheses_history
        
    return dfa