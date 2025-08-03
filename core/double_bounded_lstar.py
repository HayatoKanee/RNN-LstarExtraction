"""
Double-Bounded L* algorithm with separate limits for queries and counterexamples.

Key insight: Long counterexamples cause state explosion because their prefixes
combined with experiments create queries beyond our reliable range.
"""

from typing import Optional, Set, List
import time
from .lstar import LStarAlgorithm
from .bounded_lstar import BoundedObservationTable
from .dfa import DFA


class DoubleBoundedLStar(LStarAlgorithm):
    """
    L* algorithm with two bounds:
    1. max_query_length: Maximum length for membership queries
    2. max_counterexample_length: Maximum length for counterexamples
    
    This prevents state explosion from long counterexamples while
    still allowing reasonable query lengths.
    """
    
    def __init__(self, teacher, 
                 max_query_length: int = 20,
                 max_counterexample_length: int = 5,
                 time_limit: Optional[float] = None,
                 validation_set: Optional[List] = None):
        """
        Initialize double-bounded L*.
        
        Args:
            max_query_length: Maximum length for membership queries
            max_counterexample_length: Maximum length for counterexamples
        """
        self.max_query_length = max_query_length
        self.max_counterexample_length = max_counterexample_length
        
        # Wrap the teacher's equivalence oracle
        self.original_teacher = teacher
        self.wrapped_teacher = BoundedEquivalenceTeacher(
            teacher, 
            max_counterexample_length
        )
        
        super().__init__(self.wrapped_teacher, time_limit, validation_set)
        
    def run(self) -> DFA:
        """
        Run double-bounded L* algorithm.
        """
        print(f"\nStarting Double-Bounded L*")
        print(f"  Max query length: {self.max_query_length}")
        print(f"  Max counterexample length: {self.max_counterexample_length}")
        
        # Initialize bounded observation table
        self.table = BoundedObservationTable(
            self.alphabet, 
            self.teacher,
            max_query_length=self.max_query_length
        )
        
        if self.time_limit is not None:
            self.table.set_time_limit(self.time_limit, self.start_time)
        
        # Run standard L* algorithm
        result = super().run()
        
        if result:
            print(f"\nDouble-Bounded L* complete!")
            print(f"  Learned DFA with {len(result.states)} states")
            print(f"  Guarantee: DFA is correct for strings up to length {self.max_counterexample_length}")
            
        return result


class BoundedEquivalenceTeacher:
    """
    Wrapper for teacher that limits counterexample length.
    """
    
    def __init__(self, teacher, max_counterexample_length: int):
        self.teacher = teacher
        self.max_ce_length = max_counterexample_length
        
        # Delegate all attributes to original teacher
        self.alphabet = teacher.alphabet
        
    def __getattr__(self, name):
        """Delegate unknown attributes to the wrapped teacher."""
        return getattr(self.teacher, name)
        
    def membership_queries(self, words):
        """Pass through membership queries."""
        return self.teacher.membership_queries(words)
    
    def classify_word(self, word):
        """Pass through single membership query."""
        return self.teacher.classify_word(word)
    
    def equivalence_query(self, hypothesis_dfa: DFA, 
                         iteration: Optional[int] = None) -> Optional[str]:
        """
        Equivalence query with bounded search.
        
        Only searches for counterexamples up to max_ce_length.
        """
        print(f"\nBounded equivalence query (iteration {iteration})")
        print(f"  Max counterexample length: {self.max_ce_length}")
        print(f"  DFA states: {len(hypothesis_dfa.states)}")
        
        # If the oracle is BFS, temporarily limit its depth
        if hasattr(self.teacher, 'equivalence_oracle'):
            oracle = self.teacher.equivalence_oracle
            if hasattr(oracle, 'max_depth'):
                original_depth = oracle.max_depth
                oracle.max_depth = min(oracle.max_depth, self.max_ce_length)
                
                try:
                    ce = self.teacher.equivalence_query(hypothesis_dfa, iteration)
                finally:
                    oracle.max_depth = original_depth
                    
                if ce and len(ce) > self.max_ce_length:
                    print(f"  Warning: Found CE of length {len(ce)}, ignoring")
                    return None
                    
                return ce
        
        # For other oracles, do exhaustive search up to bound
        return self._bounded_search(hypothesis_dfa)
    
    def _bounded_search(self, hypothesis_dfa: DFA) -> Optional[str]:
        """
        Exhaustive search for counterexamples up to max length.
        """
        print("  Using exhaustive bounded search...")
        
        for length in range(self.max_ce_length + 1):
            if length == 0:
                strings = ['']
            else:
                # Generate all strings of this length
                strings = []
                for i in range(len(self.alphabet) ** length):
                    s = ''
                    n = i
                    for _ in range(length):
                        s = self.alphabet[n % len(self.alphabet)] + s
                        n //= len(self.alphabet)
                    strings.append(s)
            
            # Check each string
            for s in strings:
                hyp_label = hypothesis_dfa.accepts(list(s))
                rnn_label = self.teacher.classify_word(s)
                
                if hyp_label != rnn_label:
                    print(f"  Counterexample found: '{s}' (length {len(s)})")
                    return s
        
        print("  No counterexample found within bound")
        return None
    
    def get_statistics(self):
        """Pass through statistics request."""
        return self.teacher.get_statistics()


def test_double_bounded():
    """Quick test of double-bounded L*."""
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    from teacher.teacher import Teacher
    from models.rnn_classifier import LSTMClassifier
    from grammars.tomita import get_tomita_grammar
    import torch
    import numpy as np
    
    # Load Tomita 3 model
    model_path = "trained_models/tomita3_lstm_h50.pt"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    checkpoint = torch.load(model_path, map_location=device)
    config = checkpoint['model_config']
    
    model = LSTMClassifier(
        alphabet_size=config['alphabet_size'],
        embedding_dim=config['embedding_dim'],
        hidden_dim=config['hidden_dim'],
        num_layers=config['num_layers'],
        device=device
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    model.alphabet = checkpoint.get('alphabet', ['0', '1'])
    
    # Add classify method
    def classify(self, word):
        from teacher.rnn_oracle import RNNOracle
        if not hasattr(self, '_oracle'):
            self._oracle = RNNOracle(self, self.alphabet)
        return self._oracle.classify_word(word)
    
    import types
    model.classify = types.MethodType(classify, model)
    
    # Create teacher with BFS oracle
    teacher = Teacher(
        rnn_model=model,
        alphabet=model.alphabet,
        oracle_type='bfs',
        oracle_params={'max_depth': 20},
        use_adaptive=False
    )
    
    # Run double-bounded L*
    dbl_lstar = DoubleBoundedLStar(
        teacher=teacher,
        max_query_length=20,
        max_counterexample_length=5
    )
    
    dfa = dbl_lstar.run()
    
    if dfa:
        # Validate
        grammar_func, _ = get_tomita_grammar(3)
        correct = 0
        total = 0
        
        for length in range(6):  # Test up to CE bound
            for i in range(2 ** length if length < 6 else 100):
                s = bin(i)[2:].zfill(length) if length > 0 else ''
                if dfa.accepts(list(s)) == grammar_func(s):
                    correct += 1
                total += 1
        
        print(f"\nAccuracy on strings ≤5: {correct}/{total} = {correct/total:.1%}")


if __name__ == "__main__":
    test_double_bounded()