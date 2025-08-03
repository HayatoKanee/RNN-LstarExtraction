"""
DFA Extractor implementation.

This is the primary interface for extracting DFAs from RNNs using the
L* algorithm with whitebox access.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Optional, Dict, Any, Union, Tuple
import time

from core.lstar import LStarAlgorithm
from core.bounded_lstar import BoundedLStar
from core.dfa import DFA
from teacher.teacher import Teacher


class DFAExtractor:
    """
    DFA extraction from RNNs using L* with various equivalence oracles.
    
    Supports multiple oracle types:
    - whitebox: Weiss et al. (2018) with RNN internals
    - pac: Statistical sampling with guarantees
    - w_method: Chow's W-method systematic testing
    - bfs: Breadth-first search for shortest counterexample
    
    Usage:
        extractor = DFAExtractor(rnn_model, alphabet=['0', '1'])
        dfa = extractor.extract(oracle_type='whitebox', time_limit=60)
    """
    
    def __init__(self,
                 rnn_model: nn.Module,
                 alphabet: List[str],
                 threshold: float = 0.5,
                 device: Optional[str] = None,
                 # Cache parameters
                 cache_size: int = 100000,
                 batch_size: int = 32,
                 validation_set: Optional[List[Tuple[str, bool]]] = None):
        """
        Initialize DFA extractor.
        
        Args:
            rnn_model: Trained RNN to extract from
            alphabet: Input alphabet
            threshold: Classification threshold
            device: Computation device (cuda/cpu)
            cache_size: Maximum membership query cache size
            batch_size: Batch size for RNN inference
        """
        self.rnn_model = rnn_model
        self.alphabet = alphabet
        self.threshold = threshold
        self.device = device
        self.teacher = None  # Will be created in extract()
        self.validation_set = validation_set or []
        
        # Statistics
        self.extraction_stats = {}
        
    def extract(self, 
               oracle_type: str = 'whitebox',
               oracle_params: Optional[Dict[str, Any]] = None,
               time_limit: Optional[float] = None,
               verbose: bool = True,
               use_starting_examples: bool = True,
               use_bounded_lstar: bool = False,
               max_query_length: int = 20) -> DFA:
        """
        Extract DFA from RNN using L* with chosen equivalence oracle.
        
        Args:
            oracle_type: Type of equivalence oracle ('whitebox', 'pac', 'w_method', 'bfs')
            oracle_params: Oracle-specific parameters
            time_limit: Time budget in seconds (None for exact extraction)
            verbose: Print progress information
            use_bounded_lstar: Use bounded L* with negotiation protocol
            max_query_length: Maximum query length for bounded L*
            
        Returns:
            Extracted DFA (exact or best approximation)
        """
        start_time = time.time()
        
        if verbose:
            print("="*60)
            print("L* DFA Extraction")
            print("="*60)
            print(f"RNN Model: {self.rnn_model.__class__.__name__}")
            print(f"Alphabet: {self.alphabet}")
            print(f"Time Limit: {time_limit}s" if time_limit else "Time Limit: None (exact)")
            print(f"Oracle Type: {oracle_type}")
            if oracle_params:
                print(f"Oracle Parameters: {oracle_params}")
            if use_bounded_lstar:
                print(f"Using Bounded L* (max query length: {max_query_length})")
            print()
            
        # Create teacher with appropriate settings
        self.teacher = Teacher(
            rnn_model=self.rnn_model,
            alphabet=self.alphabet,
            initial_split_depth=10,
            device=self.device,
            oracle_type=oracle_type,
            oracle_params=oracle_params or {},
            use_starting_examples=use_starting_examples
        )
        
        # Set time limit if specified
        if time_limit:
            self.teacher.set_time_limit(time_limit, start_time)
                
        # Phase 2: L* Learning
        if verbose:
            print("\nPhase 2: L* Learning")
            print("-" * 30)
            
        # Create L* learner (bounded or standard)
        if use_bounded_lstar:
            learner = BoundedLStar(
                teacher=self.teacher,
                max_query_length=max_query_length,
                time_limit=time_limit,
                validation_set=self.validation_set
            )
        else:
            learner = LStarAlgorithm(
                teacher=self.teacher,
                time_limit=time_limit,
                validation_set=self.validation_set
            )
        
        # Run extraction
        dfa = learner.run()
        
        # Attach learning history to DFA
        if hasattr(learner, 'hypotheses_history'):
            dfa.learning_history = learner.hypotheses_history
        
        # Store learner reference for accessing learning history
        self.learner = learner
        
        # Collect statistics
        self.extraction_stats = {
            'total_time': time.time() - start_time,
            'lstar_stats': learner.get_statistics(),
            'teacher_stats': self.teacher.get_statistics(),
            'final_dfa_states': len(dfa.states)
        }
        
        if verbose:
            self._print_extraction_summary()
            
        return dfa
    
    def get_learning_history(self):
        """
        Get the learning history from the most recent extraction.
        
        Returns:
            List of hypothesis dictionaries, each containing:
            - 'iteration': iteration number
            - 'time': time elapsed
            - 'states': number of states
            - 'dfa_object': the hypothesis DFA
            - 'counterexample': counterexample info if rejected
        """
        if hasattr(self, 'learner') and hasattr(self.learner, 'hypotheses_history'):
            return self.learner.hypotheses_history
        return []
        
    def _print_extraction_summary(self):
        """Print detailed extraction summary."""
        stats = self.extraction_stats
        
        print("\n" + "="*60)
        print("L* Extraction Summary")
        print("="*60)
        
        # Overall performance
        print(f"Total extraction time: {stats['total_time']:.2f}s")
        print(f"Final DFA states: {stats['final_dfa_states']}")
        
        # L* statistics
        lstar = stats['lstar_stats']
        print(f"\nL* Algorithm:")
        print(f"  Iterations: {lstar['iterations']}")
        print(f"  Counterexamples: {lstar['counterexamples']}")
        print(f"  Avg CE length: {lstar['avg_ce_length']:.1f}")
        
        # Teacher statistics  
        teacher = stats['teacher_stats']
        oracle = teacher.get('oracle_stats', {})
        print(f"\nTeacher:")
        print(f"  Partitions: {teacher.get('partitions', 'N/A')}")
        if 'avg_ce_time' in teacher:
            print(f"  Avg CE generation time: {teacher['avg_ce_time']:.2f}s")
        
        if oracle:
            print(f"  Total queries: {oracle.get('total_queries', 0):,}")
            print(f"  Cache hit rate: {oracle.get('cache_hit_rate', 0):.1%}")
        
        # Time breakdown
        breakdown = lstar['time_breakdown']
        print(f"\nTime Breakdown:")
        print(f"  Table refinement: {breakdown['refinement']:.2f}s "
              f"({breakdown['refinement']/stats['total_time']*100:.1f}%)")
        print(f"  Equivalence checking: {breakdown['equivalence']:.2f}s "
              f"({breakdown['equivalence']/stats['total_time']*100:.1f}%)")
              
        print("="*60)
        
    def get_statistics(self) -> Dict[str, Any]:
        """Get extraction statistics."""
        return self.extraction_stats
        
    def __str__(self) -> str:
        """String representation."""
        return f"DFAExtractor(alphabet={self.alphabet})"