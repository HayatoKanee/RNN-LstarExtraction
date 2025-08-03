"""
Abstract base class for equivalence oracles in L* learning.

This provides a common interface for different equivalence checking strategies:
- Whitebox (access to RNN internals)
- PAC (statistical sampling)
- Adaptive (hybrid approaches)
- Future implementations
"""

from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any
from core.dfa import DFA


class EquivalenceOracle(ABC):
    """
    Abstract base class for equivalence oracles.
    
    All equivalence oracles must implement the find_counterexample method
    and can optionally provide statistics about their performance.
    """
    
    def __init__(self, rnn_oracle, alphabet: List[str], **kwargs):
        """
        Initialize the equivalence oracle.
        
        Args:
            rnn_oracle: Oracle for RNN membership queries
            alphabet: Input alphabet
            **kwargs: Additional oracle-specific parameters
        """
        self.rnn_oracle = rnn_oracle
        self.alphabet = alphabet
        
        # Statistics tracking
        self.total_queries = 0
        self.counterexamples_found = 0
        self.total_time = 0.0
        
        # Blacklist for negotiation protocol
        self.blacklist = set()
        
    @abstractmethod
    def find_counterexample(self, 
                          hypothesis_dfa: DFA,
                          iteration: int,
                          time_limit: Optional[float] = None) -> Optional[str]:
        """
        Find a counterexample where hypothesis_dfa and RNN disagree.
        
        Args:
            hypothesis_dfa: Current hypothesis DFA from L*
            iteration: Current L* iteration number
            time_limit: Optional time limit in seconds
            
        Returns:
            Counterexample string or None if equivalent
        """
        pass
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get performance statistics for the oracle.
        
        Returns:
            Dictionary of statistics
        """
        return {
            'type': self.__class__.__name__,
            'total_queries': self.total_queries,
            'counterexamples_found': self.counterexamples_found,
            'total_time': self.total_time,
            'avg_time_per_query': self.total_time / max(1, self.total_queries)
        }
    
    def reset_statistics(self):
        """Reset statistics counters."""
        self.total_queries = 0
        self.counterexamples_found = 0
        self.total_time = 0.0
    
    def set_blacklist(self, blacklist: set):
        """
        Set the blacklist of counterexamples to avoid.
        
        Args:
            blacklist: Set of strings that should not be returned as counterexamples
        """
        self.blacklist = blacklist
        
    def __str__(self) -> str:
        """String representation."""
        return f"{self.__class__.__name__}(alphabet_size={len(self.alphabet)})"