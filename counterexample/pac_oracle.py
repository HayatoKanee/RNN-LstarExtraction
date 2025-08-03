"""
PAC (Probably Approximately Correct) Equivalence Oracle

Implements statistical sampling-based equivalence checking:
- Draws samples according to a distribution
- Provides probabilistic guarantees on finding counterexamples
- Black-box approach (no RNN internals needed)
"""

import random
import time
import numpy as np
from typing import Optional, List, Tuple, Callable, Dict, Any
from collections import defaultdict

from .base_oracle import EquivalenceOracle
from core.dfa import DFA


class PACEquivalenceOracle(EquivalenceOracle):
    """
    PAC equivalence oracle using statistical sampling.
    
    Guarantees: With probability at least (1 - δ), if the hypothesis
    has error rate > ε, we will find a counterexample.
    """
    
    def __init__(self, rnn_oracle, alphabet: List[str],
                 epsilon: float = 0.1,
                 delta: float = 0.1,
                 max_length: int = 30,
                 distribution: str = "geometric",
                 **kwargs):
        """
        Initialize PAC oracle.
        
        Args:
            rnn_oracle: RNN wrapper with membership queries
            alphabet: List of alphabet symbols
            epsilon: Error tolerance (default 0.1)
            delta: Confidence parameter (default 0.1) 
            max_length: Maximum string length to test
            distribution: Sampling distribution ("uniform", "geometric", "empirical")
        """
        super().__init__(rnn_oracle, alphabet, **kwargs)
        self.epsilon = epsilon
        self.delta = delta
        self.max_length = max_length
        self.distribution = distribution
        
        # Track round number for proper PAC bounds
        self.round = 0
        
        # For empirical distribution
        self.observed_lengths = []
        
        # PAC-specific statistics
        self.total_samples = 0
        
    def find_counterexample(self, hypothesis_dfa: DFA, iteration: int,
                          time_limit: Optional[float] = None) -> Optional[str]:
        """
        Find counterexample using PAC sampling.
        
        Args:
            hypothesis_dfa: Current hypothesis DFA
            iteration: L* iteration number
            time_limit: Optional time limit
            
        Returns:
            Counterexample string or None
        """
        
        # Increment round counter
        self.round += 1
        
        # Calculate sample size using proper PAC bounds
        # m = (1/ε) * (ln(1/δ) + round * ln(2))
        sample_size = int(np.ceil((1.0 / self.epsilon) * (np.log(1.0 / self.delta) + self.round * np.log(2))))
        
        print(f"\nPAC Equivalence Query (iteration {iteration})")
        print(f"  Round: {self.round}")
        print(f"  Sample size: {sample_size} (ε={self.epsilon}, δ={self.delta})")
        print(f"  Distribution: {self.distribution}")
        
        start_time = time.time()
        samples_checked = 0
        
        # Update base class statistics
        self.total_queries += 1
        
        # Draw samples according to distribution
        for _ in range(sample_size):
            if time_limit and (time.time() - start_time) > time_limit:
                print(f"  Time limit reached after {samples_checked} samples")
                break
                
            # Sample a string
            word = self._sample_string()
            samples_checked += 1
            
            # Check if it's a counterexample
            dfa_accepts = hypothesis_dfa.accepts(word)
            rnn_accepts = self.rnn_oracle.classify_word(word)
            
            if dfa_accepts != rnn_accepts:
                # Update statistics
                self.counterexamples_found += 1
                self.total_time += time.time() - start_time
                self.total_samples += samples_checked
                
                print(f"  Counterexample found: '{word}' (length {len(word)})")
                print(f"  Samples checked: {samples_checked}")
                
                # Update empirical distribution
                if self.distribution == "empirical":
                    self.observed_lengths.append(len(word))
                    
                return word
        
        # No counterexample found
        self.total_samples += samples_checked
        self.total_time += time.time() - start_time
        print(f"  No counterexample found in {samples_checked} samples")
        return None
        
    def _sample_string(self) -> str:
        """Sample a string according to the chosen distribution."""
        # Sample length
        if self.distribution == "uniform":
            # Uniform over lengths 0 to max_length
            length = random.randint(0, self.max_length)
            
        elif self.distribution == "geometric":
            # Geometric distribution favoring shorter strings
            # P(length = k) = (1-p)^k * p
            p = 0.2  # Higher p means shorter strings on average
            length = min(np.random.geometric(p) - 1, self.max_length)  # -1 since geometric starts at 1
            
        elif self.distribution == "empirical":
            # Use observed counterexample lengths
            if self.observed_lengths:
                # Sample from observed distribution
                length = random.choice(self.observed_lengths)
                # Add some noise
                length = max(0, min(length + random.randint(-2, 2), self.max_length))
            else:
                # Fall back to geometric for first iteration
                length = min(np.random.geometric(0.1), self.max_length)
        else:
            raise ValueError(f"Unknown distribution: {self.distribution}")
            
        # Generate random string of that length
        return ''.join(random.choice(self.alphabet) for _ in range(length))
        
    def get_statistics(self) -> Dict[str, Any]:
        """Return oracle statistics."""
        # Get base statistics
        stats = super().get_statistics()
        
        # Add PAC-specific statistics
        current_sample_size = int(np.ceil((1.0 / self.epsilon) * (np.log(1.0 / self.delta) + self.round * np.log(2))))
        stats.update({
            'total_samples': self.total_samples,
            'current_round': self.round,
            'current_sample_size': current_sample_size,
            'epsilon': self.epsilon,
            'delta': self.delta,
            'distribution': self.distribution,
            'avg_samples_per_counterexample': (
                self.total_samples / max(1, self.counterexamples_found)
            )
        })
        
        return stats


class WeightedPACOracle(PACEquivalenceOracle):
    """
    Enhanced PAC oracle that learns a weighted distribution
    from previously found counterexamples.
    """
    
    def __init__(self, rnn_oracle, alphabet: List[str], **kwargs):
        super().__init__(rnn_oracle, alphabet, **kwargs)
        
        # Track patterns in counterexamples
        self.ce_patterns = defaultdict(int)
        self.length_distribution = defaultdict(int)
        
    def find_counterexample(self, hypothesis_dfa, iteration: int,
                          time_limit: Optional[float] = None) -> Optional[str]:
        """Find counterexample with adaptive sampling."""
        ce = super().find_counterexample(hypothesis_dfa, iteration, time_limit)
        
        if ce:
            # Update pattern statistics
            self._update_patterns(ce)
            
        return ce
        
    def _update_patterns(self, counterexample: str):
        """Update pattern statistics from counterexample."""
        # Track length
        self.length_distribution[len(counterexample)] += 1
        
        # Track character patterns
        for i in range(len(counterexample)):
            # Single character
            self.ce_patterns[counterexample[i]] += 1
            
            # Bigrams
            if i < len(counterexample) - 1:
                bigram = counterexample[i:i+2]
                self.ce_patterns[bigram] += 1
                
    def _sample_string(self) -> str:
        """Sample using learned patterns."""
        if not self.length_distribution:
            # No patterns yet, use parent method
            return super()._sample_string()
            
        # Sample length from observed distribution
        lengths = list(self.length_distribution.keys())
        weights = list(self.length_distribution.values())
        length = random.choices(lengths, weights=weights)[0]
        
        # Add noise
        length = max(0, min(length + random.randint(-2, 2), self.max_length))
        
        # Generate string with pattern bias
        result = []
        for _ in range(length):
            if result and random.random() < 0.7:  # 70% chance to use patterns
                # Try to continue with a bigram
                last_char = result[-1]
                bigram_candidates = [p for p in self.ce_patterns 
                                   if len(p) == 2 and p[0] == last_char]
                if bigram_candidates:
                    bigram = random.choice(bigram_candidates)
                    result.append(bigram[1])
                    continue
                    
            # Otherwise random character
            result.append(random.choice(self.alphabet))
            
        return ''.join(result)


def create_pac_oracle(rnn_oracle, alphabet: List[str], 
                     enhanced: bool = False,
                     **kwargs) -> PACEquivalenceOracle:
    """
    Factory function to create PAC oracle.
    
    Args:
        rnn_oracle: RNN oracle for membership queries
        alphabet: Input alphabet
        enhanced: Whether to use weighted/adaptive version
        **kwargs: Additional arguments for oracle
        
    Returns:
        PAC equivalence oracle instance
    """
    if enhanced:
        return WeightedPACOracle(rnn_oracle, alphabet, **kwargs)
    else:
        return PACEquivalenceOracle(rnn_oracle, alphabet, **kwargs)