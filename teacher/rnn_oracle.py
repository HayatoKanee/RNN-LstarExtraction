"""
RNN Oracle implementation for L* learning.

Provides membership queries with aggressive caching since RNN inference
is the primary computational bottleneck in DFA extraction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Optional, Tuple, Union
from collections import OrderedDict
import time


class RNNOracle:
    """
    Oracle interface for L* learning from RNNs.
    
    Implements efficient membership queries through caching and batching,
    with support for hidden state extraction needed by whitebox methods.
    """
    
    def __init__(self, 
                 rnn_model: nn.Module,
                 alphabet: List[str],
                 threshold: float = 0.5,
                 device: Optional[str] = None,
                 cache_size: int = 100000,
                 batch_size: int = 32):
        """
        Initialize RNN oracle.
        
        Args:
            rnn_model: Trained RNN model (LSTM/GRU)
            alphabet: Input alphabet
            threshold: Classification threshold
            device: Computation device (cuda/cpu)
            cache_size: Maximum cached queries
            batch_size: Batch size for inference
        """
        self.rnn = rnn_model
        self.alphabet = alphabet
        self.threshold = threshold
        self.batch_size = batch_size
        
        # Symbol to index mapping
        self.char_to_idx = {char: idx for idx, char in enumerate(alphabet)}
        
        # Device handling
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.rnn.to(self.device)
        self.rnn.eval()  # Always in eval mode
        
        # Caching with LRU eviction
        self.cache = OrderedDict()
        self.cache_size = cache_size
        
        # Statistics
        self.query_count = 0
        self.cache_hits = 0
        self.forward_passes = 0
        
        # For time limits
        self.partitioner = None
        self.time_limit = None
        self.start_time = None
        
        # Extract RNN architecture info
        self._analyze_rnn_architecture()
        
    def set_time_limit(self, time_limit: float, start_time: float):
        """Set time limit for operations."""
        self.time_limit = time_limit
        self.start_time = start_time
        
    def membership_queries(self, words: List[str]) -> List[bool]:
        """
        Batch membership queries with caching.
        
        Args:
            words: List of strings to classify
            
        Returns:
            List of boolean classifications
            
        Critical for performance: processes uncached words in batches
        to maximize GPU utilization.
        """
        results = []
        uncached_words = []
        uncached_indices = []
        
        # Check cache first
        for i, word in enumerate(words):
            self.query_count += 1
            
            if word in self.cache:
                self.cache_hits += 1
                # Move to end for LRU
                self.cache.move_to_end(word)
                results.append(self.cache[word]['label'])
            else:
                uncached_words.append(word)
                uncached_indices.append(i)
                results.append(None)  # Placeholder
                
        # Process uncached words in batches
        if uncached_words:
            uncached_results = self._batch_classify(uncached_words)
            
            # Fill results and update cache
            for word, result, idx in zip(uncached_words, uncached_results, uncached_indices):
                results[idx] = result['label']
                self._add_to_cache(word, result)
                
        return results
    
    def classify_word(self, word: str) -> bool:
        """Single word classification (for compatibility)."""
        return self.membership_queries([word])[0]
        
    def get_probability(self, word: str) -> float:
        """
        Get RNN output probability for word.
        
        Used by priority search to compute uncertainty.
        """
        if word in self.cache:
            self.cache.move_to_end(word)
            return self.cache[word]['probability']
            
        # Compute if not cached
        result = self._classify_single(word)
        self._add_to_cache(word, result)
        return result['probability']
        
    def get_hidden_states(self, word: str) -> np.ndarray:
        """
        Get final hidden state after processing word.
        
        Used for state clustering.
        """
        if word in self.cache and 'hidden' in self.cache[word]:
            self.cache.move_to_end(word)
            return self.cache[word]['hidden']
            
        # Compute with hidden state extraction
        result = self._classify_single(word, extract_hidden=True)
        self._add_to_cache(word, result)
        return result['hidden']
        
    def get_first_RState(self) -> Tuple[np.ndarray, bool]:
        """
        Get initial RNN state and classification.
        
        Returns:
            Tuple of (state_vector, accepting)
        """
        # Get initial hidden state directly
        with torch.no_grad():
            # Get learnable initial states from the model
            if self.rnn_type == 'lstm':
                h0 = self.rnn.h0  # Shape: [num_layers, 1, hidden_size]
                c0 = self.rnn.c0  # Shape: [num_layers, 1, hidden_size]
                # Concatenate c and h like Weiss: [c1...cn, h1...hn]
                state_vec = torch.cat([c0.flatten(), h0.flatten()]).cpu().numpy()
            else:  # GRU
                h0 = self.rnn.h0  # Shape: [num_layers, 1, hidden_size]
                state_vec = h0.flatten().cpu().numpy()
            
            # Get initial classification by actually classifying empty string
            # This ensures consistency with how the RNN classifies ""
            accepting = self.classify_word("")
            
        return state_vec, accepting
        
    def get_next_RState(self, current_state: np.ndarray, symbol: str) -> Tuple[np.ndarray, bool]:
        """
        Get next RNN state after processing symbol.
        
        Args:
            current_state: Current RNN state vector
            symbol: Next symbol to process
            
        Returns:
            Tuple of (state_vector, accepting)
        """
        with torch.no_grad():
            # Convert state back to torch tensors
            if self.rnn_type == 'lstm':
                # Split concatenated state back into c and h (Weiss order: [c1...cn, h1...hn])
                state_size = len(current_state) // 2
                c_flat = current_state[:state_size]
                h_flat = current_state[state_size:]
                
                # Reshape to [num_layers, batch_size, hidden_size]
                c = torch.from_numpy(c_flat).float().to(self.device)
                c = c.view(self.num_layers, 1, self.hidden_size)
                h = torch.from_numpy(h_flat).float().to(self.device)
                h = h.view(self.num_layers, 1, self.hidden_size)
                hidden = (h, c)
            else:  # GRU
                h = torch.from_numpy(current_state).float().to(self.device)
                h = h.view(self.num_layers, 1, self.hidden_size)
                hidden = h
            
            # Process single symbol
            idx = self.char_to_idx[symbol]
            x = torch.tensor([[idx]], device=self.device)  # [1, 1]
            
            # Forward pass
            if hasattr(self.rnn, 'embedding'):
                embedded = self.rnn.embedding(x)
                output, new_hidden = self.rnn_layer(embedded, hidden)
            else:
                # One-hot encoding
                one_hot = torch.zeros(1, 1, len(self.alphabet), device=self.device)
                one_hot[0, 0, idx] = 1
                output, new_hidden = self.rnn_layer(one_hot, hidden)
            
            # Extract new hidden state vector
            if self.rnn_type == 'lstm':
                h_n, c_n = new_hidden
                # Concatenate in Weiss order: [c1...cn, h1...hn]
                state_vec = torch.cat([c_n.flatten(), h_n.flatten()]).cpu().numpy()
            else:  # GRU
                state_vec = new_hidden.flatten().cpu().numpy()
            
            # Get classification
            if hasattr(self.rnn, 'output'):
                logits = self.rnn.output(output[:, -1, :])  # Shape: [1, 2]
            elif hasattr(self.rnn, 'fc'):
                logits = self.rnn.fc(output[:, -1, :])
            else:
                # Fallback: assume the output already has logits
                logits = output[:, -1, :]
            
            # Apply softmax for 2-class classification
            probs = F.softmax(logits, dim=1)
            prob_true = probs[0, 1].item()
            
        return state_vec, prob_true > self.threshold
        
    def compute_positional_importance(self, word: str) -> np.ndarray:
        """
        Compute importance of each position via gradients.
        
        Returns:
            Array of importance scores for each position
        """
        if not word:
            return np.array([])
            
        # Enable gradients temporarily
        self.rnn.train()
        
        # Encode word
        indices = [self.char_to_idx[c] for c in word]
        x = torch.tensor(indices, device=self.device).unsqueeze(0)
        x.requires_grad = True
        
        # Forward pass
        output = self.rnn(x)
        if isinstance(output, tuple):
            output = output[0]
        output = output[:, -1, :]  # Final timestep
        
        # Compute gradient w.r.t. input
        output.sum().backward()
        
        # Extract importance (gradient magnitudes)
        importance = x.grad.abs().squeeze().cpu().numpy()
        
        self.rnn.eval()
        return importance
        
    def _batch_classify(self, words: List[str]) -> List[Dict]:
        """
        Classify batch of words efficiently.
        
        Handles variable length sequences via padding.
        """
        results = []
        
        # Process in mini-batches
        for i in range(0, len(words), self.batch_size):
            batch_words = words[i:i + self.batch_size]
            batch_results = self._process_batch(batch_words)
            results.extend(batch_results)
            
        return results
        
    def _process_batch(self, words: List[str]) -> List[Dict]:
        """Process single batch of words."""
        if not words:
            return []
            
        self.forward_passes += 1
        
        # Separate empty strings from non-empty
        empty_indices = []
        non_empty_words = []
        non_empty_indices = []
        
        for i, word in enumerate(words):
            if word:
                non_empty_words.append(word)
                non_empty_indices.append(i)
            else:
                empty_indices.append(i)
                
        # Initialize results
        batch_results = [None] * len(words)
        
        # Handle empty strings separately
        if empty_indices:
            empty_result = self._classify_single("", extract_hidden=False)
            for idx in empty_indices:
                batch_results[idx] = empty_result
                
        # Process non-empty strings
        if non_empty_words:
            # Sort by length for efficient packing (required by pack_padded_sequence)
            sorted_indices = sorted(range(len(non_empty_words)), key=lambda i: len(non_empty_words[i]), reverse=True)
            sorted_words = [non_empty_words[i] for i in sorted_indices]
            sorted_orig_indices = [non_empty_indices[i] for i in sorted_indices]
            
            # Encode and pad sequences
            max_len = max(len(w) for w in sorted_words)
            batch_size = len(sorted_words)
            
            # Create padded tensor
            padded = torch.zeros(batch_size, max_len, dtype=torch.long, device=self.device)
            lengths = []
            
            for i, word in enumerate(sorted_words):
                indices = [self.char_to_idx[c] for c in word]
                padded[i, :len(indices)] = torch.tensor(indices)
                lengths.append(len(indices))
            
            # Pack sequences to handle variable lengths properly
            from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
            
            # Embed the sequences
            embedded = self.rnn.embedding(padded)
            
            # Pack the embedded sequences
            packed = pack_padded_sequence(embedded, lengths, batch_first=True)
            
            # RNN forward pass with packed sequences
            with torch.no_grad():
                # Get initial state
                initial_state = self.rnn.get_initial_state(batch_size)
                
                # Run RNN on packed sequences
                if self.rnn.rnn_type == 'lstm':
                    _, (hidden, cell) = self.rnn.rnn(packed, initial_state)
                else:
                    _, hidden = self.rnn.rnn(packed, initial_state)
                
                # Get final hidden state from last layer
                final_hidden = hidden[-1]  # [batch_size, hidden_dim]
                
                # Classification
                logits = self.rnn.output(final_hidden)  # Shape: [batch_size, 2]
            
            # Apply softmax to get probabilities
            probs = F.softmax(logits, dim=1)
            
            # Extract probability of class 1 (True/Accept)
            prob_true = probs[:, 1]
            
            # Fill results (unsort to match original order)
            for sorted_i, orig_idx in enumerate(sorted_orig_indices):
                batch_results[orig_idx] = {
                    'label': prob_true[sorted_i].item() > self.threshold,
                    'probability': prob_true[sorted_i].item(),
                    'logits': logits[sorted_i].cpu().numpy()
                }
                
        return batch_results
        
    def _classify_single(self, word: str, extract_hidden: bool = False) -> Dict:
        """Classify single word with optional hidden state extraction."""
        self.forward_passes += 1
        
        # Create input tensor
        if not word:
            # Empty string needs special shape [1, 0]
            x = torch.zeros(1, 0, dtype=torch.long, device=self.device)
        else:
            indices = [self.char_to_idx[c] for c in word]
            x = torch.tensor([indices], dtype=torch.long, device=self.device)
            
        with torch.no_grad():
            # Check if the model returns hidden states
            output = self.rnn(x)
            
            # Extract hidden state if needed
            hidden = None
            if extract_hidden:
                # Use the model's get_final_state method to extract proper hidden states
                if hasattr(self.rnn, 'get_final_state'):
                    # If the model has a get_final_state method, use it
                    hidden = self.rnn.get_final_state(word)
                else:
                    # Otherwise, use forward with return_all_hidden if available
                    if hasattr(self.rnn, 'forward'):
                        try:
                            # Try to get hidden states using return_all_hidden
                            result = self.rnn.forward(x, return_all_hidden=True)
                            if isinstance(result, tuple) and len(result) == 3:
                                _, _, (h_n, c_n) = result
                                
                                # Extract hidden state based on RNN type
                                if self.rnn_type == 'lstm' and c_n is not None:
                                    # Concatenate c and h states: [c1...cn, h1...hn]
                                    c_flat = c_n.squeeze(1).flatten().cpu().numpy()
                                    h_flat = h_n.squeeze(1).flatten().cpu().numpy()
                                    hidden = np.concatenate([c_flat, h_flat])
                                else:
                                    # GRU or other RNN types
                                    hidden = h_n.squeeze(1).flatten().cpu().numpy()
                        except:
                            # If that fails, use get_final_state method from below
                            hidden = self.get_final_state(word)
                
            # The output should be shape [1, 2]
            logits = output  # Shape: [1, 2]
            
            # Apply softmax to get probabilities
            probs = F.softmax(logits, dim=1)
            
            # Extract probability of class 1 (True/Accept)
            prob_true = probs[0, 1].item()
            
        return {
            'label': prob_true > self.threshold,
            'probability': prob_true,
            'logits': logits[0].cpu().numpy(),
            'hidden': hidden
        }
        
    def _add_to_cache(self, word: str, result: Dict):
        """Add result to cache with LRU eviction."""
        # Evict oldest if at capacity
        if len(self.cache) >= self.cache_size:
            self.cache.popitem(last=False)
            
        self.cache[word] = result
        
    def get_statistics(self) -> Dict[str, Union[int, float]]:
        """Return oracle statistics."""
        return {
            'total_queries': self.query_count,
            'cache_hits': self.cache_hits,
            'cache_hit_rate': self.cache_hits / max(1, self.query_count),
            'forward_passes': self.forward_passes,
            'cache_size': len(self.cache),
            'queries_per_forward': self.query_count / max(1, self.forward_passes)
        }
        
    def clear_cache(self):
        """Clear the membership query cache."""
        self.cache.clear()
        
    def _analyze_rnn_architecture(self):
        """Analyze RNN model to extract architecture information."""
        self.rnn_type = None
        self.rnn_layer = None
        self.hidden_size = None
        self.num_layers = None
        
        # Find the RNN layer
        for name, module in self.rnn.named_modules():
            if isinstance(module, nn.LSTM):
                self.rnn_type = 'lstm'
                self.rnn_layer = module
                self.hidden_size = module.hidden_size
                self.num_layers = module.num_layers
                break
            elif isinstance(module, nn.GRU):
                self.rnn_type = 'gru'
                self.rnn_layer = module  
                self.hidden_size = module.hidden_size
                self.num_layers = module.num_layers
                break
        
        if self.rnn_layer is None:
            raise ValueError("No LSTM or GRU layer found in the model")
    
    def __str__(self) -> str:
        """String representation."""
        stats = self.get_statistics()
        return (f"RNNOracle(alphabet_size={len(self.alphabet)}, "
                f"cache_hit_rate={stats['cache_hit_rate']:.1%}, "
                f"queries_per_forward={stats['queries_per_forward']:.1f})")


    def get_final_state(self, word: str) -> np.ndarray:
        """
        Get the final RNN hidden state after processing a word.
        
        Args:
            word: Input string
            
        Returns:
            Final hidden state as numpy array (concatenated h and c for LSTM)
        """
        # Process with RNN model
        with torch.no_grad():
            if not word:  # Empty string
                # Get initial hidden state from the model's learned parameters
                if self.rnn_type == 'lstm':
                    h0 = self.rnn.h0  # Shape: [num_layers, 1, hidden_size]
                    c0 = self.rnn.c0  # Shape: [num_layers, 1, hidden_size]
                    # Concatenate c and h like Weiss: [c1...cn, h1...hn]
                    final_state = torch.cat([c0.flatten(), h0.flatten()]).cpu().numpy()
                else:  # GRU
                    h0 = self.rnn.h0  # Shape: [num_layers, 1, hidden_size]
                    final_state = h0.flatten().cpu().numpy()
            else:
                # Encode word
                indices = [self.char_to_idx[c] for c in word]
                x = torch.tensor([indices], dtype=torch.long, device=self.device)
                
                # Get initial state from model
                if self.rnn_type == 'lstm':
                    h0 = self.rnn.h0.expand(-1, 1, -1).contiguous()
                    c0 = self.rnn.c0.expand(-1, 1, -1).contiguous()
                    initial_state = (h0, c0)
                else:
                    h0 = self.rnn.h0.expand(-1, 1, -1).contiguous()
                    initial_state = h0
                
                # Get embedded representation
                if hasattr(self.rnn, 'embedding'):
                    embedded = self.rnn.embedding(x)
                else:
                    # One-hot encoding
                    one_hot = torch.zeros(1, len(word), len(self.alphabet), device=self.device)
                    for i, idx in enumerate(indices):
                        one_hot[0, i, idx] = 1
                    embedded = one_hot
                
                # Run through RNN layer with initial state
                if self.rnn_type == 'lstm':
                    _, (h_n, c_n) = self.rnn_layer(embedded, initial_state)
                    # Concatenate c and h states like Weiss: [c1...cn, h1...hn]
                    final_state = torch.cat([c_n.flatten(), h_n.flatten()]).cpu().numpy()
                else:  # GRU
                    _, h_n = self.rnn_layer(embedded, initial_state)
                    final_state = h_n.flatten().cpu().numpy()
        
        return final_state


class RNNWrapper:
    """
    Compatibility wrapper for different RNN types.
    
    Provides uniform interface regardless of underlying architecture.
    """
    
    def __init__(self, model: nn.Module, input_dim: int, hidden_dim: int):
        """
        Wrap RNN model for uniform access.
        
        Args:
            model: PyTorch RNN model
            input_dim: Input dimension (alphabet size)
            hidden_dim: Hidden state dimension
        """
        self.model = model
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
    def get_initial_state(self, batch_size: int = 1) -> torch.Tensor:
        """Get initial hidden state."""
        device = next(self.model.parameters()).device
        
        if isinstance(self.model, nn.LSTM):
            h0 = torch.zeros(1, batch_size, self.hidden_dim, device=device)
            c0 = torch.zeros(1, batch_size, self.hidden_dim, device=device)
            return (h0, c0)
        else:
            return torch.zeros(1, batch_size, self.hidden_dim, device=device)