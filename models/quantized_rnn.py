"""
Quantized RNN Classifier for bounded DFA extraction.

This model uses quantized hidden states to encourage finite-state behavior,
making it easier to extract clean DFAs using the whitebox approach.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional

from .rnn_classifier import RNNClassifier


class QuantizedLSTMClassifier(RNNClassifier):
    """
    LSTM with quantized hidden states for better DFA extraction.
    
    Uses vector quantization to discretize hidden states into a finite set,
    encouraging the network to learn finite-state behavior.
    """
    
    def __init__(
        self,
        alphabet_size: int = 2,
        embedding_dim: int = 3,
        hidden_dim: int = 5,
        num_layers: int = 1,
        num_discrete_states: int = 20,  # Number of discrete hidden states
        quantization_levels: int = 4,    # Quantization granularity per dimension
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        **kwargs
    ):
        # Initialize parent with LSTM type
        super().__init__(
            alphabet_size=alphabet_size,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            rnn_type='lstm',
            device=device
        )
        
        self.num_discrete_states = num_discrete_states
        self.quantization_levels = quantization_levels
        
        # For simple quantization, we'll quantize each dimension independently
        # This creates a grid of possible hidden states
        self.quantization_scale = quantization_levels / 2.0  # Map [-1, 1] to levels
        
    def quantize_hidden_state(self, hidden: torch.Tensor) -> torch.Tensor:
        """
        Quantize hidden states to discrete values.
        
        Simple approach: quantize each dimension independently
        to one of `quantization_levels` values in range [-1, 1].
        """
        # Clamp to [-1, 1] range
        hidden_clamped = torch.clamp(hidden, -1.0, 1.0)
        
        # Scale to [0, quantization_levels]
        scaled = (hidden_clamped + 1.0) * self.quantization_scale
        
        # Round to nearest level
        quantized = torch.round(scaled)
        
        # Scale back to [-1, 1]
        result = (quantized / self.quantization_scale) - 1.0
        
        # Use straight-through estimator for gradients
        return hidden + (result - hidden).detach()
    
    def forward(self, x: torch.Tensor, return_all_hidden: bool = False):
        """
        Forward pass with quantized hidden states.
        """
        # Handle different input types
        if x.dim() == 1:
            x = x.unsqueeze(0)  # Add batch dimension
        
        batch_size, seq_len = x.shape
        
        # Special handling for empty sequences
        if seq_len == 0:
            # For empty string, use initial state directly
            initial_state = self.get_initial_state(batch_size)
            if self.rnn_type == 'lstm':
                h0, c0 = initial_state
                # Quantize initial hidden state
                h0_quantized = self.quantize_hidden_state(h0)
                final_hidden = h0_quantized[-1]  # Last layer
            else:
                final_hidden = self.quantize_hidden_state(initial_state[-1])
            
            logits = self.output(final_hidden)
            return logits
        
        # Embed input
        embedded = self.embedding(x)
        
        # Get initial state
        initial_state = self.get_initial_state(batch_size)
        
        # Process sequence step by step with quantization
        all_hidden_states = []
        
        if self.rnn_type == 'lstm':
            h, c = initial_state
            # Quantize initial hidden state
            h = self.quantize_hidden_state(h)
            
            for t in range(seq_len):
                # Get input for this timestep
                x_t = embedded[:, t:t+1, :]
                
                # Run one step of LSTM
                _, (h, c) = self.rnn(x_t, (h, c))
                
                # Quantize hidden state after each step
                h = self.quantize_hidden_state(h)
                
                if return_all_hidden:
                    all_hidden_states.append(h)
            
            final_hidden = h[-1]  # Last layer
            final_states = (h, c)
        
        # Classification
        logits = self.output(final_hidden)
        
        if return_all_hidden:
            # Stack hidden states
            all_hidden = torch.stack(all_hidden_states, dim=1)  # [seq_len, batch, layers, hidden]
            all_hidden = all_hidden.permute(2, 1, 0, 3)  # [layers, batch, seq_len, hidden]
            return logits, all_hidden[:, :, :, :], final_states
        
        return logits
    
    def get_state_vector(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get quantized state vector for L* extraction.
        """
        with torch.no_grad():
            if x.numel() == 0:  # Empty sequence
                initial_state = self.get_initial_state(1)
                if self.rnn_type == 'lstm':
                    h, c = initial_state
                    # Quantize before concatenating
                    h_quantized = self.quantize_hidden_state(h)
                    # Concatenate cell states then hidden states
                    state_vec = torch.cat([c.squeeze(1).flatten(), h_quantized.squeeze(1).flatten()])
                else:
                    state_vec = self.quantize_hidden_state(initial_state).squeeze(1).flatten()
            else:
                _, _, (hidden, cell) = self.forward(x, return_all_hidden=True)
                
                if self.rnn_type == 'lstm':
                    # Hidden is already quantized from forward pass
                    # Concatenate cell states first, then hidden states (Weiss format)
                    cell_flat = cell.squeeze(1).flatten()
                    hidden_flat = hidden.squeeze(1).flatten()
                    state_vec = torch.cat([cell_flat, hidden_flat])
                else:
                    state_vec = hidden.squeeze(1).flatten()
        
        return state_vec