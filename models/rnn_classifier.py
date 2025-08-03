"""
RNN Classifier implementation following Weiss et al. (2018).

Notable architectural choices:
- Learnable initial states (crucial for DFA extraction)
- Small embedding dimension (3) for interpretability
- State vector concatenation format: [c1...cn, h1...hn] for LSTM
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional


class RNNClassifier(nn.Module):
    """
    RNN Classifier following Weiss et al.'s exact architecture.
    """
    
    def __init__(
        self,
        alphabet_size: int = 2,
        embedding_dim: int = 3,  # Weiss uses 3
        hidden_dim: int = 5,     # Weiss default is 5
        num_layers: int = 2,     # Weiss default is 2
        rnn_type: str = 'lstm',
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        super().__init__()
        
        self.alphabet_size = alphabet_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.rnn_type = rnn_type
        self.device = device
        
        # Embedding layer - small dimension like Weiss
        self.embedding = nn.Embedding(alphabet_size, embedding_dim)
        
        # CRITICAL: Learnable initial states (this is what Weiss does!)
        if rnn_type == 'lstm':
            # For LSTM: both hidden and cell states
            self.h0 = nn.Parameter(torch.randn(num_layers, 1, hidden_dim))
            self.c0 = nn.Parameter(torch.randn(num_layers, 1, hidden_dim))
            # Clip initial states like Weiss
            with torch.no_grad():
                self.h0.clamp_(-1, 1)
                self.c0.clamp_(-1, 1)
        else:
            # For GRU: only hidden state
            self.h0 = nn.Parameter(torch.randn(num_layers, 1, hidden_dim))
            with torch.no_grad():
                self.h0.clamp_(-1, 1)
        
        # RNN layer
        if rnn_type == 'lstm':
            self.rnn = nn.LSTM(
                embedding_dim, hidden_dim, num_layers,
                batch_first=True
            )
        else:
            self.rnn = nn.GRU(
                embedding_dim, hidden_dim, num_layers,
                batch_first=True
            )
        
        # Output layer - exactly 2 classes
        self.output = nn.Linear(hidden_dim, 2)
        
        self.to(device)
    
    def get_initial_state(self, batch_size: int):
        """Get initial hidden state for batch."""
        if self.rnn_type == 'lstm':
            # Expand learnable initial states to batch size
            h0 = self.h0.expand(-1, batch_size, -1).contiguous()
            c0 = self.c0.expand(-1, batch_size, -1).contiguous()
            return (h0, c0)
        else:
            h0 = self.h0.expand(-1, batch_size, -1).contiguous()
            return h0
    
    def forward(self, x: torch.Tensor, return_all_hidden: bool = False):
        """
        Forward pass matching Weiss's computation.
        
        Args:
            x: Input tensor [batch_size, seq_len] or single string
            return_all_hidden: Return all hidden states for L* extraction
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
                final_hidden = initial_state[0][-1]  # Last layer hidden
            else:
                final_hidden = initial_state[-1]
            logits = self.output(final_hidden)
            return logits
        
        # Embed input
        embedded = self.embedding(x)
        
        # Get initial state
        initial_state = self.get_initial_state(batch_size)
        
        # Run RNN
        output, (hidden, cell) = self.rnn(embedded, initial_state) if self.rnn_type == 'lstm' else \
                                 (self.rnn(embedded, initial_state)[0], (self.rnn(embedded, initial_state)[1], None))
        
        # Get final hidden state from last layer
        if self.rnn_type == 'lstm':
            final_hidden = hidden[-1]  # [batch_size, hidden_dim]
        else:
            final_hidden = hidden[-1]
        
        # Classification
        logits = self.output(final_hidden)
        
        if return_all_hidden:
            # Return all states for L* extraction
            return logits, output, (hidden, cell)
        
        return logits
    
    def get_state_vector(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get state vector for L* extraction, matching Weiss's format.
        Weiss concatenates: [c1, c2, ..., cn, h1, h2, ..., hn]
        """
        with torch.no_grad():
            if x.numel() == 0:  # Empty sequence
                initial_state = self.get_initial_state(1)
                if self.rnn_type == 'lstm':
                    h, c = initial_state
                    # Concatenate cell states then hidden states
                    state_vec = torch.cat([c.squeeze(1).flatten(), h.squeeze(1).flatten()])
                else:
                    state_vec = initial_state.squeeze(1).flatten()
            else:
                _, _, (hidden, cell) = self.forward(x, return_all_hidden=True)
                
                if self.rnn_type == 'lstm':
                    # Weiss concatenates cell states first, then hidden states
                    # Shape: [num_layers, batch_size, hidden_dim]
                    cell_flat = cell.squeeze(1).flatten()  # Flatten all cell states
                    hidden_flat = hidden.squeeze(1).flatten()  # Flatten all hidden states
                    state_vec = torch.cat([cell_flat, hidden_flat])
                else:
                    state_vec = hidden.squeeze(1).flatten()
        
        return state_vec
    
    def classify_string(self, string: str) -> bool:
        """Classify a single string."""
        # Get the actual device of the model parameters
        device = next(self.parameters()).device
        
        if len(string) == 0:
            x = torch.zeros(1, 0, dtype=torch.long, device=device)
        else:
            indices = [int(c) for c in string]
            x = torch.tensor([indices], dtype=torch.long, device=device)
        
        with torch.no_grad():
            logits = self.forward(x)
            probs = F.softmax(logits, dim=1)
            return probs[0, 1].item() > 0.5  # True class probability
    
    def probability_of_string(self, string: str, label: bool) -> float:
        """Get probability of string having given label (matching Weiss)."""
        # Get the actual device of the model parameters
        device = next(self.parameters()).device
        
        if len(string) == 0:
            x = torch.zeros(1, 0, dtype=torch.long, device=device)
        else:
            indices = [int(c) for c in string]
            x = torch.tensor([indices], dtype=torch.long, device=device)
        
        with torch.no_grad():
            logits = self.forward(x)
            probs = F.softmax(logits, dim=1)
            if label:
                return probs[0, 1].item()
            else:
                return probs[0, 0].item()


class LSTMClassifier(RNNClassifier):
    """LSTM version matching Weiss exactly."""
    def __init__(self, **kwargs):
        kwargs['rnn_type'] = 'lstm'
        super().__init__(**kwargs)


class GRUClassifier(RNNClassifier):
    """GRU version of the RNN classifier."""
    def __init__(self, **kwargs):
        kwargs['rnn_type'] = 'gru'
        super().__init__(**kwargs)


class VanillaRNNClassifier(nn.Module):
    """Vanilla RNN classifier (using tanh activation)."""
    
    def __init__(
        self,
        alphabet_size: int = 2,
        embedding_dim: int = 3,
        hidden_dim: int = 5,
        num_layers: int = 2,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        super().__init__()
        
        self.alphabet_size = alphabet_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.device = device
        
        # Embedding layer
        self.embedding = nn.Embedding(alphabet_size, embedding_dim)
        
        # Learnable initial state
        self.h0 = nn.Parameter(torch.randn(num_layers, 1, hidden_dim))
        with torch.no_grad():
            self.h0.clamp_(-1, 1)
        
        # Vanilla RNN layer
        self.rnn = nn.RNN(
            embedding_dim, hidden_dim, num_layers,
            batch_first=True, nonlinearity='tanh'
        )
        
        # Output layer
        self.output = nn.Linear(hidden_dim, 2)
        
        self.to(device)
    
    def get_initial_state(self, batch_size: int):
        """Get initial hidden state for batch."""
        h0 = self.h0.expand(-1, batch_size, -1).contiguous()
        return h0
    
    def forward(self, x: torch.Tensor, return_all_hidden: bool = False):
        """Forward pass for vanilla RNN."""
        if x.dim() == 1:
            x = x.unsqueeze(0)
        
        batch_size, seq_len = x.shape
        
        # Special handling for empty sequences
        if seq_len == 0:
            h0 = self.get_initial_state(batch_size)
            final_hidden = h0[-1]  # Last layer
            logits = self.output(final_hidden)
            if return_all_hidden:
                return logits, h0.transpose(0, 1).contiguous().view(batch_size, -1)
            return logits
        
        # Standard forward pass
        embedded = self.embedding(x)
        h0 = self.get_initial_state(batch_size)
        
        output, hidden = self.rnn(embedded, h0)
        
        # Get final hidden state from last timestep
        final_hidden = output[:, -1, :]
        logits = self.output(final_hidden)
        
        if return_all_hidden:
            # For vanilla RNN, return flattened hidden states
            return logits, hidden.transpose(0, 1).contiguous().view(batch_size, -1)
        
        return logits
    
    def classify_string(self, string: str) -> bool:
        """Classify a string as True/False."""
        device = next(self.parameters()).device
        
        if len(string) == 0:
            x = torch.zeros(1, 0, dtype=torch.long, device=device)
        else:
            indices = [int(c) for c in string]
            x = torch.tensor([indices], dtype=torch.long, device=device)
        
        with torch.no_grad():
            logits = self.forward(x)
            probs = F.softmax(logits, dim=1)
            return probs[0, 1].item() > 0.5
    
    def get_state_vector(self, string: str) -> torch.Tensor:
        """Get state vector after processing string."""
        device = next(self.parameters()).device
        
        if len(string) == 0:
            x = torch.zeros(1, 0, dtype=torch.long, device=device)
        else:
            indices = [int(c) for c in string]
            x = torch.tensor([indices], dtype=torch.long, device=device)
        
        with torch.no_grad():
            _, hidden_states = self.forward(x, return_all_hidden=True)
            return hidden_states[0]
    
    def get_proba(self, string: str, label: bool) -> float:
        """Get probability of string having given label."""
        device = next(self.parameters()).device
        
        if len(string) == 0:
            x = torch.zeros(1, 0, dtype=torch.long, device=device)
        else:
            indices = [int(c) for c in string]
            x = torch.tensor([indices], dtype=torch.long, device=device)
        
        with torch.no_grad():
            logits = self.forward(x)
            probs = F.softmax(logits, dim=1)
            if label:
                return probs[0, 1].item()
            else:
                return probs[0, 0].item()