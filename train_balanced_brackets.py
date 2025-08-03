#!/usr/bin/env python3
"""
Train an RNN on the balanced brackets language.
This is a context-free language that cannot be represented by any DFA.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import sys
import time
from typing import List, Tuple

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.rnn_classifier import LSTMClassifier
from grammars.balanced_brackets import is_balanced_brackets


class BracketDataset(Dataset):
    """Dataset for balanced brackets - no padding, variable length."""
    
    def __init__(self, data: List[Tuple[str, bool]], alphabet: List[str]):
        self.data = data
        self.alphabet = alphabet
        self.char_to_idx = {char: idx for idx, char in enumerate(alphabet)}
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        string, label = self.data[idx]
        
        # Convert string to indices (no padding)
        indices = [self.char_to_idx[char] for char in string]
        
        return indices, label


def train_rnn_on_language(
    language_name: str,
    is_accepted_fn,
    alphabet: List[str],
    hidden_dim: int = 50,
    num_epochs: int = 100,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    num_samples: int = 10000,
    max_length: int = 20
):
    """Train an RNN on a language (regular or non-regular)."""
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Training on {device}")
    
    # Generate dataset
    print(f"\nGenerating {language_name} dataset...")
    from grammars.balanced_brackets import generate_dataset
    dataset = generate_dataset(num_samples, max_length)
    
    # Split into train/val/test
    np.random.shuffle(dataset)
    n = len(dataset)
    train_data = dataset[:int(0.7*n)]
    val_data = dataset[int(0.7*n):int(0.85*n)]
    test_data = dataset[int(0.85*n):]
    
    print(f"Dataset sizes - Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
    
    # Count positive examples
    train_pos = sum(1 for _, label in train_data if label)
    print(f"Train set balance: {train_pos}/{len(train_data)} positive ({train_pos/len(train_data)*100:.1f}%)")
    
    # Create data loaders (no padding)
    train_dataset = BracketDataset(train_data, alphabet)
    val_dataset = BracketDataset(val_data, alphabet)
    test_dataset = BracketDataset(test_data, alphabet)
    
    # Custom collate function for variable length sequences
    def collate_fn(batch):
        # For single sample training like Tomita
        return batch
    
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=1, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=1, collate_fn=collate_fn)
    
    # Create model 
    model = LSTMClassifier(
        alphabet_size=len(alphabet), 
        embedding_dim=3,
        hidden_dim=hidden_dim,
        num_layers=2,
        device=device
    )
    
    # Loss and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    best_val_acc = 0
    best_model_state = None
    
    print("\nTraining...")
    for epoch in range(num_epochs):
        # Train
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for batch in train_loader:
            # Single sample per batch
            indices, label = batch[0]
            
            # Convert to tensor
            if len(indices) == 0:
                x = torch.zeros(1, 0, dtype=torch.long, device=device)
            else:
                x = torch.tensor([indices], dtype=torch.long, device=device)
            
            # Forward pass
            optimizer.zero_grad()
            logits = model(x)  # Shape: [1, 2]
            probs = torch.softmax(logits, dim=1)
            
            # Use negative log likelihood loss like Tomita training
            if label:
                p = probs[0, 1]  # Probability of True
            else:
                p = probs[0, 0]  # Probability of False
            
            loss = -torch.log(p + 1e-10)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            # Calculate accuracy
            prediction = probs[0, 1].item() > 0.5
            train_correct += (prediction == label)
            train_total += 1
        
        # Validate
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                indices, label = batch[0]
                
                # Convert to tensor
                if len(indices) == 0:
                    x = torch.zeros(1, 0, dtype=torch.long, device=device)
                else:
                    x = torch.tensor([indices], dtype=torch.long, device=device)
                
                # Forward pass
                logits = model(x)
                probs = torch.softmax(logits, dim=1)
                
                # Loss
                if label:
                    p = probs[0, 1]
                else:
                    p = probs[0, 0]
                
                loss = -torch.log(p + 1e-10)
                val_loss += loss.item()
                
                # Accuracy
                prediction = probs[0, 1].item() > 0.5
                val_correct += (prediction == label)
                val_total += 1
        
        train_acc = train_correct / train_total
        val_acc = val_correct / val_total
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{num_epochs} - "
                  f"Train Loss: {train_loss/len(train_loader):.4f}, "
                  f"Train Acc: {train_acc:.4f}, "
                  f"Val Acc: {val_acc:.4f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict()
    
    # Load best model
    model.load_state_dict(best_model_state)
    
    # Test
    model.eval()
    test_correct = 0
    test_total = 0
    
    with torch.no_grad():
        for batch in test_loader:
            indices, label = batch[0]
            
            # Convert to tensor
            if len(indices) == 0:
                x = torch.zeros(1, 0, dtype=torch.long, device=device)
            else:
                x = torch.tensor([indices], dtype=torch.long, device=device)
            
            # Forward pass
            logits = model(x)
            probs = torch.softmax(logits, dim=1)
            
            # Accuracy
            prediction = probs[0, 1].item() > 0.5
            test_correct += (prediction == label)
            test_total += 1
    
    test_acc = test_correct / test_total
    print(f"\nFinal Test Accuracy: {test_acc:.4f}")
    
    # Save model
    save_dir = 'trained_models_imperfect'
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'{language_name}_lstm_h{hidden_dim}.pt')
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': {
            'alphabet_size': len(alphabet),
            'embedding_dim': 3,
            'hidden_dim': hidden_dim,
            'num_layers': 2,
            'rnn_type': 'lstm'
        },
        'alphabet': alphabet,
        'test_accuracy': test_acc,
        'language': language_name,
        'final_loss': train_loss / train_total
    }, save_path)
    
    print(f"Model saved to {save_path}")
    
    # Test on some specific examples
    print("\nTesting on specific examples:")
    test_examples = [
        "",
        "()",
        "(())",
        "((()))",
        "()()",
        "(()())",
        "(",
        ")",
        ")(",
        "(()",
        "())",
        "((())",
        "(()))"
    ]
    
    model.eval()
    with torch.no_grad():
        for example in test_examples:
            # Convert to tensor (no padding)
            char_to_idx = {char: idx for idx, char in enumerate(alphabet)}
            if len(example) == 0:
                x = torch.zeros(1, 0, dtype=torch.long, device=device)
            else:
                indices = [char_to_idx[c] for c in example]
                x = torch.tensor([indices], dtype=torch.long, device=device)
            
            # Forward pass
            logits = model(x)
            probs = torch.softmax(logits, dim=1)
            prob_true = probs[0, 1].item()
            
            prediction = prob_true > 0.5
            actual = is_balanced_brackets(example)
            status = "✓" if prediction == actual else "✗"
            
            print(f"  '{example}' - Predicted: {prediction}, Actual: {actual}, "
                  f"Prob: {prob_true:.3f} {status}")
    
    return model, test_acc


def main():
    """Train RNN on balanced brackets language."""
    
    print("="*80)
    print("Training RNN on Balanced Brackets Language")
    print("="*80)
    
    model, acc = train_rnn_on_language(
        language_name='balanced_brackets',
        is_accepted_fn=is_balanced_brackets,
        alphabet=['(', ')'],
        hidden_dim=50,
        num_epochs=50,
        num_samples=10000,
        max_length=20
    )
    
    print(f"\n{'='*80}")
    print("Summary:")
    print(f"Test Accuracy: {acc:.4f}")
    print("\nNote: RNNs can learn context-free languages like balanced brackets,")
    print("but DFA extraction should fail or produce poor approximations since")
    print("these languages cannot be represented by finite automata.")


if __name__ == "__main__":
    main()