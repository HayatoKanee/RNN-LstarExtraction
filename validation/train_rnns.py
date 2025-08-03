#!/usr/bin/env python
"""
Train RNNs on Generated Datasets

This script trains RNN models (LSTM, GRU, vanilla) on datasets generated from DFAs.
Supports multiple architectures and saves trained models for extraction validation.

Usage:
    python train_rnns.py --dataset-dir validation/datasets
    python train_rnns.py --dataset-id alphabet2_states5_v1 --architectures lstm gru
"""

import os
import sys
import json
import time
import torch
import random
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))
from models.rnn_classifier import LSTMClassifier, GRUClassifier, VanillaRNNClassifier


class StringDataset(Dataset):
    """Dataset for string classification."""
    
    def __init__(self, data: Dict[str, bool], alphabet: List[str], max_length: int = 100):
        """
        Initialize dataset.
        
        Args:
            data: Dictionary mapping strings to labels
            alphabet: List of alphabet symbols
            max_length: Maximum string length (for padding)
        """
        self.strings = list(data.keys())
        self.labels = [data[s] for s in self.strings]
        self.alphabet = alphabet
        self.char_to_idx = {char: idx for idx, char in enumerate(alphabet)}
        self.max_length = max_length
    
    def __len__(self):
        return len(self.strings)
    
    def __getitem__(self, idx):
        string = self.strings[idx]
        label = self.labels[idx]
        
        # Convert string to indices
        indices = [self.char_to_idx[char] for char in string if char in self.char_to_idx]
        
        # Pad or truncate
        if len(indices) < self.max_length:
            indices += [0] * (self.max_length - len(indices))
        else:
            indices = indices[:self.max_length]
        
        return torch.tensor(indices, dtype=torch.long), torch.tensor(1 if label else 0, dtype=torch.long)


def load_dataset(dataset_path: Path) -> Tuple[Dict[str, bool], List[str]]:
    """Load dataset and detect alphabet."""
    # Load dataset
    with open(dataset_path / 'dataset.json', 'r') as f:
        dataset_data = json.load(f)
    
    data = dataset_data['data']
    
    # Detect alphabet from the data
    alphabet_set = set()
    for string in data.keys():
        alphabet_set.update(string)
    
    # Sort alphabet for consistency
    alphabet = sorted(list(alphabet_set))
    
    # Handle empty strings
    if '' in data:
        # Don't add empty string to alphabet
        pass
    
    return data, alphabet


def mixed_curriculum_train(model, data: Dict[str, bool], alphabet: List[str],
                          device: str = 'cpu',
                          outer_loops: int = 3,
                          stop_threshold: float = 0.00001,
                          learning_rate: float = 0.001,
                          length_epochs: int = 5,
                          random_batch_epochs: int = 200,
                          single_batch_epochs: int = 200,
                          random_batch_size: int = 20) -> float:
    """
    Mixed curriculum training following Weiss et al.'s approach.
    
    Returns:
        Final accuracy achieved
    """
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Group by length
    length_groups = defaultdict(list)
    for word, label in data.items():
        length_groups[len(word)].append((word, label))
    
    lengths = sorted(length_groups.keys())
    
    # Convert alphabet to char->idx mapping
    char_to_idx = {char: idx for idx, char in enumerate(alphabet)}
    
    for outer in range(outer_loops):
        print(f"        Outer loop {outer + 1}/{outer_loops}")
        
        # Stage 1: Length curriculum
        for l in lengths[:16]:
            training = length_groups[l]
            
            if len(set(label for _, label in training)) <= 1:
                continue
            
            if len(training) > 100:
                print(f"          Length {l}: {len(training)} examples")
            
            for epoch in range(length_epochs):
                total_loss = 0
                
                # Process in mini-batches for efficiency
                batch_size = min(32, len(training))
                for i in range(0, len(training), batch_size):
                    batch = training[i:i+batch_size]
                    batch_loss = 0
                    
                    for word, label in batch:
                        # Convert to tensor
                        if len(word) == 0:
                            x = torch.zeros(1, 0, dtype=torch.long, device=device)
                        else:
                            indices = [char_to_idx[c] for c in word if c in char_to_idx]
                            if not indices:
                                continue
                            x = torch.tensor([indices], dtype=torch.long, device=device)
                        
                        # Forward pass
                        logits = model(x)
                        probs = torch.softmax(logits, dim=1)
                        
                        # -log(p) loss
                        if label:
                            p = probs[0, 1]
                        else:
                            p = probs[0, 0]
                        
                        loss = -torch.log(p + 1e-10)
                        batch_loss += loss
                    
                    # Update after batch
                    optimizer.zero_grad()
                    (batch_loss / len(batch)).backward()
                    optimizer.step()
                    
                    total_loss += batch_loss.item()
                
                avg_loss = total_loss / len(training)
                if avg_loss < stop_threshold:
                    break
        
        # Stage 2: Random batches
        all_data = list(data.items())
        
        for epoch in range(random_batch_epochs):
            random.shuffle(all_data)
            epoch_loss = 0
            
            for i in range(0, len(all_data), random_batch_size):
                batch = all_data[i:i+random_batch_size]
                
                for word, label in batch:
                    if len(word) == 0:
                        x = torch.zeros(1, 0, dtype=torch.long, device=device)
                    else:
                        indices = [char_to_idx[c] for c in word if c in char_to_idx]
                        if not indices:
                            continue
                        x = torch.tensor([indices], dtype=torch.long, device=device)
                    
                    logits = model(x)
                    probs = torch.softmax(logits, dim=1)
                    
                    if label:
                        p = probs[0, 1]
                    else:
                        p = probs[0, 0]
                    
                    loss = -torch.log(p + 1e-10)
                    
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    epoch_loss += loss.item()
            
            avg_loss = epoch_loss / len(all_data)
            
            if epoch % 10 == 0 or epoch < 5:
                print(f"          Random batch epoch {epoch}: avg loss = {avg_loss:.6f}")
            
            if avg_loss < stop_threshold:
                print(f"          Converged with loss {avg_loss:.6f}")
                return compute_accuracy(model, data, alphabet, device)
        
        # Stage 3: Single batch (all data)
        for epoch in range(single_batch_epochs):
            total_loss = 0
            
            for word, label in all_data:
                if len(word) == 0:
                    x = torch.zeros(1, 0, dtype=torch.long, device=device)
                else:
                    indices = [char_to_idx[c] for c in word if c in char_to_idx]
                    if not indices:
                        continue
                    x = torch.tensor([indices], dtype=torch.long, device=device)
                
                logits = model(x)
                probs = torch.softmax(logits, dim=1)
                
                if label:
                    p = probs[0, 1]
                else:
                    p = probs[0, 0]
                
                loss = -torch.log(p + 1e-10)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(all_data)
            
            if epoch % 10 == 0 or epoch < 5:
                print(f"          Single batch epoch {epoch}: avg loss = {avg_loss:.6f}")
            
            if avg_loss < stop_threshold:
                print(f"          Converged with loss {avg_loss:.6f}")
                return compute_accuracy(model, data, alphabet, device)
    
    return compute_accuracy(model, data, alphabet, device)


def compute_accuracy(model, data: Dict[str, bool], alphabet: List[str], device: str) -> float:
    """Compute accuracy on the full dataset."""
    model.eval()
    correct = 0
    total = 0
    
    char_to_idx = {char: idx for idx, char in enumerate(alphabet)}
    
    with torch.no_grad():
        for word, label in data.items():
            if len(word) == 0:
                x = torch.zeros(1, 0, dtype=torch.long, device=device)
            else:
                indices = [char_to_idx[c] for c in word if c in char_to_idx]
                if not indices:
                    continue
                x = torch.tensor([indices], dtype=torch.long, device=device)
            
            logits = model(x)
            pred = logits.argmax(dim=1).item()
            
            if pred == (1 if label else 0):
                correct += 1
            total += 1
    
    return correct / total if total > 0 else 0.0


def train_single_architecture(data: Dict[str, bool], 
                            alphabet: List[str],
                            architecture: str,
                            hidden_dim: int = 50,
                            device: str = 'cpu') -> Tuple[nn.Module, float]:
    """
    Train a single RNN architecture using mixed curriculum training.
    
    Returns:
        Trained model and final accuracy
    """
    # Create model
    alphabet_size = len(alphabet)
    
    if architecture == 'lstm':
        model = LSTMClassifier(
            alphabet_size=alphabet_size,
            embedding_dim=10,
            hidden_dim=hidden_dim,
            num_layers=2,
            device=device
        )
    elif architecture == 'gru':
        model = GRUClassifier(
            alphabet_size=alphabet_size,
            embedding_dim=10,
            hidden_dim=hidden_dim,
            num_layers=2,
            device=device
        )
    else:  # vanilla
        model = VanillaRNNClassifier(
            alphabet_size=alphabet_size,
            embedding_dim=10,
            hidden_dim=hidden_dim,
            num_layers=2,
            device=device
        )
    
    model = model.to(device)
    
    # Train model using mixed curriculum
    print(f"    Training {architecture.upper()} (hidden_dim={hidden_dim}) with mixed curriculum...")
    start_time = time.time()
    
    final_acc = mixed_curriculum_train(
        model, data, alphabet, device,
        outer_loops=3,
        learning_rate=0.001,
        length_epochs=5,
        random_batch_epochs=100,
        single_batch_epochs=100
    )
    
    training_time = time.time() - start_time
    print(f"    Training completed in {training_time:.2f}s, final accuracy: {final_acc:.4f}")
    
    if final_acc < 1.0:
        print("    Warning: Did not achieve 100% accuracy!")
        test_strings = ['', '0', '1', '00', '01', '10', '11']
        print("    Testing critical strings:")
        char_to_idx = {char: idx for idx, char in enumerate(alphabet)}
        
        with torch.no_grad():
            for s in test_strings:
                if len(s) == 0:
                    x = torch.zeros(1, 0, dtype=torch.long, device=device)
                else:
                    indices = [char_to_idx[c] for c in s if c in char_to_idx]
                    x = torch.tensor([indices], dtype=torch.long, device=device)
                
                logits = model(x)
                pred = logits.argmax(dim=1).item()
                true_label = data.get(s, None)
                
                if true_label is not None and pred != (1 if true_label else 0):
                    print(f"      ERROR: '{s}' - Expected: {true_label}, Got: {bool(pred)}")
    
    return model, final_acc


def save_model(model: nn.Module, output_path: Path, dataset_id: str, 
               architecture: str, accuracy: float, alphabet: List[str]):
    """Save trained model with metadata."""
    model_dir = output_path / dataset_id / architecture
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Save model state
    model_path = model_dir / 'model.pt'
    torch.save(model.state_dict(), model_path)
    
    # Save metadata
    metadata = {
        'dataset_id': dataset_id,
        'architecture': architecture,
        'accuracy': accuracy,
        'alphabet': alphabet,
        'alphabet_size': len(alphabet),
        'hidden_dim': model.hidden_dim,
        'num_layers': model.num_layers,
        'embedding_dim': model.embedding_dim
    }
    
    metadata_path = model_dir / 'metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"    Saved model to {model_dir}/")


def main():
    parser = argparse.ArgumentParser(
        description="Train RNNs on generated datasets"
    )
    
    parser.add_argument(
        '--dataset-dir',
        type=str,
        default='validation/datasets',
        help='Directory containing datasets'
    )
    
    parser.add_argument(
        '--dataset-id',
        type=str,
        help='Train on specific dataset ID only'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='validation/trained_models',
        help='Output directory for trained models'
    )
    
    parser.add_argument(
        '--architectures',
        type=str,
        nargs='+',
        default=['lstm', 'gru', 'vanilla'],
        choices=['lstm', 'gru', 'vanilla'],
        help='RNN architectures to train'
    )
    
    parser.add_argument(
        '--hidden-dim',
        type=int,
        default=50,
        help='Hidden dimension size (default: 50)'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Training device'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed (default: 42)'
    )
    
    args = parser.parse_args()
    
    # Set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # Find datasets to process
    dataset_path = Path(args.dataset_dir)
    
    if args.dataset_id:
        # Process specific dataset
        dataset_dirs = [dataset_path / args.dataset_id]
    else:
        # Process all datasets
        dataset_dirs = [d for d in dataset_path.iterdir() 
                       if d.is_dir() and (d / 'dataset.json').exists()]
    
    if not dataset_dirs:
        print(f"No datasets found in {dataset_path}")
        return
    
    # Create output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Training RNN Models")
    print(f"Dataset directory: {dataset_path}")
    print(f"Output directory: {output_path}")
    print(f"Architectures: {args.architectures}")
    print(f"Hidden dimension: {args.hidden_dim}")
    print(f"Device: {args.device}")
    print("-" * 60)
    
    # Process each dataset
    for dataset_dir in sorted(dataset_dirs):
        dataset_id = dataset_dir.name
        print(f"\nProcessing {dataset_id}...")
        
        # Load dataset
        try:
            data, alphabet = load_dataset(dataset_dir)
            print(f"  Loaded dataset: {len(data)} examples, alphabet {alphabet}")
            
            # Train each architecture
            for arch in args.architectures:
                model, accuracy = train_single_architecture(
                    data, alphabet, arch, 
                    hidden_dim=args.hidden_dim,
                    device=args.device
                )
                
                # Save model
                save_model(model, output_path, dataset_id, arch, accuracy, alphabet)
                
        except Exception as e:
            print(f"  Error processing dataset: {e}")
            continue
    
    print(f"\nTraining complete! Models saved to {output_path}/")


if __name__ == "__main__":
    main()