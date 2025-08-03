#!/usr/bin/env python
"""
Specialized script to fix the empty string classification issue.
Focuses specifically on training models to correctly classify the empty string.
"""

import sys
import torch
import json
import numpy as np
import random
from pathlib import Path
from typing import Dict, List, Tuple
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, str(Path(__file__).parent.parent))
from models.rnn_classifier import LSTMClassifier, GRUClassifier, VanillaRNNClassifier
from core.dfa import DFA
from validation.validate_exhaustive import load_ground_truth_dfa, load_trained_model


def create_empty_string_focused_dataset(dfa: DFA, alphabet: List[str], 
                                      empty_string_weight: int = 10) -> Dict[str, bool]:
    """
    Create a dataset that heavily emphasizes the empty string.
    """
    data = {}
    
    # Add empty string multiple times
    empty_label = dfa.classify_word("")
    for _ in range(empty_string_weight):
        # Use slightly different keys to avoid overwriting
        data[f"EMPTY_{_}"] = empty_label
    
    # Add other short strings for contrast
    for symbol in alphabet:
        data[symbol] = dfa.classify_word(symbol)
    
    # Add some two-character strings
    for s1 in alphabet:
        for s2 in alphabet:
            data[s1 + s2] = dfa.classify_word(s1 + s2)
    
    # Add some three-character strings
    for s1 in alphabet:
        for s2 in alphabet:
            for s3 in alphabet:
                data[s1 + s2 + s3] = dfa.classify_word(s1 + s2 + s3)
    
    print(f"Created focused dataset with {len(data)} examples")
    print(f"Empty string label: {empty_label}")
    
    return data


def train_with_empty_string_focus(model, dfa: DFA, alphabet: List[str], 
                                 device: str = 'cuda', max_epochs: int = 500) -> float:
    """
    Train model with special focus on empty string classification.
    """
    char_to_idx = {char: idx for idx, char in enumerate(alphabet)}
    
    # Create focused dataset
    data = create_empty_string_focused_dataset(dfa, alphabet, empty_string_weight=20)
    
    # Convert to training examples
    examples = []
    for key, label in data.items():
        if key.startswith("EMPTY_"):
            string = ""
        else:
            string = key
        examples.append((string, label))
    
    # Optimizer with lower learning rate
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    
    print(f"Training with {len(examples)} examples...")
    
    best_accuracy = 0.0
    empty_string_label = dfa.classify_word("")
    
    for epoch in range(max_epochs):
        model.train()
        total_loss = 0.0
        correct = 0
        empty_correct = 0
        empty_count = 0
        
        # Shuffle examples
        random.shuffle(examples)
        
        for string, label in examples:
            # Convert to tensor
            if len(string) == 0:
                x = torch.zeros(1, 0, dtype=torch.long, device=device)
                empty_count += 1
            else:
                indices = [char_to_idx[c] for c in string]
                x = torch.tensor([indices], dtype=torch.long, device=device)
            
            # Forward pass
            logits = model(x)
            probs = torch.softmax(logits, dim=1)
            
            # Compute loss with extra weight for empty string
            if len(string) == 0:
                # Extra weight for empty string
                if label:
                    p = probs[0, 1]
                else:
                    p = probs[0, 0]
                loss = -5.0 * torch.log(p + 1e-10)  # 5x weight for empty string
            else:
                if label:
                    p = probs[0, 1]
                else:
                    p = probs[0, 0]
                loss = -torch.log(p + 1e-10)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # Check prediction
            pred = logits.argmax(dim=1).item()
            if pred == (1 if label else 0):
                correct += 1
                if len(string) == 0:
                    empty_correct += 1
        
        # Calculate accuracy
        accuracy = correct / len(examples)
        empty_accuracy = empty_correct / empty_count if empty_count > 0 else 0
        avg_loss = total_loss / len(examples)
        
        # Test empty string specifically
        model.eval()
        with torch.no_grad():
            x = torch.zeros(1, 0, dtype=torch.long, device=device)
            logits = model(x)
            pred = logits.argmax(dim=1).item()
            empty_pred_correct = (pred == (1 if empty_string_label else 0))
        
        if (epoch + 1) % 50 == 0 or empty_pred_correct:
            print(f"  Epoch {epoch+1}: Accuracy={accuracy:.4f}, EmptyAcc={empty_accuracy:.4f}, "
                  f"Loss={avg_loss:.6f}, EmptyPred={'✓' if empty_pred_correct else '✗'}")
        
        # Stop if we got empty string correct
        if empty_pred_correct and accuracy > 0.95:
            print(f"  Success! Empty string classified correctly at epoch {epoch+1}")
            return accuracy
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
    
    return accuracy


def fix_model_empty_string(model_path: Path, dfa_path: Path, device: str = 'cuda'):
    """
    Fix a model's empty string classification.
    """
    # Load model
    model, metadata = load_trained_model(model_path, device)
    
    # Load ground truth DFA
    dfa = load_ground_truth_dfa(dfa_path)
    alphabet = metadata['alphabet']
    
    print(f"\nFixing empty string for {model_path.parent.name}/{model_path.name}")
    print(f"Ground truth empty string label: {dfa.classify_word('')}")
    
    # Test current behavior
    model.eval()
    with torch.no_grad():
        x = torch.zeros(1, 0, dtype=torch.long, device=device)
        logits = model(x)
        pred = logits.argmax(dim=1).item()
        print(f"Current prediction: {bool(pred)} (logits: {logits[0].tolist()})")
    
    # Train with focus on empty string
    final_acc = train_with_empty_string_focus(model, dfa, alphabet, device)
    
    # Test again
    model.eval()
    with torch.no_grad():
        x = torch.zeros(1, 0, dtype=torch.long, device=device)
        logits = model(x)
        pred = logits.argmax(dim=1).item()
        print(f"\nFinal prediction: {bool(pred)} (logits: {logits[0].tolist()})")
    
    # Save if successful
    if pred == (1 if dfa.classify_word('') else 0):
        torch.save(model.state_dict(), model_path / 'model_fixed.pt')
        metadata['empty_string_fixed'] = True
        with open(model_path / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        print("Model saved with fix!")
        return True
    else:
        print("Failed to fix empty string classification")
        return False


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Fix empty string classification issues"
    )
    
    parser.add_argument('--model-dir', type=str, default='validation/trained_models')
    parser.add_argument('--dfa-dir', type=str, default='validation/random_dfas')
    parser.add_argument('--dataset-id', type=str, default='alphabet2_states5_v1')
    parser.add_argument('--architecture', type=str, default='lstm')
    parser.add_argument('--device', type=str, default='cuda')
    
    args = parser.parse_args()
    
    model_path = Path(args.model_dir) / args.dataset_id / args.architecture
    dfa_path = Path(args.dfa_dir) / args.dataset_id / 'dfa.json'
    
    if not model_path.exists() or not dfa_path.exists():
        print(f"Model or DFA not found")
        return
    
    # Set seeds for reproducibility
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    fix_model_empty_string(model_path, dfa_path, args.device)


if __name__ == "__main__":
    main()