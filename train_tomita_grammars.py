#!/usr/bin/env python
"""
Train and test PyTorch RNNs on Tomita grammars.
This is the main script for training models compatible with DFA extraction.
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import json
import time
import argparse
import itertools
from collections import defaultdict

# Add parent directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.rnn_classifier import LSTMClassifier
from grammars.tomita import TOMITA_GRAMMARS


def n_words_of_length(n, length, alphabet):
    """Helper function for generating words."""
    if 50 * n >= pow(len(alphabet), length):
        # If asking for more than 1/50th of all possible strings, generate all and sample
        all_words = [''.join(list(b)) for b in itertools.product(alphabet, repeat=length)]
        random.shuffle(all_words)
        return all_words[:n]
    else:
        # Otherwise generate randomly
        res = set()
        while len(res) < n:
            word = ''.join(random.choice(alphabet) for _ in range(length))
            res.add(word)
        return list(res)


def make_train_set_for_target(target, alphabet="01", lengths=None, 
                             max_train_samples_per_length=300,
                             search_size_per_length=1000):
    """Generate training data for a target language."""
    train_set = {}
    
    if lengths is None:
        lengths = list(range(15)) + [15, 20, 25, 30]
    
    for l in lengths:
        # Generate samples 
        if l == 0:
            samples = ['']
        else:
            samples = n_words_of_length(search_size_per_length, l, alphabet)
        
        # Classify samples
        pos = [w for w in samples if target(w)]
        neg = [w for w in samples if not target(w)]
        
        # Balance classes
        pos = pos[:int(max_train_samples_per_length/2)]
        neg = neg[:int(max_train_samples_per_length/2)]
        
        minority = min(len(pos), len(neg))
        pos = pos[:minority+20] 
        neg = neg[:minority+20]
        
        # Add to training set
        train_set.update({w: True for w in pos})
        train_set.update({w: False for w in neg})
    
    pos_count = sum(1 for v in train_set.values() if v)
    print(f"made train set of size: {len(train_set)}, of which positive examples: {pos_count}")
    
    return train_set


def mixed_curriculum_train(model, train_set, device='cuda',
                          outer_loops=3, stop_threshold=0.001,
                          learning_rate=0.001, length_epochs=5,
                          random_batch_epochs=100, single_batch_epochs=100,
                          random_batch_size=20):
    """Exact replication of Weiss's mixed curriculum training."""
    
    # Use Adam optimizer like Weiss
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Group by length
    lengths = sorted(list(set(len(w) for w in train_set)))
    length_groups = defaultdict(list)
    for word, label in train_set.items():
        length_groups[len(word)].append((word, label))
    
    all_losses = []
    
    for outer in range(outer_loops):
        # Stage 1: Length curriculum
        for l in lengths:
            training = length_groups[l]
            
            # Skip if only one classification
            if len(set(label for _, label in training)) <= 1:
                continue
            
            # Train on this length
            for epoch in range(length_epochs):
                # Process ALL strings of this length together 
                total_loss = 0
                
                for word, label in training:
                    # Convert to tensor
                    if len(word) == 0:
                        x = torch.zeros(1, 0, dtype=torch.long, device=device)
                    else:
                        indices = [int(c) for c in word]
                        x = torch.tensor([indices], dtype=torch.long, device=device)
                    
                    # Forward pass
                    logits = model(x)
                    probs = torch.softmax(logits, dim=1)
                    
                    # -log(p) where p is prob of correct label
                    if label:
                        p = probs[0, 1]  # Probability of True
                    else:
                        p = probs[0, 0]  # Probability of False
                    
                    loss = -torch.log(p + 1e-10)  # Add epsilon for stability
                    
                    # Backward
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    total_loss += loss.item()
                
                avg_loss = total_loss / len(training)
                
                # Check convergence
                if avg_loss < stop_threshold:
                    break
        
        # Stage 2: Random batches
        all_data = [(w, l) for w, l in train_set.items()]
        
        for epoch in range(random_batch_epochs):
            random.shuffle(all_data)
            epoch_loss = 0
            
            # Process in batches
            for i in range(0, len(all_data), random_batch_size):
                batch = all_data[i:i+random_batch_size]
                batch_loss = 0
                
                for word, label in batch:
                    if len(word) == 0:
                        x = torch.zeros(1, 0, dtype=torch.long, device=device)
                    else:
                        indices = [int(c) for c in word]
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
                    
                    batch_loss += loss.item()
                
                epoch_loss += batch_loss
            
            avg_loss = epoch_loss / len(all_data)
            all_losses.append(avg_loss)
            
            if epoch % 20 == 0:
                print(f"current average loss is: {avg_loss}")
            
            # Check convergence
            if avg_loss < stop_threshold:
                print(f"classification loss on last batch was: {avg_loss}")
                return all_losses
        
        # Stage 3: Single batch (all data together)
        for epoch in range(single_batch_epochs):
            total_loss = 0
            
            for word, label in all_data:
                if len(word) == 0:
                    x = torch.zeros(1, 0, dtype=torch.long, device=device)
                else:
                    indices = [int(c) for c in word]
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
            all_losses.append(avg_loss)
            
            if epoch % 20 == 0:
                print(f"current average loss is: {avg_loss}")
            
            if avg_loss < stop_threshold:
                print(f"classification loss on last batch was: {avg_loss}")
                return all_losses
    
    print(f"classification loss on last batch was: {all_losses[-1] if all_losses else 'N/A'}")
    return all_losses


def test_model(model_path, device='cuda'):
    """Test a trained model."""
    from teacher.rnn_oracle import RNNOracle
    
    print(f"\nTesting model: {model_path}")
    print("-" * 50)
    
    # Load model
    checkpoint = torch.load(model_path, map_location=device)
    config = checkpoint['model_config']
    grammar_id = checkpoint['grammar_id']
    
    # Create model
    model = LSTMClassifier(
        alphabet_size=config['alphabet_size'],
        embedding_dim=config['embedding_dim'],
        hidden_dim=config['hidden_dim'],
        num_layers=config['num_layers'],
        device=device
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Get grammar
    grammar_func, description = TOMITA_GRAMMARS[grammar_id]
    print(f"Grammar: Tomita {grammar_id} - {description}")
    print(f"Training accuracy: {checkpoint.get('accuracy', 'N/A'):.4f}")
    
    # Test with some strings
    test_strings = ['', '0', '1', '00', '01', '10', '11', '000', '111', '0101', '1010']
    correct = 0
    
    print("\nTest predictions:")
    for s in test_strings:
        pred = model.classify_string(s)
        truth = grammar_func(s)
        match = pred == truth
        correct += match
        print(f"  '{s}': pred={pred}, truth={truth} {'✓' if match else '✗'}")
    
    print(f"\nTest accuracy: {correct}/{len(test_strings)} = {correct/len(test_strings):.2f}")
    
    # Test RNNOracle compatibility
    print("\nTesting RNNOracle compatibility...")
    try:
        oracle = RNNOracle(model=model, alphabet="01", device=device)
        result = oracle.membership_query("01")
        print("✓ Model is compatible with RNNOracle!")
    except Exception as e:
        print(f"✗ RNNOracle error: {e}")


def train(args):
    """Train all Tomita grammars."""
    
    # Use GPU if requested
    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = 'cpu'
    print(f"Using device: {device}")
    
    # Set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device == 'cuda':
        torch.cuda.manual_seed(args.seed)
    
    # Output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Train each grammar with different hidden dimensions
    results = []
    
    for grammar_id in args.grammars:
        grammar_func, description = TOMITA_GRAMMARS[grammar_id]
        
        for hidden_dim in args.hidden_dims:
            print(f"\n{'='*60}")
            print(f"Training Tomita {grammar_id} - Hidden dim: {hidden_dim}")
            print(f"Description: {description}")
            print(f"{'='*60}")
            
            try:
                print(f"\nCreating LSTM model...")
                model = LSTMClassifier(
                    alphabet_size=2,
                    embedding_dim=3, 
                    hidden_dim=hidden_dim,
                    num_layers=2,
                    device=device
                )
                print(f"Model created successfully on {device}")
                
                # Generate training data
                print(f"\nGenerating training data...")
                train_set = make_train_set_for_target(grammar_func)
                
                start_time = time.time()
                
                losses = mixed_curriculum_train(
                    model, train_set, device=device,
                    stop_threshold=0.0005 
                )
                
                training_time = time.time() - start_time
                
                # Evaluate accuracy
                model.eval()
                correct = 0
                with torch.no_grad():
                    for word, label in train_set.items():
                        pred = model.classify_string(word)
                        if pred == label:
                            correct += 1
                
                accuracy = correct / len(train_set)
                print(f"\nTraining accuracy: {accuracy:.4f}")
                print(f"Training time: {training_time:.2f}s")
                
                # Save model
                model_path = os.path.join(
                    args.output_dir,
                    f"tomita{grammar_id}_lstm_h{hidden_dim}.pt"
                )
                
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'model_config': {
                        'alphabet_size': 2,
                        'embedding_dim': 3,
                        'hidden_dim': hidden_dim,
                        'num_layers': 2,
                        'rnn_type': 'lstm'
                    },
                    'grammar_id': grammar_id,
                    'accuracy': accuracy,
                    'training_time': training_time,
                    'final_loss': losses[-1] if losses else None
                }, model_path)
                
                results.append({
                    'grammar_id': grammar_id,
                    'hidden_dim': hidden_dim,
                    'accuracy': accuracy,
                    'training_time': training_time,
                    'final_loss': losses[-1] if losses else None,
                    'model_path': model_path
                })
                
                print(f"Model saved to: {model_path}")
                
            except Exception as e:
                print(f"Error training grammar {grammar_id} with h={hidden_dim}: {e}")
                import traceback
                traceback.print_exc()
    
    # Save summary
    with open(os.path.join(args.output_dir, 'training_summary.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print("\nSummary:")
    for r in results:
        print(f"Tomita {r['grammar_id']} (h={r['hidden_dim']}): "
              f"acc={r['accuracy']:.4f}, loss={r['final_loss']:.6f}, "
              f"time={r['training_time']:.1f}s")


def main():
    """Main entry point with command line interface."""
    parser = argparse.ArgumentParser(
        description="Train and test PyTorch RNNs on Tomita grammars (Weiss et al. approach)"
    )
    
    subparsers = parser.add_subparsers(dest='mode', help='Mode: train or test')
    
    # Train mode
    train_parser = subparsers.add_parser('train', help='Train models')
    train_parser.add_argument('--grammars', type=int, nargs='+', 
                             default=list(range(1, 8)),
                             help='Grammar IDs to train (default: 1-7)')
    train_parser.add_argument('--hidden-dims', type=int, nargs='+',
                             default=[50],
                             help='Hidden dimensions (default: 5 10 50)')
    train_parser.add_argument('--output-dir', type=str,
                             default='trained_models',
                             help='Output directory')
    train_parser.add_argument('--device', type=str, default='cuda',
                             choices=['cuda', 'cpu'],
                             help='Device to use')
    train_parser.add_argument('--seed', type=int, default=42,
                             help='Random seed')
    
    # Test mode
    test_parser = subparsers.add_parser('test', help='Test a model')
    test_parser.add_argument('model_path', type=str,
                            help='Path to model file')
    test_parser.add_argument('--device', type=str, default='cuda',
                            choices=['cuda', 'cpu'],
                            help='Device to use')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        train(args)
    elif args.mode == 'test':
        test_model(args.model_path, device=args.device)
    else:
        # Default: show usage
        parser.print_help()


if __name__ == "__main__":
    main()