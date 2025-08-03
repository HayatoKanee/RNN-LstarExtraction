#!/usr/bin/env python
"""
Train PyTorch RNNs on Tomita grammars with early stopping at target accuracy.
This simulates realistic training where practitioners stop at "good enough" accuracy.
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
from collections import defaultdict

# Add parent directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.rnn_classifier import LSTMClassifier
from grammars.tomita import TOMITA_GRAMMARS
from train_tomita_grammars import make_train_set_for_target


def mixed_curriculum_train_with_early_stop(model, train_set, device='cuda',
                          target_accuracy=0.95, 
                          outer_loops=3, stop_threshold=0.001,
                          learning_rate=0.001, length_epochs=5,
                          random_batch_epochs=100, single_batch_epochs=100,
                          random_batch_size=20):
    """Modified curriculum training that stops at target accuracy."""
    
    # Use Adam optimizer like Weiss
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Group by length
    lengths = sorted(list(set(len(w) for w in train_set)))
    length_groups = defaultdict(list)
    for word, label in train_set.items():
        length_groups[len(word)].append((word, label))
    
    all_losses = []
    
    def check_accuracy():
        """Check current model accuracy."""
        model.eval()
        correct = 0
        with torch.no_grad():
            for word, label in train_set.items():
                pred = model.classify_string(word)
                if pred == label:
                    correct += 1
        model.train()
        return correct / len(train_set)
    
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
                
                # Check if we've reached target accuracy
                if epoch % 5 == 0:
                    current_acc = check_accuracy()
                    if current_acc >= target_accuracy:
                        print(f"Early stopping: reached {current_acc:.4f} accuracy (target: {target_accuracy})")
                        return all_losses, current_acc
                
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
                current_acc = check_accuracy()
                print(f"Epoch {epoch}: loss={avg_loss:.4f}, accuracy={current_acc:.4f}")
                
                if current_acc >= target_accuracy:
                    print(f"Early stopping: reached {current_acc:.4f} accuracy (target: {target_accuracy})")
                    return all_losses, current_acc
            
            # Check convergence
            if avg_loss < stop_threshold:
                print(f"Loss converged at {avg_loss}")
                current_acc = check_accuracy()
                return all_losses, current_acc
        
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
                current_acc = check_accuracy()
                print(f"Epoch {epoch}: loss={avg_loss:.4f}, accuracy={current_acc:.4f}")
                
                if current_acc >= target_accuracy:
                    print(f"Early stopping: reached {current_acc:.4f} accuracy (target: {target_accuracy})")
                    return all_losses, current_acc
            
            if avg_loss < stop_threshold:
                print(f"Loss converged at {avg_loss}")
                current_acc = check_accuracy()
                return all_losses, current_acc
    
    # Training complete
    final_acc = check_accuracy()
    print(f"Training complete. Final accuracy: {final_acc:.4f}")
    return all_losses, final_acc


def train_with_early_stop(args):
    """Train models with early stopping at target accuracy."""
    
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
    
    results = []
    
    for grammar_id in args.grammars:
        grammar_func, description = TOMITA_GRAMMARS[grammar_id]
        
        print(f"\n{'='*60}")
        print(f"Training Tomita {grammar_id} with early stopping")
        print(f"Description: {description}")
        print(f"Target accuracy: {args.target_accuracy:.1%}")
        print(f"{'='*60}")
        
        try:
            # Create model
            model = LSTMClassifier(
                alphabet_size=2,
                embedding_dim=3,
                hidden_dim=args.hidden_dim,
                num_layers=2,
                device=device
            )
            
            # Generate training data
            print("\nGenerating training data...")
            train_set = make_train_set_for_target(grammar_func)
            
            # Train with early stopping
            print("\nTraining with early stopping...")
            start_time = time.time()
            
            losses, final_accuracy = mixed_curriculum_train_with_early_stop(
                model, train_set, device=device,
                target_accuracy=args.target_accuracy,
                stop_threshold=0.0005
            )
            
            training_time = time.time() - start_time
            
            print(f"\nTraining complete:")
            print(f"  Final accuracy: {final_accuracy:.4f}")
            print(f"  Training time: {training_time:.2f}s")
            print(f"  Final loss: {losses[-1] if losses else 'N/A'}")
            
            # Save model
            model_path = os.path.join(
                args.output_dir,
                f"tomita{grammar_id}_lstm_h{args.hidden_dim}.pt"
            )
            
            torch.save({
                'model_state_dict': model.state_dict(),
                'model_config': {
                    'alphabet_size': 2,
                    'embedding_dim': 3,
                    'hidden_dim': args.hidden_dim,
                    'num_layers': 2,
                    'rnn_type': 'lstm'
                },
                'grammar_id': grammar_id,
                'accuracy': final_accuracy,
                'target_accuracy': args.target_accuracy,
                'training_time': training_time,
                'final_loss': losses[-1] if losses else None
            }, model_path)
            
            results.append({
                'grammar_id': grammar_id,
                'accuracy': final_accuracy,
                'target_accuracy': args.target_accuracy,
                'training_time': training_time,
                'final_loss': losses[-1] if losses else None,
                'model_path': model_path
            })
            
            print(f"Model saved to: {model_path}")
            
        except Exception as e:
            print(f"Error training grammar {grammar_id}: {e}")
            import traceback
            traceback.print_exc()
    
    # Save summary
    summary_file = os.path.join(args.output_dir, f'training_summary_{int(args.target_accuracy*100)}pct.json')
    with open(summary_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "="*60)
    print(f"TRAINING COMPLETE (Target: {args.target_accuracy:.1%})")
    print("="*60)
    print("\nSummary:")
    for r in results:
        print(f"Tomita {r['grammar_id']}: "
              f"acc={r['accuracy']:.4f} (target: {r['target_accuracy']:.3f}), "
              f"time={r['training_time']:.1f}s")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Train RNNs with early stopping at target accuracy"
    )
    
    parser.add_argument('--grammars', type=int, nargs='+',
                       default=list(range(1, 8)),
                       help='Grammar IDs to train (default: 1-7)')
    parser.add_argument('--target-accuracy', type=float, default=0.95,
                       help='Target accuracy for early stopping (default: 0.95)')
    parser.add_argument('--hidden-dim', type=int, default=50,
                       help='Hidden dimension (default: 50)')
    parser.add_argument('--output-dir', type=str,
                       default='trained_models_early_stop',
                       help='Output directory')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Device to use')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    train_with_early_stop(args)


if __name__ == "__main__":
    main()