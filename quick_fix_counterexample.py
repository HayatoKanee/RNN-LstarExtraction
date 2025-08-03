#!/usr/bin/env python
"""
Quick script to fix a specific counterexample misclassification.
Example usage:
    python quick_fix_counterexample.py trained_models/tomita5_lstm_h50.pt "00001000000000111011"
"""

import os
import sys
import torch
import torch.optim as optim
import argparse

# Add parent directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.rnn_classifier import LSTMClassifier
from grammars.tomita import TOMITA_GRAMMARS


def main():
    parser = argparse.ArgumentParser(
        description="Quickly fix a counterexample misclassification"
    )
    parser.add_argument('model_path', help='Path to model')
    parser.add_argument('counterexample', help='The misclassified string')
    parser.add_argument('--epochs', type=int, default=100, help='Training epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    
    args = parser.parse_args()
    
    # Load model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    checkpoint = torch.load(args.model_path, map_location=device)
    config = checkpoint['model_config']
    grammar_id = checkpoint['grammar_id']
    
    model = LSTMClassifier(
        alphabet_size=config['alphabet_size'],
        embedding_dim=config['embedding_dim'],
        hidden_dim=config['hidden_dim'],
        num_layers=config['num_layers'],
        device=device
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Get correct label
    grammar_func, description = TOMITA_GRAMMARS[grammar_id]
    true_label = grammar_func(args.counterexample)
    
    print(f"Grammar: Tomita {grammar_id} - {description}")
    print(f"Counterexample: '{args.counterexample}'")
    print(f"Correct label: {true_label}")
    
    # Test current prediction
    current_pred = model.classify_string(args.counterexample)
    print(f"Current prediction: {current_pred} {'✓' if current_pred == true_label else '✗'}")
    
    if current_pred == true_label:
        print("Model already classifies this correctly!")
        return
    
    # Fine-tune on this example
    print(f"\nFine-tuning for {args.epochs} epochs...")
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Convert string to tensor
    indices = [int(c) for c in args.counterexample]
    x = torch.tensor([indices], dtype=torch.long, device=device)
    
    for epoch in range(args.epochs):
        # Forward pass
        logits = model(x)
        probs = torch.softmax(logits, dim=1)
        
        # Loss
        if true_label:
            p = probs[0, 1]
        else:
            p = probs[0, 0]
        loss = -torch.log(p + 1e-10)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}: loss={loss.item():.6f}, p(correct)={p.item():.4f}")
    
    # Test again
    model.eval()
    new_pred = model.classify_string(args.counterexample)
    print(f"\nNew prediction: {new_pred} {'✓' if new_pred == true_label else '✗'}")
    
    if new_pred == true_label:
        print("✓ Fixed!")
        
        # Save updated model (overwrite original)
        torch.save({
            'model_state_dict': model.state_dict(),
            'model_config': config,
            'grammar_id': grammar_id,
            'fixed_counterexample': args.counterexample,
            'accuracy': checkpoint.get('accuracy', 'unknown')
        }, args.model_path)
        print(f"Model updated: {args.model_path}")
    else:
        print("✗ Still incorrect. Try more epochs or lower learning rate.")


if __name__ == "__main__":
    main()