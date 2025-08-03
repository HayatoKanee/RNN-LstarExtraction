#!/usr/bin/env python
"""
Ensure all LSTM and GRU models achieve 100% accuracy.
"""

import sys
import torch
import json
import random
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))
from models.rnn_classifier import LSTMClassifier, GRUClassifier
from core.dfa import DFA
from validation.validate_exhaustive import generate_all_strings, load_ground_truth_dfa, load_trained_model
from validation.fix_empty_string import train_with_empty_string_focus


def check_model_accuracy(model, dfa: DFA, alphabet: List[str], max_length: int = 15, 
                        device: str = 'cuda') -> Tuple[float, List[Tuple[str, bool]]]:
    """
    Check model accuracy exhaustively and return accuracy and errors.
    """
    char_to_idx = {char: idx for idx, char in enumerate(alphabet)}
    all_strings = generate_all_strings(max_length, alphabet)
    
    total = 0
    correct = 0
    errors = []
    
    model.eval()
    with torch.no_grad():
        for length in range(max_length + 1):
            strings = all_strings[length]
            
            for string in strings:
                # Get ground truth
                dfa_result = dfa.classify_word(string)
                
                # Get model prediction
                if len(string) == 0:
                    x = torch.zeros(1, 0, dtype=torch.long, device=device)
                else:
                    indices = [char_to_idx[c] for c in string]
                    x = torch.tensor([indices], dtype=torch.long, device=device)
                
                logits = model(x)
                pred = logits.argmax(dim=1).item()
                model_result = bool(pred)
                
                total += 1
                if model_result == dfa_result:
                    correct += 1
                else:
                    errors.append((string, dfa_result))
    
    accuracy = correct / total if total > 0 else 0
    return accuracy, errors


def focused_training(model, dfa: DFA, alphabet: List[str], errors: List[Tuple[str, bool]], 
                    device: str = 'cuda', max_epochs: int = 200) -> float:
    """
    Train model focusing on error examples while maintaining overall performance.
    """
    char_to_idx = {char: idx for idx, char in enumerate(alphabet)}
    
    # Create training data that includes errors multiple times
    training_data = []
    
    # Add each error 10 times
    for string, label in errors:
        for _ in range(10):
            training_data.append((string, label))
    
    # Add some correct examples for balance
    # Generate examples of various lengths
    for length in range(min(10, max(len(e[0]) for e in errors) + 2)):
        if length == 0:
            strings = ['']
        else:
            # Generate a few random strings of this length
            num_strings = min(20, 2 ** length)
            strings = set()
            while len(strings) < num_strings:
                s = ''.join(random.choice(alphabet) for _ in range(length))
                strings.add(s)
        
        for s in strings:
            label = dfa.classify_word(s)
            training_data.append((s, label))
    
    # Shuffle training data
    random.shuffle(training_data)
    
    print(f"    Training on {len(training_data)} examples (including {len(errors)} errors repeated)")
    
    # Train with lower learning rate to preserve existing knowledge
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    best_accuracy = 0
    
    for epoch in range(max_epochs):
        model.train()
        total_loss = 0
        correct = 0
        
        # Shuffle each epoch
        random.shuffle(training_data)
        
        for string, label in training_data:
            # Convert to tensor
            if len(string) == 0:
                x = torch.zeros(1, 0, dtype=torch.long, device=device)
            else:
                indices = [char_to_idx[c] for c in string]
                x = torch.tensor([indices], dtype=torch.long, device=device)
            
            # Forward pass
            logits = model(x)
            probs = torch.softmax(logits, dim=1)
            
            # Compute loss with extra weight for original errors
            if (string, label) in errors:
                weight = 5.0  # Higher weight for error examples
            else:
                weight = 1.0
            
            if label:
                p = probs[0, 1]
            else:
                p = probs[0, 0]
            
            loss = -weight * torch.log(p + 1e-10)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # Check prediction
            pred = logits.argmax(dim=1).item()
            if pred == (1 if label else 0):
                correct += 1
        
        accuracy = correct / len(training_data)
        
        if (epoch + 1) % 20 == 0 or accuracy == 1.0:
            print(f"      Epoch {epoch+1}: Accuracy={accuracy:.4f}, Loss={total_loss/len(training_data):.4f}")
        
        if accuracy == 1.0:
            print(f"      Achieved 100% on training data at epoch {epoch+1}")
            break
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
    
    return accuracy


def ensure_model_accuracy(model_path: Path, dfa_path: Path, device: str = 'cuda') -> Dict:
    """
    Ensure a model achieves 100% accuracy.
    """
    # Load model
    model, metadata = load_trained_model(model_path, device)
    
    # Load ground truth DFA
    dfa = load_ground_truth_dfa(dfa_path)
    alphabet = metadata['alphabet']
    
    dataset_id = model_path.parent.name
    arch = model_path.name
    
    print(f"\n  Processing {dataset_id}/{arch}")
    print(f"    Metadata accuracy: {metadata['accuracy']:.4f}")
    
    # Check actual accuracy
    print(f"    Checking accuracy exhaustively...")
    accuracy, errors = check_model_accuracy(model, dfa, alphabet, max_length=15, device=device)
    
    print(f"    Actual accuracy: {accuracy:.6f} ({len(errors)} errors)")
    
    if len(errors) == 0:
        print(f"    ✓ Model already perfect!")
        return {'success': True, 'initial_errors': 0, 'final_errors': 0}
    
    # Show some errors
    print(f"    First few errors:")
    for string, label in errors[:5]:
        print(f"      '{string}' should be {label}")
    
    # Special handling for empty string if it's an error
    empty_string_error = any(s == '' for s, _ in errors)
    if empty_string_error:
        print(f"    Empty string is misclassified, using special training...")
        train_with_empty_string_focus(model, dfa, alphabet, device)
    
    # Train on errors
    print(f"    Training to fix {len(errors)} errors...")
    focused_training(model, dfa, alphabet, errors, device)
    
    # Re-check accuracy
    print(f"    Re-checking accuracy...")
    final_accuracy, final_errors = check_model_accuracy(model, dfa, alphabet, max_length=15, device=device)
    
    print(f"    Final accuracy: {final_accuracy:.6f} ({len(final_errors)} errors)")
    
    if len(final_errors) == 0:
        print(f"    ✓ SUCCESS! Model now has 100% accuracy")
        
        # Save the model - overwrite the original
        torch.save(model.state_dict(), model_path / 'model.pt')
        
        # Update metadata
        metadata['accuracy'] = 1.0
        metadata['verified_perfect'] = True
        metadata['exhaustive_check_length'] = 15
        
        with open(model_path / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
            
        return {'success': True, 'initial_errors': len(errors), 'final_errors': 0}
    else:
        print(f"    ✗ Still has {len(final_errors)} errors")
        return {'success': False, 'initial_errors': len(errors), 'final_errors': len(final_errors)}


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Ensure all LSTM and GRU models achieve 100% accuracy"
    )
    
    parser.add_argument('--model-dir', type=str, default='trained_models')
    parser.add_argument('--dfa-dir', type=str, default='random_dfas')
    parser.add_argument('--dataset-id', type=str, help='Process specific dataset only')
    parser.add_argument('--architectures', type=str, nargs='+', 
                       default=['lstm', 'gru'],
                       choices=['lstm', 'gru'])
    parser.add_argument('--device', type=str, default='cuda')
    
    args = parser.parse_args()
    
    # Set seeds for reproducibility
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    model_dir = Path(args.model_dir)
    dfa_dir = Path(args.dfa_dir)
    
    print("="*80)
    print("ENSURING 100% ACCURACY FOR LSTM AND GRU MODELS")
    print("="*80)
    
    # Find datasets to process
    if args.dataset_id:
        dataset_ids = [args.dataset_id]
    else:
        dataset_ids = sorted(set(d.parent.name for d in model_dir.glob('*/*/model.pt')))
    
    results = {}
    
    for dataset_id in dataset_ids:
        dfa_path = dfa_dir / dataset_id / 'dfa.json'
        if not dfa_path.exists():
            continue
        
        print(f"\nDataset: {dataset_id}")
        
        for arch in args.architectures:
            model_path = model_dir / dataset_id / arch
            if not (model_path / 'model.pt').exists():
                continue
            
            try:
                result = ensure_model_accuracy(model_path, dfa_path, args.device)
                results[f"{dataset_id}/{arch}"] = result
            except Exception as e:
                print(f"    Error: {e}")
                import traceback
                traceback.print_exc()
                results[f"{dataset_id}/{arch}"] = {'success': False, 'error': str(e)}
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    successful = [k for k, v in results.items() if v.get('success', False)]
    failed = [k for k, v in results.items() if not v.get('success', False)]
    
    print(f"\nTotal models processed: {len(results)}")
    print(f"Successful (100% accuracy): {len(successful)}")
    print(f"Failed: {len(failed)}")
    
    if successful:
        print("\nSuccessful models:")
        for model in successful:
            result = results[model]
            if result['initial_errors'] > 0:
                print(f"  ✓ {model}: Fixed {result['initial_errors']} errors")
            else:
                print(f"  ✓ {model}: Already perfect")
    
    if failed:
        print("\nFailed models:")
        for model in failed:
            result = results[model]
            if 'error' in result:
                print(f"  ✗ {model}: {result['error']}")
            else:
                print(f"  ✗ {model}: {result['initial_errors']} → {result['final_errors']} errors")


if __name__ == "__main__":
    main()