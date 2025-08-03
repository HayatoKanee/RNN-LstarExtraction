#!/usr/bin/env python
"""
Exhaustively validate trained RNN models against ground truth DFAs.
Tests all strings up to a specified length to ensure 100% accuracy.
"""

import os
import sys
import torch
import json
import argparse
from pathlib import Path
from typing import List, Tuple, Dict
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))
from models.rnn_classifier import LSTMClassifier, GRUClassifier, VanillaRNNClassifier
from core.dfa import DFA


def generate_all_strings(max_length: int, alphabet: List[str]) -> Dict[int, List[str]]:
    """Generate all strings up to max_length grouped by length."""
    strings_by_length = defaultdict(list)
    
    # Empty string
    strings_by_length[0] = ['']
    
    # Generate strings of each length
    for length in range(1, max_length + 1):
        if length == 1:
            strings_by_length[1] = alphabet[:]
        else:
            # Generate all strings of this length
            prev_strings = strings_by_length[length - 1]
            for s in prev_strings:
                for char in alphabet:
                    strings_by_length[length].append(s + char)
    
    return strings_by_length


def load_ground_truth_dfa(dfa_path: Path) -> DFA:
    """Load ground truth DFA from JSON."""
    with open(dfa_path, 'r') as f:
        data = json.load(f)
    
    states = set(data['states'])
    alphabet = data['alphabet']
    initial_state = data['initial_state']
    accepting_states = set(data['accepting_states'])
    
    transitions = {}
    for state in states:
        transitions[state] = {}
    
    for trans in data['transitions']:
        transitions[trans['from']][trans['symbol']] = trans['to']
    
    return DFA(
        states=states,
        alphabet=alphabet,
        transitions=transitions,
        initial_state=initial_state,
        final_states=accepting_states
    )


def load_trained_model(model_dir: Path, device: str = 'cpu'):
    """Load trained RNN model."""
    with open(model_dir / 'metadata.json', 'r') as f:
        metadata = json.load(f)
    
    architecture = metadata['architecture']
    
    if architecture == 'lstm':
        model = LSTMClassifier(
            alphabet_size=metadata['alphabet_size'],
            embedding_dim=metadata['embedding_dim'],
            hidden_dim=metadata['hidden_dim'],
            num_layers=metadata['num_layers'],
            device=device
        )
    elif architecture == 'gru':
        model = GRUClassifier(
            alphabet_size=metadata['alphabet_size'],
            embedding_dim=metadata['embedding_dim'],
            hidden_dim=metadata['hidden_dim'],
            num_layers=metadata['num_layers'],
            device=device
        )
    else:  # vanilla
        model = VanillaRNNClassifier(
            alphabet_size=metadata['alphabet_size'],
            embedding_dim=metadata['embedding_dim'],
            hidden_dim=metadata['hidden_dim'],
            num_layers=metadata['num_layers'],
            device=device
        )
    
    # Load model weights
    model.load_state_dict(torch.load(model_dir / 'model.pt', map_location=device))
    model = model.to(device)
    model.eval()
    
    return model, metadata


def validate_model_exhaustive(model, dfa: DFA, alphabet: List[str], 
                            max_length: int = 20, device: str = 'cpu') -> Dict:
    """Exhaustively validate model against DFA."""
    # Generate all strings
    strings_by_length = generate_all_strings(max_length, alphabet)
    
    # Validation results
    results = {
        'total_strings': 0,
        'correct': 0,
        'errors': [],
        'errors_by_length': defaultdict(list),
        'accuracy_by_length': {}
    }
    
    # Create char to index mapping
    char_to_idx = {char: idx for idx, char in enumerate(alphabet)}
    
    # Test each length
    for length in range(max_length + 1):
        strings = strings_by_length[length]
        correct_at_length = 0
        
        for string in strings:
            # Get ground truth
            dfa_result = dfa.classify_word(string)
            
            # Get model prediction
            if len(string) == 0:
                x = torch.zeros(1, 0, dtype=torch.long, device=device)
            else:
                indices = [char_to_idx[c] for c in string]
                x = torch.tensor([indices], dtype=torch.long, device=device)
            
            with torch.no_grad():
                logits = model(x)
                pred = logits.argmax(dim=1).item()
                model_result = bool(pred)
            
            results['total_strings'] += 1
            
            if model_result == dfa_result:
                results['correct'] += 1
                correct_at_length += 1
            else:
                error_info = {
                    'string': string,
                    'dfa_result': dfa_result,
                    'model_result': model_result,
                    'length': length
                }
                results['errors'].append(error_info)
                results['errors_by_length'][length].append(string)
        
        # Calculate accuracy for this length
        results['accuracy_by_length'][length] = correct_at_length / len(strings) if strings else 1.0
        
        # Print progress
        accuracy = correct_at_length / len(strings) if strings else 1.0
        if accuracy < 1.0 or length <= 5:
            print(f"    Length {length}: {correct_at_length}/{len(strings)} ({accuracy:.4f})")
            if accuracy < 1.0 and length <= 10:
                # Show first few errors
                for err in results['errors_by_length'][length][:5]:
                    dfa_val = dfa.classify_word(err)
                    print(f"      Error: '{err}' - DFA: {dfa_val}, Model: {not dfa_val}")
    
    results['overall_accuracy'] = results['correct'] / results['total_strings']
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Exhaustively validate trained models against ground truth DFAs"
    )
    
    parser.add_argument(
        '--model-dir',
        type=str,
        default='validation/trained_models',
        help='Directory containing trained models'
    )
    
    parser.add_argument(
        '--dfa-dir',
        type=str,
        default='validation/random_dfas',
        help='Directory containing ground truth DFAs'
    )
    
    parser.add_argument(
        '--model-id',
        type=str,
        required=True,
        help='Model ID to validate (e.g., alphabet2_states5_v1)'
    )
    
    parser.add_argument(
        '--architecture',
        type=str,
        default='lstm',
        choices=['lstm', 'gru', 'vanilla'],
        help='Model architecture'
    )
    
    parser.add_argument(
        '--max-length',
        type=int,
        default=20,
        help='Maximum string length to test'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device for computation'
    )
    
    args = parser.parse_args()
    
    # Load ground truth DFA
    dfa_path = Path(args.dfa_dir) / args.model_id / 'dfa.json'
    if not dfa_path.exists():
        print(f"Error: DFA not found at {dfa_path}")
        return
    
    dfa = load_ground_truth_dfa(dfa_path)
    print(f"Loaded ground truth DFA: {len(dfa.states)} states, alphabet {dfa.alphabet}")
    
    # Load trained model
    model_path = Path(args.model_dir) / args.model_id / args.architecture
    if not model_path.exists():
        print(f"Error: Model not found at {model_path}")
        return
    
    model, metadata = load_trained_model(model_path, args.device)
    print(f"Loaded {args.architecture.upper()} model: accuracy {metadata['accuracy']:.4f}")
    
    # Exhaustive validation
    print(f"\nExhaustive validation up to length {args.max_length}:")
    print("-" * 60)
    
    results = validate_model_exhaustive(
        model, dfa, dfa.alphabet, 
        max_length=args.max_length,
        device=args.device
    )
    
    # Summary
    print("\n" + "="*60)
    print("VALIDATION SUMMARY")
    print("="*60)
    print(f"Total strings tested: {results['total_strings']}")
    print(f"Correct predictions: {results['correct']}")
    print(f"Overall accuracy: {results['overall_accuracy']:.6f}")
    print(f"Total errors: {len(results['errors'])}")
    
    if results['errors']:
        print(f"\nFirst 10 errors:")
        for err in results['errors'][:10]:
            print(f"  '{err['string']}' (len {err['length']}): "
                  f"DFA={err['dfa_result']}, Model={err['model_result']}")
    
    # Save results
    output_dir = Path('validation/exhaustive_results')
    output_dir.mkdir(exist_ok=True)
    
    output_file = output_dir / f"{args.model_id}_{args.architecture}_exhaustive.json"
    with open(output_file, 'w') as f:
        json.dump({
            'model_id': args.model_id,
            'architecture': args.architecture,
            'max_length': args.max_length,
            'results': results
        }, f, indent=2)
    
    print(f"\nResults saved to {output_file}")
    
    # Return success/failure
    return results['overall_accuracy'] == 1.0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)