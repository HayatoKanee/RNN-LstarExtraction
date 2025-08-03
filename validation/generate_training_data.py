#!/usr/bin/env python
"""
Generate Training Datasets from DFAs

This script reads DFAs and generates training datasets for RNN training.
Supports both exhaustive generation for small strings and random sampling.

Usage:
    python generate_training_data.py --dfa-dir validation/random_dfas
    python generate_training_data.py --dfa-id alphabet2_states5_v1 --max-length 30
"""

import os
import sys
import json
import random
import argparse
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))
from core.dfa import DFA


def load_dfa_from_json(json_path: Path) -> DFA:
    """Load DFA from JSON file."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Reconstruct DFA
    states = set(data['states'])
    alphabet = data['alphabet']
    initial_state = data['initial_state']
    accepting_states = set(data['accepting_states'])
    
    # Reconstruct transitions
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


def generate_all_strings(alphabet: List[str], max_length: int) -> List[str]:
    """Generate all strings up to max_length."""
    strings = ['']  # Empty string
    
    for length in range(1, max_length + 1):
        # Generate all strings of this length
        if len(alphabet) ** length > 100000:  # Safety limit
            break
            
        current_strings = []
        if length == 1:
            current_strings = alphabet[:]
        else:
            # Generate by extending previous length strings
            for s in strings:
                if len(s) == length - 1:
                    for symbol in alphabet:
                        current_strings.append(s + symbol)
        
        strings.extend(current_strings)
    
    return strings


def generate_training_dataset(dfa: DFA, 
                            exhaustive_length: int = 8,
                            max_length: int = 30,
                            samples_per_length: int = 100,
                            balance_classes: bool = True) -> Dict[str, bool]:
    """
    Generate training dataset from DFA.
    
    Args:
        dfa: Source DFA
        exhaustive_length: Generate all strings up to this length
        max_length: Maximum string length for random sampling
        samples_per_length: Number of random samples per length
        balance_classes: Whether to balance positive/negative examples
        
    Returns:
        Dictionary mapping strings to labels
    """
    dataset = {}
    
    # Part 1: Exhaustive generation for short strings
    print(f"  Generating exhaustive strings up to length {exhaustive_length}...")
    
    all_short_strings = generate_all_strings(dfa.alphabet, exhaustive_length)
    for string in all_short_strings:
        dataset[string] = dfa.classify_word(string)
    
    short_positive = sum(1 for label in dataset.values() if label)
    print(f"    Generated {len(dataset)} strings ({short_positive} positive)")
    
    # Part 2: Random sampling for longer strings
    if max_length > exhaustive_length:
        print(f"  Generating random strings (length {exhaustive_length+1} to {max_length})...")
        
        for length in range(exhaustive_length + 1, max_length + 1):
            positive_samples = []
            negative_samples = []
            
            # Generate more samples than needed to ensure we get both classes
            attempts = samples_per_length * 5
            for _ in range(attempts):
                string = ''.join(random.choice(dfa.alphabet) for _ in range(length))
                
                # Skip if already in dataset
                if string in dataset:
                    continue
                
                label = dfa.classify_word(string)
                if label:
                    positive_samples.append(string)
                else:
                    negative_samples.append(string)
                
                # Stop if we have enough of both classes
                if len(positive_samples) >= samples_per_length and len(negative_samples) >= samples_per_length:
                    break
            
            # Add samples to dataset
            if balance_classes:
                # Balance classes
                min_samples = min(len(positive_samples), len(negative_samples), samples_per_length // 2)
                for string in positive_samples[:min_samples]:
                    dataset[string] = True
                for string in negative_samples[:min_samples]:
                    dataset[string] = False
            else:
                # Add all samples up to limit
                for string in positive_samples[:samples_per_length]:
                    dataset[string] = True
                for string in negative_samples[:samples_per_length]:
                    dataset[string] = False
    
    # Final statistics
    total_positive = sum(1 for label in dataset.values() if label)
    total_negative = len(dataset) - total_positive
    
    print(f"  Total dataset: {len(dataset)} strings")
    print(f"    Positive: {total_positive} ({total_positive/len(dataset)*100:.1f}%)")
    print(f"    Negative: {total_negative} ({total_negative/len(dataset)*100:.1f}%)")
    
    return dataset


def save_dataset(dataset: Dict[str, bool], output_path: Path, dataset_id: str):
    """Save dataset in multiple formats."""
    dataset_dir = output_path / dataset_id
    dataset_dir.mkdir(parents=True, exist_ok=True)
    
    # Save as JSON (compact format)
    json_path = dataset_dir / 'dataset.json'
    with open(json_path, 'w') as f:
        json.dump({
            'id': dataset_id,
            'size': len(dataset),
            'data': dataset
        }, f)
    
    # Save as CSV for easy inspection
    csv_path = dataset_dir / 'dataset.csv'
    with open(csv_path, 'w') as f:
        f.write("string,label\n")
        for string, label in sorted(dataset.items()):
            # Escape special characters
            escaped_string = string.replace('"', '""')
            f.write(f'"{escaped_string}",{int(label)}\n')
    
    # Save statistics
    stats = {
        'total_examples': len(dataset),
        'positive_examples': sum(1 for l in dataset.values() if l),
        'negative_examples': sum(1 for l in dataset.values() if not l),
        'max_length': max(len(s) for s in dataset.keys()) if dataset else 0,
        'unique_lengths': len(set(len(s) for s in dataset.keys()))
    }
    
    stats_path = dataset_dir / 'stats.json'
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="Generate training datasets from DFAs"
    )
    
    parser.add_argument(
        '--dfa-dir',
        type=str,
        default='validation/random_dfas',
        help='Directory containing generated DFAs'
    )
    
    parser.add_argument(
        '--dfa-id',
        type=str,
        help='Generate dataset for specific DFA ID only'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='validation/datasets',
        help='Output directory for datasets'
    )
    
    parser.add_argument(
        '--exhaustive-length',
        type=int,
        default=8,
        help='Generate all strings up to this length (default: 8)'
    )
    
    parser.add_argument(
        '--max-length',
        type=int,
        default=30,
        help='Maximum string length for random sampling (default: 30)'
    )
    
    parser.add_argument(
        '--samples-per-length',
        type=int,
        default=100,
        help='Random samples per length (default: 100)'
    )
    
    parser.add_argument(
        '--balance-classes',
        action='store_true',
        default=True,
        help='Balance positive/negative examples'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed (default: 42)'
    )
    
    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.seed)
    
    # Create output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find DFAs to process
    dfa_path = Path(args.dfa_dir)
    
    if args.dfa_id:
        # Process specific DFA
        dfa_dirs = [dfa_path / args.dfa_id]
    else:
        # Process all DFAs
        dfa_dirs = [d for d in dfa_path.iterdir() if d.is_dir() and (d / 'dfa.json').exists()]
    
    if not dfa_dirs:
        print(f"No DFAs found in {dfa_path}")
        return
    
    print(f"Generating Training Datasets")
    print(f"DFA directory: {dfa_path}")
    print(f"Output directory: {output_path}")
    print(f"Exhaustive length: {args.exhaustive_length}")
    print(f"Max length: {args.max_length}")
    print(f"Samples per length: {args.samples_per_length}")
    print("-" * 60)
    
    # Process each DFA
    for dfa_dir in sorted(dfa_dirs):
        dfa_id = dfa_dir.name
        print(f"\nProcessing {dfa_id}...")
        
        # Load DFA
        dfa = load_dfa_from_json(dfa_dir / 'dfa.json')
        print(f"  Loaded DFA: {len(dfa.states)} states, alphabet {dfa.alphabet}")
        
        # Generate dataset
        dataset = generate_training_dataset(
            dfa,
            exhaustive_length=args.exhaustive_length,
            max_length=args.max_length,
            samples_per_length=args.samples_per_length,
            balance_classes=args.balance_classes
        )
        
        # Save dataset
        save_dataset(dataset, output_path, dfa_id)
        print(f"  Saved dataset to {output_path / dfa_id}/")
    
    print(f"\nGenerated {len(dfa_dirs)} datasets")


if __name__ == "__main__":
    main()