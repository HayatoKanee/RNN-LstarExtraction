#!/usr/bin/env python
"""
Run Extraction Validation

This script performs DFA extraction validation by loading trained RNN models
and comparing extracted DFAs with ground truth DFAs.

Usage:
    python run_extraction_validation.py --model-dir validation/trained_models
    python run_extraction_validation.py --model-id alphabet2_states5_v1 --oracle whitebox
"""

import os
import sys
import json
import time
import torch
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))
from core.dfa import DFA
from models.rnn_classifier import LSTMClassifier, GRUClassifier, VanillaRNNClassifier
from extraction.dfa_extractor import DFAExtractor
from benchmarks.oracle_config import get_default_configs


def load_ground_truth_dfa(dfa_path: Path) -> DFA:
    """Load ground truth DFA from JSON file."""
    with open(dfa_path, 'r') as f:
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


def load_trained_model(model_dir: Path, device: str = 'cpu'):
    """Load trained RNN model with metadata."""
    # Load metadata
    with open(model_dir / 'metadata.json', 'r') as f:
        metadata = json.load(f)
    
    # Create model based on architecture
    architecture = metadata['architecture']
    alphabet_size = metadata['alphabet_size']
    hidden_dim = metadata['hidden_dim']
    num_layers = metadata['num_layers']
    embedding_dim = metadata['embedding_dim']
    
    if architecture == 'lstm':
        model = LSTMClassifier(
            alphabet_size=alphabet_size,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            device=device
        )
    elif architecture == 'gru':
        model = GRUClassifier(
            alphabet_size=alphabet_size,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            device=device
        )
    else:  # vanilla
        model = VanillaRNNClassifier(
            alphabet_size=alphabet_size,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            device=device
        )
    
    # Load state dict
    model.load_state_dict(torch.load(model_dir / 'model.pt', map_location=device))
    model = model.to(device)
    model.eval()
    
    return model, metadata


def check_dfa_equivalence(dfa1: DFA, dfa2: DFA, max_length: int = 15) -> Tuple[bool, Optional[str]]:
    """
    Check if two DFAs are equivalent by testing strings.
    
    Returns:
        (is_equivalent, counterexample)
    """
    # Check same alphabet
    if set(dfa1.alphabet) != set(dfa2.alphabet):
        return False, "Different alphabets"
    
    # Test empty string
    if dfa1.classify_word("") != dfa2.classify_word(""):
        return False, ""
    
    # Exhaustive test up to max_length
    for length in range(1, max_length + 1):
        for i in range(len(dfa1.alphabet) ** length):
            # Generate string
            string = ""
            num = i
            for _ in range(length):
                string = dfa1.alphabet[num % len(dfa1.alphabet)] + string
                num //= len(dfa1.alphabet)
            
            if dfa1.classify_word(string) != dfa2.classify_word(string):
                return False, string
    
    return True, None


def validate_extraction(model, model_metadata: Dict, ground_truth_dfa: DFA,
                       oracle_type: str, oracle_config: Dict,
                       device: str = 'cpu', time_limit: float = 60.0) -> Dict:
    """
    Validate DFA extraction for a single model and oracle.
    
    Returns:
        Validation results dictionary
    """
    result = {
        'oracle': oracle_type,
        'extraction_successful': False,
        'is_equivalent': False,
        'counterexample': None,
        'extraction_time': 0.0,
        'extracted_states': 0,
        'queries_made': 0,
        'error': None
    }
    
    try:
        # Create extractor
        extractor = DFAExtractor(
            rnn_model=model,
            alphabet=model_metadata['alphabet'],
            device=device
        )
        
        # Extract DFA
        print(f"      Extracting with {oracle_type} oracle...")
        start_time = time.time()
        
        extracted_dfa = extractor.extract(
            oracle_type=oracle_type,
            oracle_params=oracle_config,
            time_limit=time_limit,
            verbose=False
        )
        
        extraction_time = time.time() - start_time
        
        # Check equivalence
        is_equivalent, counterexample = check_dfa_equivalence(
            ground_truth_dfa, extracted_dfa
        )
        
        # Update result
        result['extraction_successful'] = True
        result['is_equivalent'] = is_equivalent
        result['counterexample'] = counterexample
        result['extraction_time'] = extraction_time
        result['extracted_states'] = len(extracted_dfa.states)
        result['queries_made'] = extractor.extraction_stats.get('total_queries', 0)
        
        status = "PASS" if is_equivalent else f"FAIL (counterexample: '{counterexample}')"
        print(f"        {status} - {len(extracted_dfa.states)} states in {extraction_time:.2f}s")
        
    except Exception as e:
        result['error'] = str(e)
        print(f"        ERROR: {str(e)}")
    
    return result


def validate_model_dataset(dataset_id: str, architecture: str,
                          model_dir: Path, dfa_dir: Path,
                          oracles: List[str], device: str = 'cpu') -> Dict:
    """Validate extraction for a specific model and dataset."""
    print(f"\n  Validating {architecture.upper()} model...")
    
    # Load model
    try:
        model, metadata = load_trained_model(model_dir, device)
        print(f"    Model accuracy: {metadata['accuracy']:.4f}")
    except Exception as e:
        print(f"    Error loading model: {e}")
        return {'error': str(e)}
    
    # Load ground truth DFA
    ground_truth_path = dfa_dir / dataset_id / 'dfa.json'
    if not ground_truth_path.exists():
        print(f"    Ground truth DFA not found at {ground_truth_path}")
        return {'error': 'Ground truth not found'}
    
    ground_truth_dfa = load_ground_truth_dfa(ground_truth_path)
    print(f"    Ground truth: {len(ground_truth_dfa.states)} states")
    
    # Get oracle configurations
    oracle_configs = get_default_configs()
    
    # Validate with each oracle
    results = {
        'architecture': architecture,
        'model_accuracy': metadata['accuracy'],
        'ground_truth_states': len(ground_truth_dfa.states),
        'oracles': {}
    }
    
    for oracle in oracles:
        if oracle not in oracle_configs:
            print(f"    Warning: Unknown oracle type '{oracle}'")
            continue
        
        oracle_config = oracle_configs[oracle]
        # Convert OracleConfig to dict for **kwargs
        config_dict = {
            'epsilon': getattr(oracle_config, 'epsilon', 0.01),
            'delta': getattr(oracle_config, 'delta', 0.01),
            'max_length': getattr(oracle_config, 'max_length', 50),
            'distribution': getattr(oracle_config, 'distribution', 'uniform'),
            'max_target_states': getattr(oracle_config, 'max_target_states', 10),
            'max_depth': getattr(oracle_config, 'max_depth', 15),
            'split_depth': getattr(oracle_config, 'split_depth', 10),
            'window_size': getattr(oracle_config, 'window_size', 1),
            'min_length': getattr(oracle_config, 'min_length', 1),
            'expected_length': getattr(oracle_config, 'expected_length', 10),
            'num_tests': getattr(oracle_config, 'num_tests', 100),
            'max_total_length': getattr(oracle_config, 'max_total_length', 5000),
            'breadth_limit': getattr(oracle_config, 'breadth_limit', 1000)
        }
        
        oracle_result = validate_extraction(
            model, metadata, ground_truth_dfa,
            oracle, config_dict,
            device=device
        )
        
        results['oracles'][oracle] = oracle_result
    
    return results


def generate_summary_report(all_results: List[Dict], output_path: Path):
    """Generate summary report of validation results."""
    # Compute statistics
    summary = {
        'total_validations': 0,
        'by_architecture': defaultdict(lambda: {
            'total': 0, 'successful': 0, 'equivalent': 0
        }),
        'by_oracle': defaultdict(lambda: {
            'total': 0, 'successful': 0, 'equivalent': 0
        }),
        'by_dfa_size': defaultdict(lambda: {
            'total': 0, 'successful': 0, 'equivalent': 0
        })
    }
    
    for result in all_results:
        if 'error' in result:
            continue
            
        for arch_result in result['results']:
            if 'error' in arch_result:
                continue
                
            arch = arch_result['architecture']
            
            for oracle, oracle_result in arch_result['oracles'].items():
                summary['total_validations'] += 1
                summary['by_architecture'][arch]['total'] += 1
                summary['by_oracle'][oracle]['total'] += 1
                summary['by_dfa_size'][result['ground_truth_states']]['total'] += 1
                
                if oracle_result['extraction_successful']:
                    summary['by_architecture'][arch]['successful'] += 1
                    summary['by_oracle'][oracle]['successful'] += 1
                    summary['by_dfa_size'][result['ground_truth_states']]['successful'] += 1
                    
                    if oracle_result['is_equivalent']:
                        summary['by_architecture'][arch]['equivalent'] += 1
                        summary['by_oracle'][oracle]['equivalent'] += 1
                        summary['by_dfa_size'][result['ground_truth_states']]['equivalent'] += 1
    
    # Write summary report
    summary_path = output_path / 'validation_summary.txt'
    with open(summary_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("DFA EXTRACTION VALIDATION SUMMARY\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"Total validations performed: {summary['total_validations']}\n\n")
        
        # By architecture
        f.write("Results by RNN Architecture:\n")
        f.write("-" * 50 + "\n")
        f.write(f"{'Architecture':<12} {'Total':<8} {'Success Rate':<15} {'Equivalence Rate':<15}\n")
        f.write("-" * 50 + "\n")
        
        for arch in ['lstm', 'gru', 'vanilla']:
            if arch in summary['by_architecture']:
                data = summary['by_architecture'][arch]
                success_rate = data['successful'] / data['total'] if data['total'] > 0 else 0
                equiv_rate = data['equivalent'] / data['total'] if data['total'] > 0 else 0
                f.write(f"{arch.upper():<12} {data['total']:<8} {success_rate:<15.1%} {equiv_rate:<15.1%}\n")
        
        # By oracle
        f.write("\n\nResults by Oracle Strategy:\n")
        f.write("-" * 50 + "\n")
        f.write(f"{'Oracle':<12} {'Total':<8} {'Success Rate':<15} {'Equivalence Rate':<15}\n")
        f.write("-" * 50 + "\n")
        
        for oracle in ['whitebox', 'pac', 'w_method', 'bfs', 'random_wp']:
            if oracle in summary['by_oracle']:
                data = summary['by_oracle'][oracle]
                success_rate = data['successful'] / data['total'] if data['total'] > 0 else 0
                equiv_rate = data['equivalent'] / data['total'] if data['total'] > 0 else 0
                f.write(f"{oracle:<12} {data['total']:<8} {success_rate:<15.1%} {equiv_rate:<15.1%}\n")
        
        # By DFA size
        f.write("\n\nResults by DFA Size:\n")
        f.write("-" * 50 + "\n")
        f.write(f"{'DFA States':<12} {'Total':<8} {'Success Rate':<15} {'Equivalence Rate':<15}\n")
        f.write("-" * 50 + "\n")
        
        for size in sorted(summary['by_dfa_size'].keys()):
            data = summary['by_dfa_size'][size]
            success_rate = data['successful'] / data['total'] if data['total'] > 0 else 0
            equiv_rate = data['equivalent'] / data['total'] if data['total'] > 0 else 0
            f.write(f"{size:<12} {data['total']:<8} {success_rate:<15.1%} {equiv_rate:<15.1%}\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("VALIDATION COMPLETE\n")
        f.write("=" * 80 + "\n")
    
    print(f"\nSummary report saved to {summary_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Run DFA extraction validation"
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
        help='Validate specific model ID only'
    )
    
    parser.add_argument(
        '--architectures',
        type=str,
        nargs='+',
        default=['lstm', 'gru', 'vanilla'],
        choices=['lstm', 'gru', 'vanilla'],
        help='RNN architectures to validate'
    )
    
    parser.add_argument(
        '--oracles',
        type=str,
        nargs='+',
        default=['whitebox', 'pac', 'w_method', 'bfs', 'random_wp'],
        help='Oracle strategies to test'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='validation/results',
        help='Output directory for results'
    )
    
    parser.add_argument(
        '--time-limit',
        type=float,
        default=60.0,
        help='Time limit per extraction (seconds)'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device for extraction'
    )
    
    args = parser.parse_args()
    
    # Find models to validate
    model_path = Path(args.model_dir)
    dfa_path = Path(args.dfa_dir)
    
    if args.model_id:
        # Validate specific model
        model_ids = [args.model_id]
    else:
        # Find all model IDs - look for directories that contain architecture subdirectories
        model_ids = []
        for d in model_path.iterdir():
            if d.is_dir():
                # Check if this directory contains architecture subdirectories
                has_models = any((d / arch / 'model.pt').exists() for arch in ['lstm', 'gru', 'vanilla'])
                if has_models:
                    model_ids.append(d.name)
        model_ids = sorted(model_ids)
    
    if not model_ids:
        print(f"No models found in {model_path}")
        return
    
    # Create output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"DFA Extraction Validation")
    print(f"Model directory: {model_path}")
    print(f"DFA directory: {dfa_path}")
    print(f"Output directory: {output_path}")
    print(f"Architectures: {args.architectures}")
    print(f"Oracles: {args.oracles}")
    print(f"Time limit: {args.time_limit}s per extraction")
    print("-" * 60)
    
    # Validate each dataset
    all_results = []
    
    for dataset_id in model_ids:
        print(f"\nValidating dataset: {dataset_id}")
        
        # Load ground truth info
        ground_truth_path = dfa_path / dataset_id / 'dfa.json'
        if not ground_truth_path.exists():
            print(f"  Warning: Ground truth DFA not found at {ground_truth_path}")
            continue
            
        with open(ground_truth_path, 'r') as f:
            ground_truth_info = json.load(f)
        print(f"  Ground truth: {ground_truth_info['num_states']} states, "
              f"alphabet size {ground_truth_info['alphabet_size']}")
        
        dataset_results = {
            'dataset_id': dataset_id,
            'ground_truth_states': ground_truth_info.get('num_states', 0),
            'results': []
        }
        
        # Validate each architecture
        for arch in args.architectures:
            arch_model_dir = model_path / dataset_id / arch
            if not arch_model_dir.exists():
                print(f"  {arch.upper()} model not found")
                continue
            
            arch_results = validate_model_dataset(
                dataset_id, arch, arch_model_dir, dfa_path,
                args.oracles, device=args.device
            )
            
            dataset_results['results'].append(arch_results)
        
        all_results.append(dataset_results)
        
        # Save intermediate results
        intermediate_path = output_path / 'validation_results.json'
        with open(intermediate_path, 'w') as f:
            json.dump(all_results, f, indent=2)
    
    # Generate summary report
    generate_summary_report(all_results, output_path)
    
    # Save final results
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    final_path = output_path / f'validation_results_{timestamp}.json'
    with open(final_path, 'w') as f:
        json.dump({
            'timestamp': timestamp,
            'config': {
                'architectures': args.architectures,
                'oracles': args.oracles,
                'time_limit': args.time_limit
            },
            'results': all_results
        }, f, indent=2)
    
    print(f"\nResults saved to {final_path}")


if __name__ == "__main__":
    main()