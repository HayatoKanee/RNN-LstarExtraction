"""
Benchmark runner for comparing different equivalence oracle approaches.

This module orchestrates the benchmarking process, running L* with different
equivalence oracles and collecting comprehensive metrics.
"""

import os
import time
import torch
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import json
import traceback
import random

from extraction.dfa_extractor import DFAExtractor
from grammars.tomita import TOMITA_GRAMMARS
from benchmarks.metrics import MetricsCollector, BenchmarkResults
from benchmarks.oracle_config import OracleConfig, get_default_configs


class BenchmarkRunner:
    """Orchestrates benchmark execution across different configurations."""
    
    def __init__(self, output_dir: str = "benchmark_results",
                 exhaustive_test_length: int = 15,
                 sample_size: int = 50000,
                 sample_min_length: int = 16,
                 sample_max_length: int = 100,
                 long_test_lengths: List[int] = None,
                 long_test_samples: int = 100,
                 num_workers: Optional[int] = None):
        """
        Initialize benchmark runner with configurable test parameters.
        
        Args:
            output_dir: Directory to save results
            exhaustive_test_length: Maximum length for exhaustive testing (default: 15)
            sample_size: Number of random samples to generate (default: 50,000)
            sample_min_length: Minimum length for random samples (default: 16)
            sample_max_length: Maximum length for random samples (default: 100)
            long_test_lengths: Lengths for long sequence testing (default: [500, 1000])
            long_test_samples: Number of samples per long test length (default: 100)
            num_workers: Number of parallel workers (default: CPU count, 1 for sequential)
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.dfa_dir = self.output_dir / "extracted_dfas"
        self.dfa_dir.mkdir(exist_ok=True)
        self.results = BenchmarkResults()
        
        # Test set configuration
        self.exhaustive_test_length = exhaustive_test_length
        self.sample_size = sample_size
        self.sample_min_length = sample_min_length
        self.sample_max_length = sample_max_length
        self.long_test_lengths = long_test_lengths or [500, 1000]
        self.long_test_samples = long_test_samples
        
        # Set number of workers
        import multiprocessing
        self.num_workers = num_workers or multiprocessing.cpu_count()
        
    def run_default_benchmark(self):
        """Run benchmark with default settings."""
        return self.run_benchmarks()
    
    def run_benchmarks(self, model_dir: Optional[str] = None):
        """Run benchmarks with default configurations.
        
        Args:
            model_dir: Directory containing trained models. If None, uses default location.
        """
        # Get default oracle configurations (all 5 types)
        oracle_configs = get_default_configs()
        
        # Run on all Tomita grammars + balanced brackets
        grammars = [f"tomita{i}" for i in range(1, 8)] + ["balanced_brackets"]
        
        # Use default model directory if not specified
        if model_dir is None:
            # Get package root directory
            package_root = Path(__file__).parent.parent
            model_dir = package_root / "trained_models"
        
        # Run benchmark with default test parameters
        return self.run_benchmark(
            grammars=grammars,
            oracle_configs=oracle_configs,
            model_dir=str(model_dir),
            num_runs=3,
            time_limit=60.0
        )
        
    def _parallel_extraction_task(self, args):
        """Execute a single extraction task for parallel processing."""
        grammar_name, oracle_name, oracle_config, run_num, model_dir, time_limit, seed, test_set = args
        
        # Set random seed for reproducibility
        import random
        import numpy as np
        import torch
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        # Load model
        model_path = self._find_model(model_dir, grammar_name)
        if not model_path:
            # Return error metrics
            from benchmarks.metrics import MetricsCollector
            metrics = MetricsCollector()
            metrics.metrics.extraction_successful = False
            metrics.metrics.error_message = f"Model not found for {grammar_name}"
            return (grammar_name, oracle_name, run_num, metrics.get_metrics())
        
        model, alphabet = self._load_model(model_path)
        
        # Get true grammar
        if grammar_name.startswith("tomita"):
            grammar_id = int(grammar_name.replace("tomita", ""))
            from grammars.tomita import TOMITA_GRAMMARS
            true_grammar, description = TOMITA_GRAMMARS[grammar_id]
        elif grammar_name == "balanced_brackets":
            from grammars.balanced_brackets import is_balanced_brackets
            true_grammar = is_balanced_brackets
        else:
            # Return error metrics
            from benchmarks.metrics import MetricsCollector
            metrics = MetricsCollector()
            metrics.metrics.extraction_successful = False
            metrics.metrics.error_message = f"Unknown grammar: {grammar_name}"
            return (grammar_name, oracle_name, run_num, metrics.get_metrics())
        
        # Create metrics collector
        from benchmarks.metrics import MetricsCollector
        metrics_collector = MetricsCollector()
        
        # Run extraction
        try:
            extraction_metrics, dfa = self._run_single_extraction(
                model=model,
                alphabet=alphabet,
                oracle_config=oracle_config,
                time_limit=time_limit,
                true_grammar=true_grammar,
                metrics_collector=metrics_collector,
                grammar_name=grammar_name,
                oracle_name=oracle_name,
                run_num=run_num,
                test_set=test_set
            )
            return (grammar_name, oracle_name, run_num, extraction_metrics)
        except Exception as e:
            import traceback
            metrics_collector.metrics.extraction_successful = False
            metrics_collector.metrics.error_message = str(e) + "\n" + traceback.format_exc()
            return (grammar_name, oracle_name, run_num, metrics_collector.get_metrics())
    
    def run_benchmark(self,
                     grammars: List[str],
                     oracle_configs: Dict[str, OracleConfig],
                     model_dir: str = "trained_models",
                     num_runs: int = 3,
                     time_limit: float = 300.0) -> BenchmarkResults:
        """
        Run comprehensive benchmark.
        
        Args:
            grammars: List of grammar names to test
            oracle_configs: Dictionary of oracle configurations
            model_dir: Directory containing trained models
            num_runs: Number of runs per configuration
            time_limit: Time limit per extraction
            
        Returns:
            BenchmarkResults object with all metrics
        """
        print("=" * 80)
        print(f"Starting RNN Extraction Benchmark")
        print(f"Workers: {self.num_workers} {'(parallel)' if self.num_workers > 1 else '(sequential)'}")
        print(f"Grammars: {grammars}")
        print(f"Oracles: {list(oracle_configs.keys())}")
        print(f"Runs per config: {num_runs}")
        print(f"Time limit: {time_limit}s")
        print(f"\nTest Set Configuration:")
        print(f"  Exhaustive testing: up to length {self.exhaustive_test_length}")
        print(f"  Random samples: {self.sample_size:,} strings (length {self.sample_min_length}-{self.sample_max_length})")
        print(f"  Long sequences: {self.long_test_samples} samples at lengths {self.long_test_lengths}")
        print("=" * 80)
        
        # Calculate total experiments accounting for deterministic oracles
        deterministic_oracles = {'whitebox', 'w_method', 'bfs'}
        
        # Pre-generate test sets for each grammar to save memory
        print("\nPre-generating test sets for all grammars...")
        grammar_test_sets = {}
        for grammar_name in grammars:
            # Determine alphabet based on grammar
            if grammar_name == 'balanced_brackets':
                alphabet = ['(', ')']
            else:
                alphabet = ['0', '1']
            
            print(f"  Generating test set for {grammar_name}...")
            test_set = self._generate_test_set(alphabet, grammar_name)
            grammar_test_sets[grammar_name] = test_set
            print(f"    Generated {len(test_set):,} test strings")
        
        # Build list of all tasks
        tasks = []
        for grammar_name in grammars:
            test_set = grammar_test_sets[grammar_name]
            for oracle_name, oracle_config in oracle_configs.items():
                # Determine number of runs
                runs_for_oracle = 1 if oracle_name in deterministic_oracles else num_runs
                for run in range(runs_for_oracle):
                    # Generate unique seed for each task
                    seed = 42 + hash(grammar_name) % 1000 + hash(oracle_name) % 100 + run * 10000
                    tasks.append((grammar_name, oracle_name, oracle_config, run, model_dir, time_limit, seed, test_set))
        
        total_experiments = len(tasks)
        print(f"\nTotal experiments: {total_experiments}")
        
        # Execute tasks using ProcessPoolExecutor
        from concurrent.futures import ProcessPoolExecutor, as_completed
        
        completed = 0
        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            # Submit all tasks
            future_to_task = {
                executor.submit(self._parallel_extraction_task, task): task 
                for task in tasks
            }
            
            # Process results as they complete
            for future in as_completed(future_to_task):
                completed += 1
                task = future_to_task[future]
                grammar_name, oracle_name, _, run_num, _, _, _, _ = task
                
                try:
                    grammar, oracle, run, metrics = future.result()
                    
                    # Add to results
                    self.results.add_result(grammar, oracle, metrics)
                    
                    # Print progress
                    if metrics.extraction_successful:
                        print(f"[{completed}/{total_experiments}] ✅ {grammar}/{oracle}/run_{run+1} "
                              f"({metrics.num_states} states, {metrics.total_time:.1f}s)")
                    else:
                        print(f"[{completed}/{total_experiments}] ❌ {grammar}/{oracle}/run_{run+1} "
                              f"(FAILED)")
                    
                except Exception as e:
                    print(f"[{completed}/{total_experiments}] ❌ {grammar_name}/{oracle_name}/run_{run_num+1} "
                          f"(ERROR: {str(e)})")
        
        # Save final results
        self._save_final_results()
        
        return self.results
    
    def _run_single_extraction(self,
                              model: torch.nn.Module,
                              alphabet: List[str],
                              oracle_config: OracleConfig,
                              time_limit: float,
                              true_grammar: callable,
                              metrics_collector: MetricsCollector,
                              grammar_name: str,
                              oracle_name: str,
                              run_num: int,
                              test_set: Optional[List[str]] = None) -> Tuple[Any, Any]:
        """Run a single extraction experiment."""
        metrics_collector.start_extraction()
        
        # Use provided test set or generate one
        if test_set is None:
            print("    Generating test set...")
            test_set = self._generate_test_set(alphabet, grammar_name)
        else:
            print(f"    Using cached test set ({len(test_set):,} strings)")
        
        # Randomly select validation subset for tracking learning progress
        print("    Selecting validation subset for learning progress tracking...")
        import random
        random.seed(42 + run_num)  # Different seed for each run
        validation_size = min(1000, len(test_set) // 10)  # 10% of test set or 1000, whichever is smaller
        validation_strings = random.sample(test_set, validation_size)
        validation_set = [(s, true_grammar(s)) for s in validation_strings]
        print(f"      Selected {len(validation_set)} validation examples")
        
        # Create DFA extractor
        # Use CUDA if available for faster inference
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if device == 'cuda':
            model = model.cuda()
            
        extractor = DFAExtractor(
            rnn_model=model,
            alphabet=alphabet,
            device=device,
            validation_set=validation_set
        )
        
        # Prepare oracle parameters
        oracle_params = oracle_config.to_dict()
        # Remove oracle_type from params as it's passed separately
        oracle_params = {k: v for k, v in oracle_params.items() if k != 'oracle_type'}
        
        # Check if we should use bounded L*
        use_bounded_lstar = oracle_params.pop('use_bounded_lstar', False)
        max_query_length = oracle_params.pop('max_query_length', 20)
        
        # Run extraction with specified oracle
        # Determine whether to use starting examples
        if hasattr(oracle_config, 'use_starting_examples_override') and oracle_config.use_starting_examples_override is not None:
            use_starting_examples = oracle_config.use_starting_examples_override
        else:
            # Default: Use starting examples for whitebox/sliding_window oracle OR when using bounded L*
            # (Bounded L* needs starting examples to avoid state explosion)
            use_starting_examples = (oracle_config.oracle_type.value in ['whitebox', 'sliding_window'] or use_bounded_lstar)
        
        try:
            dfa = extractor.extract(
                oracle_type=oracle_config.oracle_type.value,
                oracle_params=oracle_params,
                time_limit=time_limit,
                verbose=False,  # Suppress output during benchmarking
                use_starting_examples=use_starting_examples,
                use_bounded_lstar=use_bounded_lstar,
                max_query_length=max_query_length
            )
            metrics_collector.end_extraction(successful=True)
        except Exception as e:
            metrics_collector.end_extraction(successful=False, failure_reason=str(e))
            raise
        
        # Get extraction statistics
        extraction_stats = extractor.get_statistics()
        
        # Determine the actual final DFA (might be different from returned dfa if timed out)
        final_dfa = dfa
        if hasattr(dfa, 'learning_history') and dfa.learning_history:
            last_hypothesis = dfa.learning_history[-1]
            if 'dfa_object' in last_hypothesis:
                last_dfa = last_hypothesis['dfa_object']
                if len(dfa.states) < len(last_dfa.states) * 0.5:
                    final_dfa = last_dfa
        
        # Record DFA properties
        metrics_collector.record_dfa_properties(
            num_states=len(final_dfa.states)
        )
        
        # Test set already generated above
        
        # Evaluate accuracy against both RNN and grammar
        print("    Computing accuracies...")
        dfa_rnn_acc, dfa_grammar_acc, dfa_rnn_metrics, dfa_grammar_metrics = self._evaluate_accuracy(
            final_dfa, model, true_grammar, test_set, grammar_name
        )
        
        # Separate evaluation on long sequences for additional insights
        long_seq_results = self._evaluate_long_sequences_detailed(
            final_dfa, model, true_grammar, alphabet, grammar_name
        )
        
        # Record primary accuracy metrics
        metrics_collector.record_accuracy(dfa_rnn_acc, dfa_grammar_acc)
        
        # Record the new accuracy metrics explicitly
        metrics_collector.record_oracle_specific('dfa_rnn_accuracy', dfa_rnn_acc)
        metrics_collector.record_oracle_specific('dfa_grammar_accuracy', dfa_grammar_acc)
        
        # Record detailed metrics for DFA vs RNN
        metrics_collector.record_oracle_specific('dfa_rnn_tp', dfa_rnn_metrics['tp'])
        metrics_collector.record_oracle_specific('dfa_rnn_fp', dfa_rnn_metrics['fp'])
        metrics_collector.record_oracle_specific('dfa_rnn_tn', dfa_rnn_metrics['tn'])
        metrics_collector.record_oracle_specific('dfa_rnn_fn', dfa_rnn_metrics['fn'])
        metrics_collector.record_oracle_specific('dfa_rnn_precision', dfa_rnn_metrics['precision'])
        metrics_collector.record_oracle_specific('dfa_rnn_recall', dfa_rnn_metrics['recall'])
        metrics_collector.record_oracle_specific('dfa_rnn_f1', dfa_rnn_metrics['f1'])
        metrics_collector.record_oracle_specific('dfa_rnn_balanced_accuracy', dfa_rnn_metrics['balanced_accuracy'])
        metrics_collector.record_oracle_specific('dfa_rnn_mcc', dfa_rnn_metrics['mcc'])
        
        # Record detailed metrics for DFA vs Grammar
        metrics_collector.record_oracle_specific('dfa_grammar_tp', dfa_grammar_metrics['tp'])
        metrics_collector.record_oracle_specific('dfa_grammar_fp', dfa_grammar_metrics['fp'])
        metrics_collector.record_oracle_specific('dfa_grammar_tn', dfa_grammar_metrics['tn'])
        metrics_collector.record_oracle_specific('dfa_grammar_fn', dfa_grammar_metrics['fn'])
        metrics_collector.record_oracle_specific('dfa_grammar_precision', dfa_grammar_metrics['precision'])
        metrics_collector.record_oracle_specific('dfa_grammar_recall', dfa_grammar_metrics['recall'])
        metrics_collector.record_oracle_specific('dfa_grammar_f1', dfa_grammar_metrics['f1'])
        metrics_collector.record_oracle_specific('dfa_grammar_balanced_accuracy', dfa_grammar_metrics['balanced_accuracy'])
        metrics_collector.record_oracle_specific('dfa_grammar_mcc', dfa_grammar_metrics['mcc'])
        
        print(f"    DFA-RNN Accuracy: {dfa_rnn_acc:.4f}")
        print(f"    DFA-Grammar Accuracy: {dfa_grammar_acc:.4f}")
        
        # Extract relevant statistics from the extraction
        if 'teacher_stats' in extraction_stats:
            teacher_stats = extraction_stats['teacher_stats']
            
            # Set the top-level membership_queries from teacher stats
            membership_queries = teacher_stats.get('membership_queries', 0)
            metrics_collector.metrics.membership_queries = membership_queries
            metrics_collector.record_oracle_specific('membership_queries', membership_queries)
            
            # Add oracle-specific statistics if available
            if 'oracle_stats' in teacher_stats:
                oracle_stats = teacher_stats['oracle_stats']
                for key, value in oracle_stats.items():
                    if key not in ['type', 'total_queries', 'counterexamples_found', 'total_time']:
                        metrics_collector.record_oracle_specific(key, value)
        
        # Record L* algorithm statistics
        if 'lstar_stats' in extraction_stats:
            lstar_stats = extraction_stats['lstar_stats']
            iterations = lstar_stats.get('iterations', 0)
            metrics_collector.record_oracle_specific('iterations', iterations)
            metrics_collector.record_oracle_specific('counterexamples', lstar_stats.get('counterexamples', 0))
            
            # Set equivalence_queries to the number of iterations (hypotheses tested)
            metrics_collector.metrics.equivalence_queries = iterations
            
            # Set counterexamples_found to the actual number of counterexamples
            metrics_collector.metrics.counterexamples_found = lstar_stats.get('counterexamples', 0)
        
        # Save the extracted DFA
        self._save_dfa(dfa, grammar_name, oracle_name, run_num)
        
        return metrics_collector.get_metrics(), final_dfa
    
    def _find_model(self, model_dir: str, grammar_name: str) -> Optional[Path]:
        """Find trained model for a grammar."""
        model_dir_path = Path(model_dir)
        
        # Look for model files
        patterns = [
            f"{grammar_name}_lstm_h50.pt",
            f"{grammar_name}_lstm.pt"
        ]
        
        for pattern in patterns:
            model_path = model_dir_path / pattern
            if model_path.exists():
                return model_path
                
        return None
    
    def _load_model(self, model_path: Path) -> Tuple[torch.nn.Module, List[str]]:
        """Load a trained RNN model and its alphabet."""
        checkpoint = torch.load(model_path, map_location='cpu')
        
        # Import the model class
        from models.rnn_classifier import LSTMClassifier
        
        # Get model configuration
        if 'model_config' in checkpoint:
            config = checkpoint['model_config']
            model = LSTMClassifier(
                alphabet_size=config.get('alphabet_size', 2),
                embedding_dim=config.get('embedding_dim', 3),
                hidden_dim=config.get('hidden_dim', 50),
                num_layers=config.get('num_layers', 2),
                device='cpu'
            )
        else:
            # Fallback to default configuration
            hidden_size = checkpoint.get('hidden_size', 50)
            model = LSTMClassifier(
                alphabet_size=2,
                embedding_dim=3,
                hidden_dim=hidden_size,
                num_layers=2,
                device='cpu'
            )
        
        # Load state dict
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        # Get alphabet from checkpoint, or determine based on grammar
        if 'alphabet' in checkpoint:
            alphabet = checkpoint['alphabet']
        else:
            # Determine alphabet based on grammar ID
            grammar_id = checkpoint.get('grammar_id', '')
            if grammar_id == 'balanced_brackets':
                alphabet = ['(', ')']
            else:
                # Default to binary for Tomita grammars
                alphabet = ['0', '1']
        
        return model, alphabet
    
    def _generate_test_set(self, 
                          alphabet: List[str],
                          grammar_name: str) -> List[str]:
        """Generate test set with configurable parameters.
        
        Returns:
            List of test strings (without labels)
        """
        test_set = []
        
        # Part 1: Exhaustive testing up to configured length
        print(f"      Generating exhaustive test set up to length {self.exhaustive_test_length}...")
        if alphabet == ['0', '1']:
            # Binary alphabet - exhaustive enumeration
            for length in range(self.exhaustive_test_length + 1):  # 0 to exhaustive_test_length
                if length == 0:
                    test_set.append("")
                else:
                    for i in range(2**length):
                        string = bin(i)[2:].zfill(length)
                        test_set.append(string)
        elif alphabet == ['(', ')']:
            # Parentheses - systematic enumeration
            for length in range(self.exhaustive_test_length + 1):
                if length == 0:
                    test_set.append("")
                else:
                    # Generate all possible parentheses strings
                    for i in range(2**length):
                        chars = []
                        for j in range(length):
                            chars.append('(' if (i >> j) & 1 == 0 else ')')
                        test_set.append(''.join(chars))
        
        exhaustive_count = len(test_set)
        expected_exhaustive = self.calculate_exhaustive_test_size(len(alphabet))
        print(f"      Generated {exhaustive_count:,} strings (exhaustive, expected: {expected_exhaustive:,})")
        
        # Part 2: Random sampling for configured length range
        print(f"      Generating {self.sample_size:,} random strings (length {self.sample_min_length}-{self.sample_max_length})...")
        import random
        random.seed(42)  # For reproducibility
        
        for _ in range(self.sample_size):
            length = random.randint(self.sample_min_length, self.sample_max_length)
            string = ''.join(random.choice(alphabet) for _ in range(length))
            test_set.append(string)
        
        # Part 3: Long sequence samples
        if self.long_test_lengths:
            lengths_str = ', '.join(str(l) for l in self.long_test_lengths)
            print(f"      Adding {self.long_test_samples} samples at lengths {lengths_str}...")
            for length in self.long_test_lengths:
                for _ in range(self.long_test_samples):
                    string = ''.join(random.choice(alphabet) for _ in range(length))
                    test_set.append(string)
        
        total_count = len(test_set)
        print(f"      Total test set size: {total_count:,} strings")
        
        return test_set
    
    def _evaluate_accuracy(self,
                          dfa,
                          rnn_model,
                          true_grammar: callable,
                          test_set: List[str],
                          grammar_name: str) -> Tuple[float, float, dict, dict]:
        """Evaluate DFA accuracy against both RNN and true grammar.
        
        Args:
            dfa: Extracted DFA
            rnn_model: Original RNN model
            true_grammar: True grammar function
            test_set: List of test strings
            grammar_name: Name of the grammar (for alphabet detection)
            
        Returns:
            Tuple of (DFA-RNN accuracy, DFA-Grammar accuracy, DFA-RNN metrics dict, DFA-Grammar metrics dict)
        """
        # Get all classifications in batch
        # 1. DFA classifications
        dfa_labels = [dfa.accepts(list(string)) for string in test_set]
        
        # 2. RNN classifications (batch processing)
        rnn_labels = self._classify_strings_batch(rnn_model, test_set, grammar_name)
        
        # 3. True grammar classifications
        true_labels = [true_grammar(string) for string in test_set]
        
        # Initialize counters for DFA vs RNN
        dfa_rnn_tp = dfa_rnn_fp = dfa_rnn_tn = dfa_rnn_fn = 0
        # Initialize counters for DFA vs Grammar
        dfa_gram_tp = dfa_gram_fp = dfa_gram_tn = dfa_gram_fn = 0
        
        # Count metrics
        for dfa_label, rnn_label, true_label in zip(dfa_labels, rnn_labels, true_labels):
            # Count for DFA vs RNN
            if dfa_label and rnn_label:
                dfa_rnn_tp += 1
            elif dfa_label and not rnn_label:
                dfa_rnn_fp += 1
            elif not dfa_label and rnn_label:
                dfa_rnn_fn += 1
            else:
                dfa_rnn_tn += 1
                
            # Count for DFA vs Grammar
            if dfa_label and true_label:
                dfa_gram_tp += 1
            elif dfa_label and not true_label:
                dfa_gram_fp += 1
            elif not dfa_label and true_label:
                dfa_gram_fn += 1
            else:
                dfa_gram_tn += 1
        
        # Calculate metrics for DFA vs RNN
        total = len(test_set)
        dfa_rnn_accuracy = (dfa_rnn_tp + dfa_rnn_tn) / total if total > 0 else 0.0
        dfa_rnn_metrics = self._calculate_metrics(dfa_rnn_tp, dfa_rnn_fp, dfa_rnn_tn, dfa_rnn_fn)
        
        # Calculate metrics for DFA vs Grammar  
        dfa_grammar_accuracy = (dfa_gram_tp + dfa_gram_tn) / total if total > 0 else 0.0
        dfa_grammar_metrics = self._calculate_metrics(dfa_gram_tp, dfa_gram_fp, dfa_gram_tn, dfa_gram_fn)
        
        return dfa_rnn_accuracy, dfa_grammar_accuracy, dfa_rnn_metrics, dfa_grammar_metrics
    
    def _calculate_metrics(self, tp: int, fp: int, tn: int, fn: int) -> dict:
        """Calculate various metrics from confusion matrix values."""
        total = tp + fp + tn + fn
        
        # Basic metrics
        accuracy = (tp + tn) / total if total > 0 else 0.0
        precision = tp / (tp + fp) if tp + fp > 0 else 0.0
        recall = tp / (tp + fn) if tp + fn > 0 else 0.0
        specificity = tn / (tn + fp) if tn + fp > 0 else 0.0
        
        # F1 score
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
        
        # Balanced accuracy
        balanced_accuracy = (recall + specificity) / 2
        
        # Matthews Correlation Coefficient
        mcc_denominator = ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5
        mcc = ((tp * tn) - (fp * fn)) / mcc_denominator if mcc_denominator > 0 else 0.0
        
        return {
            'tp': tp,
            'fp': fp,
            'tn': tn,
            'fn': fn,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'specificity': specificity,
            'f1': f1,
            'balanced_accuracy': balanced_accuracy,
            'mcc': mcc
        }
    
    def _classify_strings_batch(self, model, strings: List[str], grammar_name: str = None, batch_size: int = 256) -> List[bool]:
        """Classify multiple strings using the RNN model in batches.
        
        Args:
            model: RNN model
            strings: List of strings to classify
            grammar_name: Name of grammar for alphabet detection
            batch_size: Number of strings to process at once
            
        Returns:
            List of boolean classifications
        """
        if not strings:
            return []
            
        model.eval()
        device = next(model.parameters()).device
        results = []
        
        # Set up character to index mapping
        if grammar_name == 'balanced_brackets':
            char_to_idx = {'(': 0, ')': 1}
        else:
            char_to_idx = {'0': 0, '1': 1}
        
        # Process in batches
        for i in range(0, len(strings), batch_size):
            batch_strings = strings[i:i + batch_size]
            
            # Group by length for efficient padding
            length_groups = {}
            for idx, s in enumerate(batch_strings):
                length = len(s)
                if length not in length_groups:
                    length_groups[length] = []
                length_groups[length].append((idx, s))
            
            batch_results = [None] * len(batch_strings)
            
            # Process each length group
            for length, group in length_groups.items():
                if length == 0:
                    # Handle empty strings specially
                    x = torch.zeros(len(group), 0, dtype=torch.long, device=device)
                else:
                    # Create padded tensor for this length group
                    indices_batch = []
                    for _, s in group:
                        indices = [char_to_idx[c] for c in s]
                        indices_batch.append(indices)
                    x = torch.tensor(indices_batch, dtype=torch.long, device=device)
                
                # Forward pass
                with torch.no_grad():
                    logits = model.forward(x)
                    probs = torch.nn.functional.softmax(logits, dim=1)
                    predictions = (probs[:, 1] > 0.5).cpu().numpy()
                
                # Store results in correct positions
                for j, (orig_idx, _) in enumerate(group):
                    batch_results[orig_idx] = bool(predictions[j])
            
            results.extend(batch_results)
        
        return results

    def _classify_string_with_rnn(self, model, string: str, grammar_name: str = None) -> bool:
        """Classify a string using the RNN model."""
        model.eval()
        
        # For balanced brackets, we need to convert parentheses to indices
        if grammar_name == 'balanced_brackets':
            # Map parentheses to indices
            char_to_idx = {'(': 0, ')': 1}
            device = next(model.parameters()).device
            
            if len(string) == 0:
                x = torch.zeros(1, 0, dtype=torch.long, device=device)
            else:
                indices = [char_to_idx[c] for c in string]
                x = torch.tensor([indices], dtype=torch.long, device=device)
            
            with torch.no_grad():
                logits = model.forward(x)
                probs = torch.nn.functional.softmax(logits, dim=1)
                return probs[0, 1].item() > 0.5
        else:
            # Default behavior for binary alphabet
            return model.classify_string(string)
    
    def _save_dfa(self, dfa, grammar_name: str, oracle_name: str, run_num: int):
        """Save extracted DFA to file in both JSON and PNG formats."""
        # Create directory structure for this run
        run_path = self.dfa_dir / grammar_name / oracle_name / f"run_{run_num + 1}"
        run_path.mkdir(parents=True, exist_ok=True)
        
        # If extraction timed out and we have learning history, use the last hypothesis
        final_dfa = dfa
        if hasattr(dfa, 'learning_history') and dfa.learning_history:
            # Check if the returned DFA is much smaller than the last hypothesis
            # This indicates a timeout where best_dfa was returned instead of the last one
            last_hypothesis = dfa.learning_history[-1]
            if 'dfa_object' in last_hypothesis:
                last_dfa = last_hypothesis['dfa_object']
                # If the returned DFA has significantly fewer states, use the last hypothesis
                if len(dfa.states) < len(last_dfa.states) * 0.5:
                    print(f"  Using last hypothesis ({len(last_dfa.states)} states) instead of best ({len(dfa.states)} states)")
                    final_dfa = last_dfa
        
        # Save final DFA as JSON
        json_file = run_path / "final.json"
        dfa_data = {
            'states': list(final_dfa.states),
            'alphabet': list(final_dfa.alphabet),
            'start_state': final_dfa.q0,
            'accept_states': list(final_dfa.F),
            'transitions': [
                {
                    'from': state,
                    'symbol': symbol,
                    'to': target
                }
                for state in final_dfa.delta
                for symbol, target in final_dfa.delta[state].items()
            ],
            'num_states': len(final_dfa.states),
            'grammar': grammar_name,
            'oracle': oracle_name,
            'run': run_num + 1
        }
        
        # Add learning history if available (exclude DFA objects)
        if hasattr(dfa, 'learning_history'):
            # Create a copy without the DFA objects
            history_for_json = []
            for item in dfa.learning_history:
                item_copy = {k: v for k, v in item.items() if k != 'dfa_object'}
                history_for_json.append(item_copy)
            dfa_data['learning_history'] = history_for_json
        
        with open(json_file, 'w') as f:
            json.dump(dfa_data, f, indent=2)
        
        # Save each intermediate hypothesis if available
        if hasattr(dfa, 'learning_history') and dfa.learning_history:
            print(f"  Saving {len(dfa.learning_history)} intermediate hypotheses...")
            
            for history_item in dfa.learning_history:
                iteration = history_item['iteration']
                
                # Skip if no DFA object (shouldn't happen with our update)
                if 'dfa_object' not in history_item:
                    continue
                    
                hypothesis_dfa = history_item['dfa_object']
                
                # Save hypothesis JSON
                hyp_json_file = run_path / f"hypothesis_{iteration}.json"
                hyp_data = {
                    'iteration': iteration,
                    'time': history_item['time'],
                    'states': list(hypothesis_dfa.states),
                    'alphabet': list(hypothesis_dfa.alphabet),
                    'start_state': hypothesis_dfa.q0,
                    'accept_states': list(hypothesis_dfa.F),
                    'transitions': [
                        {
                            'from': state,
                            'symbol': symbol,
                            'to': target
                        }
                        for state in hypothesis_dfa.delta
                        for symbol, target in hypothesis_dfa.delta[state].items()
                    ],
                    'num_states': len(hypothesis_dfa.states)
                }
                
                # Add counterexample information if this hypothesis was rejected
                if 'counterexample' in history_item and history_item['counterexample']:
                    hyp_data['counterexample_that_rejected_this'] = history_item['counterexample']
                
                with open(hyp_json_file, 'w') as f:
                    json.dump(hyp_data, f, indent=2)
                
                # Save hypothesis PNG (skip if too large)
                if len(hypothesis_dfa.states) <= 50:
                    try:
                        dot_content = hypothesis_dfa.to_dot()
                        hyp_png_file = run_path / f"hypothesis_{iteration}.png"
                        
                        # Use graphviz to convert DOT to PNG
                        import subprocess
                        import tempfile
                        
                        with tempfile.NamedTemporaryFile(mode='w', suffix='.dot', delete=False) as tmp:
                            tmp.write(dot_content)
                            tmp_path = tmp.name
                        
                        subprocess.run(['dot', '-Tpng', tmp_path, '-o', str(hyp_png_file)], 
                                     check=True, capture_output=True, timeout=3)
                        
                        # Clean up temp file
                        import os
                        os.unlink(tmp_path)
                        
                    except Exception as e:
                        pass  # Skip PNG generation on error
        
        # Save final DFA as PNG using graphviz (always generate for final DFA)
        # For very large DFAs, increase the size limit or use a different layout
        try:
            import subprocess
            dot_content = final_dfa.to_dot()
            
            # Save DOT file
            dot_file = run_path / "final.dot"
            with open(dot_file, 'w') as f:
                f.write(dot_content)
            
            # Convert to PNG with timeout - use appropriate settings for large graphs
            png_file = run_path / "final.png"
            if len(final_dfa.states) > 100:
                # For large graphs, use sfdp layout which is better for large graphs
                subprocess.run(['sfdp', '-Tpng', str(dot_file), '-o', str(png_file)], 
                             check=True, capture_output=True, timeout=30)
            else:
                subprocess.run(['dot', '-Tpng', str(dot_file), '-o', str(png_file)], 
                             check=True, capture_output=True, timeout=10)
            
            # Remove intermediate DOT file
            dot_file.unlink()
            
            print(f"  Saved DFA to: {run_path.relative_to(self.output_dir)}/ ({len(final_dfa.states)} states)")
        except (ImportError, subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired) as e:
            print(f"  Saved DFA: {json_file.relative_to(self.output_dir)} (PNG generation failed: {type(e).__name__})")
    
    def _evaluate_long_sequences_detailed(self,
                                        dfa,
                                        rnn_model,
                                        true_grammar: callable,
                                        alphabet: List[str],
                                        grammar_name: str) -> Dict[int, Dict[str, float]]:
        """Detailed evaluation on long sequences.
        
        Returns accuracy breakdown by length for both DFA-RNN and DFA-Grammar.
        """
        import random
        random.seed(42)  # For reproducibility
        
        results = {}
        # Use configured long test lengths, or analyze specific lengths if different from config
        lengths = self.long_test_lengths if hasattr(self, 'long_test_lengths') else [100, 500, 1000]
        
        for length in lengths:
            dfa_rnn_correct = 0
            dfa_grammar_correct = 0
            total = self.long_test_samples if hasattr(self, 'long_test_samples') else 100
            
            for _ in range(total):
                string = ''.join(random.choice(alphabet) for _ in range(length))
                
                # Get all classifications
                dfa_label = dfa.accepts(list(string))
                rnn_label = self._classify_string_with_rnn(rnn_model, string, grammar_name)
                true_label = true_grammar(string)
                
                # Check agreements
                if dfa_label == rnn_label:
                    dfa_rnn_correct += 1
                if dfa_label == true_label:
                    dfa_grammar_correct += 1
            
            results[length] = {
                'dfa_rnn_accuracy': dfa_rnn_correct / total if total > 0 else 0.0,
                'dfa_grammar_accuracy': dfa_grammar_correct / total if total > 0 else 0.0
            }
        
        return results
    
    def _save_intermediate_results(self):
        """Save intermediate results during benchmark."""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        # Save raw results
        json_path = self.output_dir / f"intermediate_{timestamp}.json"
        self.results.save_to_json(json_path)
        
        # Save CSV
        csv_path = self.output_dir / f"intermediate_{timestamp}.csv"
        self.results.export_to_csv(csv_path)
    
    def _save_final_results(self):
        """Save final benchmark results."""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        # Save raw results
        json_path = self.output_dir / f"benchmark_results_{timestamp}.json"
        self.results.save_to_json(json_path)
        print(f"\nResults saved to: {json_path}")
        
        # Save CSV for analysis
        csv_path = self.output_dir / f"benchmark_results_{timestamp}.csv"
        self.results.export_to_csv(csv_path)
        print(f"CSV saved to: {csv_path}")
        
        # Note: Plotting can be done by loading the saved results and using pandas/matplotlib


    def calculate_exhaustive_test_size(self, alphabet_size: int) -> int:
        """Calculate the number of strings in exhaustive test set.
        
        Args:
            alphabet_size: Size of the alphabet (e.g., 2 for binary)
            
        Returns:
            Total number of strings from length 0 to exhaustive_test_length
        """
        # Sum of alphabet_size^i for i from 0 to exhaustive_test_length
        # This equals (alphabet_size^(n+1) - 1) / (alphabet_size - 1)
        if alphabet_size == 1:
            return self.exhaustive_test_length + 1
        else:
            return (alphabet_size ** (self.exhaustive_test_length + 1) - 1) // (alphabet_size - 1)
    
    def calculate_total_test_size(self, alphabet_size: int) -> int:
        """Calculate total expected test set size.
        
        Args:
            alphabet_size: Size of the alphabet
            
        Returns:
            Total number of test strings
        """
        exhaustive = self.calculate_exhaustive_test_size(alphabet_size)
        random_samples = self.sample_size
        long_samples = len(self.long_test_lengths) * self.long_test_samples
        return exhaustive + random_samples + long_samples


def run_default_benchmark():
    """Run benchmark with default configurations."""
    # Create runner with Chapter 4 default parameters
    runner = BenchmarkRunner(
        exhaustive_test_length=15,      # Exhaustive up to length 15
        sample_size=50000,              # 50,000 random samples
        sample_min_length=16,           # Random samples from length 16
        sample_max_length=100,          # Random samples up to length 100
        long_test_lengths=[500, 1000],  # Long sequence tests
        long_test_samples=100           # 100 samples per long length
    )
    
    # Get default oracle configurations (all 5 types)
    oracle_configs = get_default_configs()
    
    # Run on all Tomita grammars + balanced brackets
    grammars = [f"tomita{i}" for i in range(1, 8)] + ["balanced_brackets"]
    
    # Run benchmark
    results = runner.run_benchmark(
        grammars=grammars,
        oracle_configs=oracle_configs,
        model_dir="trained_models",
        num_runs=3,
        time_limit=60.0
    )
    
    return results
