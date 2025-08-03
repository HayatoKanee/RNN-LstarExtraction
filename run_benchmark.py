#!/usr/bin/env python
"""
Main entry point for the RNN extraction benchmark.

This script provides a comprehensive command-line interface for running
RNN-to-DFA extraction benchmarks with various configurations.

Usage:
    python run_default_benchmark.py [options]
    python run_default_benchmark.py --grammars tomita1 tomita6 --oracles whitebox
    python run_default_benchmark.py --all --runs 3
"""

import argparse
import sys
import time
from pathlib import Path
from typing import List, Dict, Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from benchmarks.benchmark_runner import BenchmarkRunner
from benchmarks.oracle_config import get_default_configs, OracleConfig, OracleType


# Expected DFA sizes for validation
EXPECTED_STATES = {
    "tomita1": 2,  # 1*
    "tomita2": 3,  # (10)*
    "tomita3": 5,  # odd 0s after odd 1s
    "tomita4": 4,  # no 000
    "tomita5": 4,  # even 0s and 1s
    "tomita6": 3,  # (#0s - #1s) mod 3 = 0
    "tomita7": 5,  # 0*1*0*1*
    "balanced_brackets": -1,  # Complex, no fixed size
}


def print_header():
    """Print the benchmark header."""
    print("=" * 70)
    print("RNN-to-DFA Extraction Benchmark Suite")
    print("=" * 70)


def parse_oracle_configs(oracle_names: Optional[List[str]]) -> Dict[str, OracleConfig]:
    """Parse oracle names and return configurations."""
    all_configs = get_default_configs()
    
    if oracle_names is None:
        return all_configs
    
    # Filter to requested oracles
    configs = {}
    for name in oracle_names:
        if name in all_configs:
            configs[name] = all_configs[name]
        else:
            print(f"Warning: Unknown oracle '{name}', skipping")
    
    return configs


def print_results_summary(results, grammars: List[str], oracle_names: List[str]):
    """Print a comprehensive summary of results."""
    print("\n" + "=" * 70)
    print("EXTRACTION RESULTS SUMMARY")
    print("=" * 70)
    
    # Header
    print(f"{'Grammar':<18}", end="")
    for oracle in oracle_names:
        print(f"{oracle:<15}", end="")
    print()
    print("-" * (18 + 15 * len(oracle_names)))
    
    # Results for each grammar
    for grammar in grammars:
        print(f"{grammar:<18}", end="")
        
        if hasattr(results, 'results') and grammar in results.results:
            grammar_result = results.results[grammar]
            
            for oracle in oracle_names:
                if oracle in grammar_result and len(grammar_result[oracle]) > 0:
                    # Get the first run's metrics
                    metrics = grammar_result[oracle][0]
                    if metrics.extraction_successful:
                        states = metrics.num_states
                        # Get both accuracy metrics directly from metrics
                        dfa_rnn_acc = metrics.dfa_rnn_accuracy
                        dfa_grammar_acc = metrics.dfa_grammar_accuracy
                        expected = EXPECTED_STATES.get(grammar, -1)
                        
                        # Choose symbol based on state count match
                        if expected == -1 or states == expected:
                            symbol = "✅"
                        else:
                            symbol = "⚠️"
                        
                        # Show states and both accuracies
                        print(f"{symbol} {states:>2}s {dfa_rnn_acc:>5.1%}/{dfa_grammar_acc:>5.1%}", end="  ")
                    else:
                        print(f"❌ FAIL        ", end="  ")
                else:
                    print(f"- N/A          ", end="  ")
        else:
            for oracle in oracle_names:
                print(f"- N/A          ", end="  ")
        print()
    
    print("\nLegend:")
    print("  ✅ = Correct number of states")
    print("  ⚠️  = Different number of states") 
    print("  ❌ = Extraction failed")
    print("  Format: [symbol] [states]s [DFA-RNN acc]/[DFA-Grammar acc]")


def main():
    """Main entry point with command line interface."""
    parser = argparse.ArgumentParser(
        description="RNN-to-DFA Extraction Benchmark Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run default benchmark (all Tomita grammars, all oracles)
  python run_default_benchmark.py
  
  # Run specific grammars with specific oracles
  python run_default_benchmark.py --grammars tomita1 tomita6 --oracles whitebox pac
  
  # Run with custom test configuration
  python run_default_benchmark.py --exhaustive-length 20 --sample-size 100000
  
  # Quick test mode
  python run_default_benchmark.py --quick --runs 1
  
  # Run with Bounded L* algorithm
  python run_default_benchmark.py --bounded-lstar --max-query-length 15
  
  # Bounded L* on specific hard cases
  python run_default_benchmark.py --bounded-lstar --grammars tomita3 tomita5 --oracles bfs pac
        """
    )
    
    # Grammar selection
    parser.add_argument(
        '--grammars',
        nargs='+',
        default=None,
        help='Grammar names to test (e.g., tomita1 tomita6 balanced_brackets). Default: all Tomita grammars'
    )
    parser.add_argument(
        '--all',
        action='store_true',
        help='Run all grammars including balanced_brackets'
    )
    
    # Oracle selection
    parser.add_argument(
        '--oracles',
        nargs='+',
        choices=['whitebox', 'pac', 'w_method', 'bfs', 'random_wp'],
        default=None,
        help='Oracle types to benchmark. Default: all oracles'
    )
    
    # Test configuration
    parser.add_argument(
        '--exhaustive-length',
        type=int,
        default=15,
        help='Maximum length for exhaustive testing (default: 15)'
    )
    parser.add_argument(
        '--sample-size',
        type=int,
        default=50000,
        help='Number of random test samples (default: 50000)'
    )
    parser.add_argument(
        '--sample-min-length',
        type=int,
        default=16,
        help='Minimum length for random samples (default: 16)'
    )
    parser.add_argument(
        '--sample-max-length',
        type=int,
        default=100,
        help='Maximum length for random samples (default: 100)'
    )
    parser.add_argument(
        '--long-test-lengths',
        nargs='+',
        type=int,
        default=[500, 1000],
        help='Lengths for long sequence tests (default: 500 1000)'
    )
    parser.add_argument(
        '--long-test-samples',
        type=int,
        default=100,
        help='Number of samples per long test length (default: 100)'
    )
    
    # Bounded L* configuration
    parser.add_argument(
        '--bounded-lstar',
        action='store_true',
        help='Use Bounded L* algorithm with negotiation protocol'
    )
    parser.add_argument(
        '--max-query-length',
        type=int,
        default=20,
        help='Maximum query length for Bounded L* (default: 20)'
    )
    
    # Starting examples configuration
    parser.add_argument(
        '--use-starting-examples',
        action='store_true',
        help='Use starting examples for all oracles (not just whitebox)'
    )
    
    # Execution configuration
    parser.add_argument(
        '--runs',
        type=int,
        default=3,
        help='Number of runs per configuration (default: 3)'
    )
    parser.add_argument(
        '--time-limit',
        type=float,
        default=60.0,
        help='Time limit per extraction in seconds (default: 60)'
    )
    parser.add_argument(
        '--model-dir',
        type=str,
        default='trained_models',
        help='Directory containing trained models (default: trained_models)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory for results (default: benchmark_results or bounded_benchmark_results)'
    )
    
    # Parallel execution
    parser.add_argument(
        '--workers',
        type=int,
        default=None,
        help='Number of parallel workers (default: CPU count, use 1 for sequential)'
    )
    
    # Quick test mode
    parser.add_argument(
        '--quick',
        action='store_true',
        help='Quick test mode with reduced parameters'
    )
    
    # Verbosity
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose output'
    )
    
    args = parser.parse_args()
    
    # Handle quick mode
    if args.quick:
        args.exhaustive_length = 8
        args.sample_size = 1000
        args.sample_max_length = 20
        args.long_test_lengths = []
        args.long_test_samples = 0
        if args.runs == 3:  # Only override if not explicitly set
            args.runs = 1
    
    # Determine grammars to test
    if args.all:
        grammars = [f"tomita{i}" for i in range(1, 8)] + ["balanced_brackets"]
    elif args.grammars:
        grammars = args.grammars
    else:
        # Default: all Tomita grammars
        grammars = [f"tomita{i}" for i in range(1, 8)]
    
    # Set default output directory based on bounded L* usage
    if args.output_dir is None:
        args.output_dir = 'bounded_benchmark_results' if args.bounded_lstar else 'benchmark_results'
    
    # Get oracle configurations
    oracle_configs = parse_oracle_configs(args.oracles)
    
    # Update configurations for bounded L* if requested
    if args.bounded_lstar:
        for config in oracle_configs.values():
            config.use_bounded_lstar = True
            config.max_query_length = args.max_query_length
    
    # Update configurations for starting examples if requested
    if args.use_starting_examples:
        for config in oracle_configs.values():
            config.use_starting_examples_override = True
            
            # Adjust BFS oracle depth to ensure counterexamples are processable
            if config.oracle_type == OracleType.BFS:
                # Set max_depth based on query length to avoid unprocessable counterexamples
                # Conservative estimate: max_depth should be at most max_query_length/2
                config.max_depth = min(config.max_depth, args.max_query_length // 2)
                if args.verbose:
                    print(f"  Adjusted BFS max_depth to {config.max_depth} for bounded L*")
    
    if not oracle_configs:
        print("Error: No valid oracles specified")
        return 1
    
    # Print configuration
    print_header()
    print(f"\nConfiguration:")
    print(f"  Grammars: {', '.join(grammars)}")
    print(f"  Oracles: {', '.join(oracle_configs.keys())}")
    print(f"  Algorithm: {'Bounded L*' if args.bounded_lstar else 'Standard L*'}")
    if args.bounded_lstar:
        print(f"  Max query length: {args.max_query_length}")
    print(f"  Runs per config: {args.runs}")
    print(f"  Time limit: {args.time_limit}s")
    print(f"  Test set: exhaustive up to {args.exhaustive_length}, "
          f"{args.sample_size:,} samples ({args.sample_min_length}-{args.sample_max_length})")
    if args.long_test_lengths:
        print(f"  Long tests: {args.long_test_samples} samples at lengths {args.long_test_lengths}")
    print()
    
    # Create benchmark runner
    runner = BenchmarkRunner(
        output_dir=args.output_dir,
        exhaustive_test_length=args.exhaustive_length,
        sample_size=args.sample_size,
        sample_min_length=args.sample_min_length,
        sample_max_length=args.sample_max_length,
        long_test_lengths=args.long_test_lengths,
        long_test_samples=args.long_test_samples,
        num_workers=args.workers
    )
    
    # Run benchmark
    start_time = time.time()
    try:
        results = runner.run_benchmark(
            grammars=grammars,
            oracle_configs=oracle_configs,
            model_dir=args.model_dir,
            num_runs=args.runs,
            time_limit=args.time_limit
        )
        
        # Print summary
        print_results_summary(results, grammars, list(oracle_configs.keys()))
        
        total_time = time.time() - start_time
        print(f"\nTotal benchmark time: {total_time:.1f} seconds")
        print(f"Results saved to: {args.output_dir}/")
        
        return 0
        
    except KeyboardInterrupt:
        print("\n\nBenchmark interrupted by user.")
        return 1
    except Exception as e:
        print(f"\n\nError during benchmark: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())