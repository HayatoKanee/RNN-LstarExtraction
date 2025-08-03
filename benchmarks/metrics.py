"""
Metrics collection and storage for benchmarking DFA extraction methods.

This module provides comprehensive metrics tracking for comparing different
equivalence oracle approaches in the L* algorithm.
"""

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


@dataclass
class ExtractionMetrics:
    """Metrics collected during a single DFA extraction run.
    
    Key accuracy metrics:
    - dfa_rnn_accuracy: How well the extracted DFA matches the RNN's behavior
    - dfa_grammar_accuracy: How well the extracted DFA matches the true grammar
    """
    
    # Timing metrics
    total_time: float = 0.0
    equivalence_query_time: float = 0.0
    membership_query_time: float = 0.0
    partitioning_time: float = 0.0  # For whitebox oracle
    
    # Query counts
    membership_queries: int = 0
    equivalence_queries: int = 0
    counterexamples_found: int = 0
    
    # Counterexample statistics
    counterexample_lengths: List[int] = field(default_factory=list)
    
    # DFA properties
    num_states: int = 0
    
    # Primary accuracy metrics
    dfa_rnn_accuracy: float = 0.0  # Agreement between DFA and RNN
    dfa_grammar_accuracy: float = 0.0  # Agreement between DFA and true grammar
    
    # Detailed accuracy by test set type
    accuracy_by_length: Dict[str, Dict[str, float]] = field(default_factory=dict)
    # Structure: {
    #   'exhaustive': {'dfa_rnn': 0.98, 'dfa_grammar': 0.95},
    #   'random': {'dfa_rnn': 0.97, 'dfa_grammar': 0.94},
    #   'long_100': {'dfa_rnn': 0.95, 'dfa_grammar': 0.90},
    #   'long_500': {'dfa_rnn': 0.93, 'dfa_grammar': 0.88},
    # }
    
    # Method-specific metrics
    oracle_specific: Dict[str, Any] = field(default_factory=dict)
    
    # Extraction success
    extraction_successful: bool = False
    failure_reason: Optional[str] = None
    
    # L* algorithm metrics
    iterations: int = 0  # Number of L* iterations
    observation_table_size: Tuple[int, int] = (0, 0)  # (|S|, |E|)
    
    @property
    def avg_counterexample_length(self) -> float:
        """Average length of counterexamples found."""
        if not self.counterexample_lengths:
            return 0.0
        return sum(self.counterexample_lengths) / len(self.counterexample_lengths)
    
    @property
    def queries_per_state(self) -> float:
        """Average number of membership queries per DFA state."""
        if self.num_states == 0:
            return 0.0
        return self.membership_queries / self.num_states
    
    @property
    def overall_accuracy(self) -> float:
        """Overall accuracy (average of DFA-RNN and DFA-Grammar)."""
        return (self.dfa_rnn_accuracy + self.dfa_grammar_accuracy) / 2


class MetricsCollector:
    """Collects metrics during DFA extraction process."""
    
    def __init__(self):
        self.metrics = ExtractionMetrics()
        self._start_time = None
        self._phase_start_time = None
        self._current_phase = None
    
    def start_extraction(self):
        """Start timing the extraction process."""
        self._start_time = time.time()
        self.metrics = ExtractionMetrics()
    
    def end_extraction(self, successful: bool = True, failure_reason: str = None):
        """End the extraction process and record final time."""
        if self._start_time:
            self.metrics.total_time = time.time() - self._start_time
        self.metrics.extraction_successful = successful
        self.metrics.failure_reason = failure_reason
    
    def start_phase(self, phase: str):
        """Start timing a specific phase (e.g., 'equivalence_query')."""
        self._current_phase = phase
        self._phase_start_time = time.time()
    
    def end_phase(self):
        """End timing the current phase."""
        if self._phase_start_time and self._current_phase:
            elapsed = time.time() - self._phase_start_time
            
            if self._current_phase == 'equivalence_query':
                self.metrics.equivalence_query_time += elapsed
            elif self._current_phase == 'membership_query':
                self.metrics.membership_query_time += elapsed
            elif self._current_phase == 'partitioning':
                self.metrics.partitioning_time += elapsed
    
    def record_membership_query(self):
        """Record a membership query."""
        self.metrics.membership_queries += 1
    
    def record_equivalence_query(self):
        """Record an equivalence query."""
        self.metrics.equivalence_queries += 1
    
    def record_counterexample(self, counterexample: str):
        """Record a counterexample found."""
        self.metrics.counterexamples_found += 1
        self.metrics.counterexample_lengths.append(len(counterexample))
    
    def record_dfa_properties(self, num_states: int):
        """Record properties of the extracted DFA."""
        self.metrics.num_states = num_states
    
    def record_accuracy(self, dfa_rnn_acc: float, dfa_grammar_acc: float):
        """Record the primary accuracy metrics.
        
        Args:
            dfa_rnn_acc: How well DFA matches RNN (0-1)
            dfa_grammar_acc: How well DFA matches true grammar (0-1)
        """
        self.metrics.dfa_rnn_accuracy = dfa_rnn_acc
        self.metrics.dfa_grammar_accuracy = dfa_grammar_acc
    
    def record_accuracy_by_length(self, test_type: str, dfa_rnn: float, dfa_grammar: float):
        """Record accuracy on a specific test set.
        
        Args:
            test_type: Type of test set (e.g., 'exhaustive', 'random', 'long_100')
            dfa_rnn: DFA-RNN accuracy on this test set
            dfa_grammar: DFA-Grammar accuracy on this test set
        """
        self.metrics.accuracy_by_length[test_type] = {
            'dfa_rnn': dfa_rnn,
            'dfa_grammar': dfa_grammar
        }
    
    def record_oracle_specific(self, key: str, value: Any):
        """Record oracle-specific metrics."""
        self.metrics.oracle_specific[key] = value
    
    def record_lstar_progress(self, iteration: int, s_size: int, e_size: int):
        """Record L* algorithm progress.
        
        Args:
            iteration: Current iteration number
            s_size: Size of S (prefixes) in observation table
            e_size: Size of E (suffixes) in observation table
        """
        self.metrics.iterations = iteration
        self.metrics.observation_table_size = (s_size, e_size)
    
    def get_metrics(self) -> ExtractionMetrics:
        """Get the collected metrics."""
        return self.metrics


@dataclass
class BenchmarkResults:
    """Stores and analyzes results from multiple benchmark runs."""
    
    results: Dict[str, Dict[str, List[ExtractionMetrics]]] = field(default_factory=dict)
    # Structure: {grammar: {oracle_type: [metrics1, metrics2, ...]}}
    
    def add_result(self, grammar: str, oracle_type: str, metrics: ExtractionMetrics):
        """Add a benchmark result."""
        if grammar not in self.results:
            self.results[grammar] = {}
        if oracle_type not in self.results[grammar]:
            self.results[grammar][oracle_type] = []
        self.results[grammar][oracle_type].append(metrics)
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert results to a pandas DataFrame for analysis."""
        data = []
        for grammar, oracle_results in self.results.items():
            for oracle_type, metrics_list in oracle_results.items():
                for i, metrics in enumerate(metrics_list):
                    row = {
                        'grammar': grammar,
                        'oracle_type': oracle_type,
                        'run': i,
                        'total_time': metrics.total_time,
                        'membership_queries': metrics.membership_queries,
                        'equivalence_queries': metrics.equivalence_queries,
                        'counterexamples': metrics.counterexamples_found,
                        'num_states': metrics.num_states,
                        'dfa_rnn_accuracy': metrics.dfa_rnn_accuracy,
                        'dfa_grammar_accuracy': metrics.dfa_grammar_accuracy,
                        'overall_accuracy': metrics.overall_accuracy,
                        'extraction_successful': metrics.extraction_successful,
                        'avg_counterexample_length': metrics.avg_counterexample_length,
                        'queries_per_state': metrics.queries_per_state,
                        'iterations': metrics.iterations,
                        'obs_table_s': metrics.observation_table_size[0],
                        'obs_table_e': metrics.observation_table_size[1],
                    }
                    
                    # Add oracle-specific metrics
                    for k, v in metrics.oracle_specific.items():
                        row[f'oracle_{k}'] = v
                    
                    # Add accuracy by length metrics
                    for test_type, accuracies in metrics.accuracy_by_length.items():
                        row[f'{test_type}_dfa_rnn'] = accuracies['dfa_rnn']
                        row[f'{test_type}_dfa_grammar'] = accuracies['dfa_grammar']
                    
                    data.append(row)
                    
        return pd.DataFrame(data)
    
    def plot_accuracy_comparison(self, save_path: Optional[Path] = None):
        """Plot DFA-RNN vs DFA-Grammar accuracy for different oracles."""
        df = self.to_dataframe()
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Prepare data for plotting
        plot_data = []
        for _, row in df.iterrows():
            plot_data.append({
                'Oracle': row['oracle_type'],
                'Grammar': row['grammar'],
                'Accuracy Type': 'DFA-RNN',
                'Accuracy': row['dfa_rnn_accuracy']
            })
            plot_data.append({
                'Oracle': row['oracle_type'],
                'Grammar': row['grammar'],
                'Accuracy Type': 'DFA-Grammar',
                'Accuracy': row['dfa_grammar_accuracy']
            })
        
        plot_df = pd.DataFrame(plot_data)
        
        # Create grouped bar plot
        sns.barplot(data=plot_df, x='Oracle', y='Accuracy', hue='Accuracy Type', ax=ax)
        ax.set_ylabel('Accuracy')
        ax.set_title('DFA-RNN vs DFA-Grammar Accuracy by Oracle Type')
        ax.set_ylim(0, 1.1)
        
        # Add value labels on bars
        for container in ax.containers:
            ax.bar_label(container, fmt='%.2f')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.close()
    
    def plot_efficiency_vs_accuracy(self, save_path: Optional[Path] = None):
        """Plot the trade-off between efficiency (queries) and accuracy."""
        df = self.to_dataframe()
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: Queries vs Overall Accuracy
        sns.scatterplot(data=df, x='membership_queries', y='overall_accuracy',
                       hue='oracle_type', style='grammar', s=100, ax=ax1)
        ax1.set_xlabel('Membership Queries')
        ax1.set_ylabel('Overall Accuracy')
        ax1.set_title('Query Efficiency vs Accuracy Trade-off')
        
        # Plot 2: Time vs Accuracy
        sns.scatterplot(data=df, x='total_time', y='overall_accuracy',
                       hue='oracle_type', style='grammar', s=100, ax=ax2)
        ax2.set_xlabel('Total Time (s)')
        ax2.set_ylabel('Overall Accuracy')
        ax2.set_title('Time Efficiency vs Accuracy Trade-off')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.close()
    
    def export_to_csv(self, path: Path):
        """Export results to CSV file."""
        df = self.to_dataframe()
        df.to_csv(path, index=False)
    
    def save_to_json(self, path: Path):
        """Save raw results to JSON file."""
        # Convert metrics to serializable format
        serializable_results = {}
        for grammar, oracle_results in self.results.items():
            serializable_results[grammar] = {}
            for oracle_type, metrics_list in oracle_results.items():
                serializable_results[grammar][oracle_type] = [
                    {
                        'total_time': m.total_time,
                        'membership_queries': m.membership_queries,
                        'equivalence_queries': m.equivalence_queries,
                        'counterexamples_found': m.counterexamples_found,
                        'num_states': m.num_states,
                        'dfa_rnn_accuracy': m.dfa_rnn_accuracy,
                        'dfa_grammar_accuracy': m.dfa_grammar_accuracy,
                        'extraction_successful': m.extraction_successful,
                        'counterexample_lengths': m.counterexample_lengths,
                        'accuracy_by_length': m.accuracy_by_length,
                        'oracle_specific': m.oracle_specific,
                        'iterations': m.iterations,
                        'observation_table_size': list(m.observation_table_size),
                    }
                    for m in metrics_list
                ]
        
        with open(path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
    
    def print_summary(self):
        """Print a clear summary of benchmark results."""
        df = self.to_dataframe()
        
        print("\n" + "="*80)
        print("BENCHMARK RESULTS SUMMARY")
        print("="*80)
        
        for grammar in df['grammar'].unique():
            print(f"\nGrammar: {grammar}")
            print("-" * 60)
            
            grammar_df = df[df['grammar'] == grammar]
            
            # Create summary with clear column names
            summary = grammar_df.groupby('oracle_type').agg({
                'total_time': 'mean',
                'dfa_rnn_accuracy': 'mean',
                'dfa_grammar_accuracy': 'mean',
                'membership_queries': 'mean',
                'num_states': 'mean',
                'extraction_successful': 'mean'
            }).round(3)
            
            # Rename columns for clarity
            summary.columns = [
                'Avg Time (s)',
                'DFA-RNN Acc',
                'DFA-Grammar Acc', 
                'Avg Queries',
                'Avg States',
                'Success Rate'
            ]
            
            print(summary.to_string())
            
        print("\n" + "="*80)