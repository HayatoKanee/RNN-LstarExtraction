# RNN-to-DFA Extraction Framework

Extract interpretable deterministic finite automata (DFA) from any trained recurrent neural network using Angluin's L* algorithm.

## Overview

This framework provides a modular implementation for extracting finite automata from RNNs, making their decision logic transparent and verifiable. It supports multiple extraction strategies and allows easy benchmarking of different approaches.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run demo notebook
jupyter notebook demo.ipynb

# Run full benchmark
python run_benchmark.py

# Or run specific configurations
python run_benchmark.py --oracles pac whitebox --grammars tomita3 --workers 1
```

## Usage Example

```python
from rnn_extraction import DFAExtractor

# Load any trained RNN model
model = torch.load("your_model.pt")

# Extract DFA using your choice of oracle
extractor = DFAExtractor(model, alphabet=['0', '1'])
dfa = extractor.extract(oracle_type='pac', time_limit=60.0)

# Use the extracted DFA
print(f"DFA has {len(dfa.states)} states")
print(dfa.accepts(['1', '0', '1']))  # True/False

# Save and visualize (requires Graphviz installed)
dot_content = dfa.to_dot()
with open('extracted_dfa.dot', 'w') as f:
    f.write(dot_content)
# Convert to PNG using Graphviz
import subprocess
subprocess.run(['dot', '-Tpng', 'extracted_dfa.dot', '-o', 'extracted_dfa.png'])
```

## Implementing Custom Oracles

```python
from rnn_extraction.counterexample import EquivalenceOracle

class MyOracle(EquivalenceOracle):
    def find_counterexample(self, hypothesis_dfa, iteration, time_limit=None):
        # Your custom logic here
        return counterexample_string or None
```

## Project Structure

```
rnn_extraction/
├── core/           # L* algorithm implementation
├── counterexample/ # Oracle strategies
├── models/         # RNN architectures
├── extraction/     # High-level API
└── benchmarks/     # Evaluation framework
```