#!/usr/bin/env python
"""
Generate Random DFAs for Validation Testing

This script generates random DFAs with various alphabet sizes and complexities,
saving them as JSON and PNG files for reproducibility.

Usage:
    python generate_random_dfas.py --alphabet-size 2 --sizes 2 5 10 --count 3
    python generate_random_dfas.py --alphabet-size 3 --sizes 5 10 --count 2
"""

import os
import sys
import json
import random
import argparse
import subprocess
from pathlib import Path
from typing import List, Set, Dict, Optional
from collections import deque

sys.path.insert(0, str(Path(__file__).parent.parent))
from core.dfa import DFA


class RandomDFAGenerator:
    """Generate random DFAs with configurable alphabet sizes."""
    
    def __init__(self, alphabet_size: int = 2):
        """
        Initialize generator with specified alphabet size.
        
        Args:
            alphabet_size: Number of symbols in the alphabet
        """
        self.alphabet_size = alphabet_size
        # Generate alphabet symbols: for size 2: ['0', '1'], for size 3: ['a', 'b', 'c'], etc.
        if alphabet_size <= 2:
            self.alphabet = ['0', '1'][:alphabet_size]
        elif alphabet_size <= 26:
            self.alphabet = [chr(ord('a') + i) for i in range(alphabet_size)]
        else:
            # For larger alphabets, use a0, a1, a2, ...
            self.alphabet = [f'a{i}' for i in range(alphabet_size)]
    
    def generate_random_dfa(self, num_states: int, 
                          accepting_ratio: float = 0.3,
                          seed: Optional[int] = None) -> DFA:
        """
        Generate a random DFA with specified parameters.
        
        Args:
            num_states: Number of states in the DFA
            accepting_ratio: Ratio of accepting states (0.0 to 1.0)
            seed: Random seed for reproducibility
            
        Returns:
            Random DFA instance
        """
        if seed is not None:
            random.seed(seed)
            
        # Create state names
        states = {f"q{i}" for i in range(num_states)}
        
        # Initial state is always q0
        initial_state = "q0"
        
        # Random accepting states based on ratio
        num_accepting = max(1, int(num_states * accepting_ratio))
        num_accepting = min(num_accepting, num_states - 1)  # Ensure at least one rejecting state
        accepting_states = set(random.sample(list(states), num_accepting))
        
        # Create random transitions ensuring connectivity
        transitions = {}
        for state in states:
            transitions[state] = {}
            for symbol in self.alphabet:
                # Random transition to any state
                target = random.choice(list(states))
                transitions[state][symbol] = target
        
        # Ensure DFA is connected from initial state
        self._ensure_connectivity(states, initial_state, transitions)
        
        # Optionally ensure we have both accepting and rejecting reachable states
        self._ensure_diversity(states, initial_state, transitions, accepting_states)
        
        return DFA(
            states=states,
            alphabet=self.alphabet,
            transitions=transitions,
            initial_state=initial_state,
            final_states=accepting_states
        )
    
    def _ensure_connectivity(self, states: Set[str], initial: str, 
                           transitions: Dict[str, Dict[str, str]]):
        """Ensure all states are reachable from initial state."""
        # Find reachable states using BFS
        reachable = {initial}
        queue = deque([initial])
        
        while queue:
            current = queue.popleft()
            for symbol in self.alphabet:
                if current in transitions and symbol in transitions[current]:
                    next_state = transitions[current][symbol]
                    if next_state not in reachable:
                        reachable.add(next_state)
                        queue.append(next_state)
        
        # Connect unreachable states
        unreachable = states - reachable
        for state in unreachable:
            # Connect from a random reachable state
            source = random.choice(list(reachable))
            symbol = random.choice(self.alphabet)
            transitions[source][symbol] = state
            reachable.add(state)
    
    def _ensure_diversity(self, states: Set[str], initial: str,
                         transitions: Dict[str, Dict[str, str]], 
                         accepting_states: Set[str]):
        """Try to ensure both accepting and rejecting states are reachable."""
        # Find reachable states
        reachable = set()
        queue = deque([initial])
        reachable.add(initial)
        
        while queue:
            current = queue.popleft()
            for symbol in self.alphabet:
                if current in transitions and symbol in transitions[current]:
                    next_state = transitions[current][symbol]
                    if next_state not in reachable:
                        reachable.add(next_state)
                        queue.append(next_state)
        
        # Check if we have both accepting and rejecting reachable states
        reachable_accepting = reachable & accepting_states
        reachable_rejecting = reachable - accepting_states
        
        # If all reachable states are accepting, make one rejecting
        if len(reachable_rejecting) == 0 and len(reachable_accepting) > 1:
            state_to_flip = random.choice(list(reachable_accepting))
            if state_to_flip != initial:  # Don't flip initial if it's the only accepting
                accepting_states.remove(state_to_flip)
        
        # If all reachable states are rejecting, make one accepting
        elif len(reachable_accepting) == 0 and len(reachable_rejecting) > 1:
            state_to_flip = random.choice(list(reachable_rejecting))
            accepting_states.add(state_to_flip)


def save_dfa(dfa: DFA, output_path: Path, dfa_id: str):
    """Save DFA as JSON and PNG."""
    # Create DFA-specific directory
    dfa_dir = output_path / dfa_id
    dfa_dir.mkdir(parents=True, exist_ok=True)
    
    # Save as JSON
    dfa_data = {
        'id': dfa_id,
        'states': list(dfa.states),
        'alphabet': list(dfa.alphabet),
        'initial_state': dfa.q0,
        'accepting_states': list(dfa.F),
        'transitions': [
            {
                'from': state,
                'symbol': symbol,
                'to': target
            }
            for state in dfa.delta
            for symbol, target in dfa.delta[state].items()
        ],
        'num_states': len(dfa.states),
        'alphabet_size': len(dfa.alphabet)
    }
    
    json_path = dfa_dir / 'dfa.json'
    with open(json_path, 'w') as f:
        json.dump(dfa_data, f, indent=2)
    
    # Save as PNG using graphviz
    try:
        dot_content = dfa.to_dot()
        dot_path = dfa_dir / 'dfa.dot'
        png_path = dfa_dir / 'dfa.png'
        
        with open(dot_path, 'w') as f:
            f.write(dot_content)
        
        # Convert to PNG
        subprocess.run(['dot', '-Tpng', str(dot_path), '-o', str(png_path)],
                      check=True, capture_output=True)
        dot_path.unlink()  # Remove temporary DOT file
        
        print(f"  Saved {dfa_id}: {len(dfa.states)} states, alphabet size {len(dfa.alphabet)}")
        
    except (subprocess.CalledProcessError, FileNotFoundError):
        print(f"  Saved {dfa_id}: JSON only (PNG generation requires graphviz)")


def main():
    parser = argparse.ArgumentParser(
        description="Generate random DFAs for validation testing"
    )
    
    parser.add_argument(
        '--alphabet-size',
        type=int,
        default=2,
        help='Size of the alphabet (default: 2)'
    )
    
    parser.add_argument(
        '--sizes',
        type=int,
        nargs='+',
        default=[2, 5, 10, 20],
        help='DFA state counts to generate (default: 2 5 10 20)'
    )
    
    parser.add_argument(
        '--count',
        type=int,
        default=3,
        help='Number of DFAs per size (default: 3)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='validation/random_dfas',
        help='Output directory for generated DFAs'
    )
    
    parser.add_argument(
        '--accepting-ratio',
        type=float,
        default=0.3,
        help='Ratio of accepting states (default: 0.3)'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Base random seed (default: 42)'
    )
    
    args = parser.parse_args()
    
    # Create output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Initialize generator
    generator = RandomDFAGenerator(alphabet_size=args.alphabet_size)
    
    print(f"Generating Random DFAs")
    print(f"Alphabet size: {args.alphabet_size} ({generator.alphabet})")
    print(f"State counts: {args.sizes}")
    print(f"DFAs per size: {args.count}")
    print(f"Output directory: {output_path}")
    print("-" * 60)
    
    # Generate DFAs
    all_dfas = []
    
    for size in args.sizes:
        for i in range(args.count):
            # Create unique seed for each DFA
            seed = args.seed + size * 1000 + i
            
            # Generate DFA
            dfa = generator.generate_random_dfa(
                num_states=size,
                accepting_ratio=args.accepting_ratio,
                seed=seed
            )
            
            # Create ID
            dfa_id = f"alphabet{args.alphabet_size}_states{size}_v{i+1}"
            
            # Save DFA
            save_dfa(dfa, output_path, dfa_id)
            
            all_dfas.append({
                'id': dfa_id,
                'num_states': size,
                'alphabet_size': args.alphabet_size,
                'seed': seed
            })
    
    # Save metadata
    metadata = {
        'alphabet_size': args.alphabet_size,
        'alphabet': generator.alphabet,
        'sizes': args.sizes,
        'count_per_size': args.count,
        'accepting_ratio': args.accepting_ratio,
        'base_seed': args.seed,
        'dfas': all_dfas
    }
    
    with open(output_path / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\nGenerated {len(all_dfas)} DFAs")
    print(f"Metadata saved to {output_path / 'metadata.json'}")


if __name__ == "__main__":
    main()