"""
Deterministic Finite Automaton (DFA) implementation.

A DFA is formally a 5-tuple (Q, Σ, δ, q₀, F) where Q is the state set,
Σ is the alphabet, δ: Q × Σ → Q is the transition function,
q₀ is the initial state, and F is the set of accepting states.
"""

from typing import Set, Dict, List, Optional, Tuple
from collections import deque


class DFA:
    """Deterministic Finite Automaton with L*-specific operations."""
    
    def __init__(self, 
                 states: Optional[Set[str]] = None,
                 alphabet: Optional[List[str]] = None,
                 transitions: Optional[Dict[str, Dict[str, str]]] = None,
                 initial_state: Optional[str] = None,
                 final_states: Optional[Set[str]] = None,
                 obs_table=None):
        """
        Initialize DFA either directly or from observation table.
        
        Args:
            states: Set of state identifiers
            alphabet: List of alphabet symbols
            transitions: Nested dict mapping state × symbol → state
            initial_state: Starting state identifier
            final_states: Set of accepting state identifiers
            obs_table: ObservationTable instance (alternative construction)
        """
        if obs_table is not None:
            self._construct_from_observation_table(obs_table)
        else:
            self.states = states or set()
            self.alphabet = alphabet or []
            self.delta = transitions or {}
            self.q0 = initial_state
            self.F = final_states or set()
            
    def _construct_from_observation_table(self, table):
        """
        Construct DFA from closed and consistent observation table.
        
        States correspond to distinct rows in S, with transitions
        determined by row equivalences.
        """
        # Extract live rows (canonical representatives)
        live_rows = table.all_live_rows()
        
        # Initialize DFA components
        self.states = set(live_rows)
        self.alphabet = list(table.A)
        self.q0 = ""  # Empty string is always initial state
        
        # Determine accepting states
        self.F = {s for s in live_rows if table.T.get(s, False)}
        
        # Build transition function
        self.delta = {}
        for state in live_rows:
            self.delta[state] = {}
            for symbol in self.alphabet:
                # Find representative for state·symbol
                next_state = table.minimum_matching_row(state + symbol)
                self.delta[state][symbol] = next_state
                
    def accepts(self, word: List[str]) -> bool:
        """
        Determine if DFA accepts given word.
        
        Args:
            word: List of symbols from alphabet
            
        Returns:
            True if word leads to accepting state
            
        Time Complexity: O(|word|)
        """
        current_state = self.q0
        
        for symbol in word:
            if symbol not in self.alphabet:
                return False
            if current_state not in self.delta:
                return False
            current_state = self.delta[current_state].get(symbol)
            if current_state is None:
                return False
                
        return current_state in self.F
    
    def classify_word(self, word: str) -> bool:
        """
        Classify word as accepted/rejected (string input variant).
        
        Args:
            word: String of concatenated symbols
            
        Returns:
            True if accepted, False otherwise
        """
        return self.accepts(list(word))
    
    def minimal_diverging_suffix(self, state1: str, state2: str) -> str:
        """
        Find shortest suffix distinguishing two states.
        
        Uses BFS to find minimal witness for state inequivalence.
        Essential for counterexample processing in L*.
        
        Args:
            state1: First state identifier
            state2: Second state identifier
            
        Returns:
            Shortest suffix where states diverge in acceptance
            
        Time Complexity: O(|Q| × |Σ|) worst case
        """
        if (state1 in self.F) != (state2 in self.F):
            return ""  # Empty suffix distinguishes
            
        # BFS for shortest distinguishing suffix
        queue = deque([("", state1, state2)])
        visited = {(state1, state2)}
        
        while queue:
            suffix, s1, s2 = queue.popleft()
            
            for symbol in self.alphabet:
                if s1 in self.delta and s2 in self.delta:
                    next_s1 = self.delta[s1].get(symbol)
                    next_s2 = self.delta[s2].get(symbol)
                    
                    if next_s1 and next_s2:
                        new_suffix = suffix + symbol
                        
                        # Check if states differ in acceptance
                        if (next_s1 in self.F) != (next_s2 in self.F):
                            return new_suffix
                            
                        # Continue search if not visited
                        if (next_s1, next_s2) not in visited:
                            visited.add((next_s1, next_s2))
                            queue.append((new_suffix, next_s1, next_s2))
                            
        return None  # States are equivalent
    
    def get_state_after(self, word: str) -> Optional[str]:
        """
        Get state reached after processing word.
        
        Args:
            word: Input string
            
        Returns:
            State identifier or None if undefined
        """
        current = self.q0
        for symbol in word:
            if current not in self.delta or symbol not in self.delta[current]:
                return None
            current = self.delta[current][symbol]
        return current
    
    def minimize(self) -> 'DFA':
        """
        Return minimized equivalent DFA using Hopcroft's algorithm.
        
        Returns:
            New minimized DFA instance
            
        Time Complexity: O(|Σ| × n × log n) where n = |Q|
        """
        # Partition refinement implementation
        # Initial partition: accepting vs non-accepting
        P = [self.F, self.states - self.F]
        P = [p for p in P if p]  # Remove empty partitions
        
        # Refine partitions
        while True:
            refined = False
            new_P = []
            
            for partition in P:
                # Try to split partition
                splits = self._refine_partition(partition, P)
                if len(splits) > 1:
                    refined = True
                new_P.extend(splits)
                
            P = new_P
            if not refined:
                break
                
        # Construct minimized DFA from partitions
        return self._build_minimized_dfa(P)
    
    def _refine_partition(self, partition: Set[str], 
                         all_partitions: List[Set[str]]) -> List[Set[str]]:
        """Helper for partition refinement in minimization."""
        if len(partition) <= 1:
            return [partition]
            
        # Group states by behavior
        groups = {}
        for state in partition:
            signature = []
            for symbol in self.alphabet:
                if state in self.delta and symbol in self.delta[state]:
                    next_state = self.delta[state][symbol]
                    # Find partition containing next_state
                    for i, p in enumerate(all_partitions):
                        if next_state in p:
                            signature.append(i)
                            break
                else:
                    signature.append(-1)
            
            sig_tuple = tuple(signature)
            if sig_tuple not in groups:
                groups[sig_tuple] = set()
            groups[sig_tuple].add(state)
            
        return list(groups.values())
    
    def _build_minimized_dfa(self, partitions: List[Set[str]]) -> 'DFA':
        """Construct minimized DFA from partition refinement result."""
        # Map states to partition representatives
        state_to_part = {}
        part_representatives = {}
        
        for i, partition in enumerate(partitions):
            representative = min(partition)  # Canonical choice
            part_representatives[i] = representative
            for state in partition:
                state_to_part[state] = i
                
        # Build new DFA
        new_states = set(part_representatives.values())
        new_transitions = {}
        new_initial = part_representatives[state_to_part[self.q0]]
        new_final = set()
        
        for i, partition in enumerate(partitions):
            rep = part_representatives[i]
            new_transitions[rep] = {}
            
            # Check if accepting partition
            if any(s in self.F for s in partition):
                new_final.add(rep)
                
            # Build transitions from any state in partition
            sample_state = next(iter(partition))
            if sample_state in self.delta:
                for symbol in self.alphabet:
                    if symbol in self.delta[sample_state]:
                        next_state = self.delta[sample_state][symbol]
                        next_part = state_to_part[next_state]
                        new_transitions[rep][symbol] = part_representatives[next_part]
                        
        return DFA(
            states=new_states,
            alphabet=self.alphabet,
            transitions=new_transitions,
            initial_state=new_initial,
            final_states=new_final
        )
    
    def __len__(self) -> int:
        """Return number of states."""
        return len(self.states)
    
    def __str__(self) -> str:
        """String representation for debugging."""
        return (f"DFA(|Q|={len(self.states)}, |Σ|={len(self.alphabet)}, "
                f"q0={self.q0}, |F|={len(self.F)})")
    
    def to_dot(self) -> str:
        """
        Generate Graphviz DOT representation.
        
        Returns:
            DOT format string for visualization
        """
        lines = ["digraph DFA {", "    rankdir=LR;", "    node [shape=circle];"]
        
        # Mark accepting states
        for state in self.F:
            lines.append(f'    "{state}" [shape=doublecircle];')
            
        # Initial state arrow
        lines.append(f'    __start__ [shape=none, label=""];')
        lines.append(f'    __start__ -> "{self.q0}";')
        
        # Transitions
        for state in self.states:
            if state in self.delta:
                # Group transitions by target
                trans_groups = {}
                for symbol, target in self.delta[state].items():
                    if target not in trans_groups:
                        trans_groups[target] = []
                    trans_groups[target].append(symbol)
                    
                # Create edges with combined labels
                for target, symbols in trans_groups.items():
                    label = ",".join(sorted(symbols))
                    lines.append(f'    "{state}" -> "{target}" [label="{label}"];')
                    
        lines.append("}")
        return "\n".join(lines)