"""
Observation Table implementation for L* algorithm.

Maintains prefix-closed set S (states) and suffix-closed set E (experiments)
with function T: (S ∪ S·Σ) × E → {0,1} via membership queries.
"""

from typing import Set, Dict, List, Optional
import time


class TableTimedOut(Exception):
    """Raised when observation table operations exceed time limit."""
    pass


class ObservationTable:
    """Observation table for L* learning with query caching."""
    
    def __init__(self, alphabet: List[str], teacher, max_table_size: Optional[int] = None):
        """
        Initialize observation table.
        
        Args:
            alphabet: Input alphabet Σ
            teacher: Oracle providing membership queries
            max_table_size: Optional limit on |S| for memory bounds
        """
        self.S = {""}  # Start with empty string
        self.E = {""}  # Start with empty suffix
        self.A = alphabet
        self.teacher = teacher
        self.max_table_size = max_table_size
        
        self.T = {}  # Membership query cache: word → bool
        self.equal_cache = set()  # Row equivalence cache
        
        # Time management
        self.time_limit = None
        self.start_time = None
        
        # Statistics for analysis
        self.query_count = 0
        self.cache_hits = 0
        
        # Initial table filling
        self._fill_T()
        self._initiate_row_equivalence_cache()
        
    def set_time_limit(self, time_limit: float, start_time: float):
        """Set time limit for table operations."""
        self.time_limit = time_limit
        self.start_time = start_time
        
    def _assert_not_timed_out(self):
        """Check if time limit exceeded."""
        if self.time_limit is not None:
            if time.time() - self.start_time > self.time_limit:
                raise TableTimedOut("Observation table timed out")
                
    def _fill_T(self, new_e_list: Optional[List[str]] = None, 
                new_s: Optional[str] = None):
        """Update T with membership queries for new entries."""
        words_to_query = self._Trange(new_e_list, new_s)
        
        # Filter out already cached words
        uncached_words = [w for w in words_to_query if w not in self.T]
        
        # DEBUG: Track query lengths (disabled for cleaner output)
        # Uncomment below for debugging long query issues
        # if uncached_words:
        #     max_len = max(len(w) for w in uncached_words)
        #     long_queries = [w for w in uncached_words if len(w) > 20]
        #     if long_queries:
        #         print(f"  [DEBUG] Querying {len(long_queries)} strings longer than 20 (max length: {max_len})")
        #         if max_len > 30:
        #             # Show some examples of very long queries
        #             examples = sorted(long_queries, key=len, reverse=True)[:3]
        #             for ex in examples:
        #                 print(f"    - Length {len(ex)}: '{ex[:20]}...{ex[-10:]}' (s+e)")
        
        if uncached_words:
            # Batch query to teacher
            self.query_count += len(uncached_words)
            results = self.teacher.membership_queries(uncached_words)
            
            # Update cache
            for word, result in zip(uncached_words, results):
                self.T[word] = result
        else:
            self.cache_hits += len(words_to_query)
            
    def _Trange(self, new_e_list: Optional[List[str]], 
                new_s: Optional[str]) -> Set[str]:
        """
        Compute words needing classification.
        
        Returns:
            Set of words in (S ∪ S·Σ) × E domain
        """
        E = self.E if new_e_list is None else new_e_list
        
        if new_s is None:
            starts = self.S | self._SdotA()
        else:
            # Only need new_s and new_s·a for each a ∈ Σ
            starts = [new_s + a for a in [""] + list(self.A)]
            
        return {s + e for s in starts for e in E}
    
    def _SdotA(self) -> Set[str]:
        """Compute S·Σ = {s·a : s ∈ S, a ∈ Σ}."""
        return {s + a for s in self.S for a in self.A}
    
    def _initiate_row_equivalence_cache(self):
        """
        Initialize cache of row equivalences.
        
        Precomputes equivalences to avoid repeated checks during
        closure and consistency operations.
        """
        self.equal_cache = set()
        
        # Check all pairs in S ∪ S·Σ
        candidates = list(self.S) + list(self._SdotA())
        
        for s1 in candidates:
            for s2 in self.S:  # Only need representatives from S
                if self._rows_are_same(s1, s2):
                    self.equal_cache.add((s1, s2))
                    
    def _update_row_equivalence_cache(self, new_e: Optional[str] = None,
                                     new_s: Optional[str] = None):
        """
        Incrementally update row equivalence cache.
        
        When E changes: only remove invalid equivalences
        When S changes: add new equivalences for new state
        """
        if new_e is not None:
            # New experiment can only break equivalences
            remove = [(s1, s2) for s1, s2 in self.equal_cache 
                     if self.T.get(s1 + new_e) != self.T.get(s2 + new_e)]
            self.equal_cache -= set(remove)
            
        elif new_s is not None:
            # New state: check equivalences with existing states
            candidates = [new_s] + [new_s + a for a in self.A]
            
            for cand in candidates:
                for s in self.S:
                    if self._rows_are_same(cand, s):
                        self.equal_cache.add((cand, s))
                    # Also check s and s·a against new_s
                    for a in [""] + list(self.A):
                        if self._rows_are_same(s + a, new_s):
                            self.equal_cache.add((s + a, new_s))
                        
    def _rows_are_same(self, s: str, t: str) -> bool:
        """
        Check if row(s) = row(t).
        
        Definition: row(s) = row(t) iff ∀e ∈ E: T(s·e) = T(t·e)
        """
        for e in self.E:
            if self.T.get(s + e) != self.T.get(t + e):
                return False
        return True
    
    def all_live_rows(self) -> List[str]:
        """
        Get canonical representatives for all equivalence classes.
        
        Returns minimal elements from S representing distinct rows.
        Used for DFA state construction.
        """
        return [s for s in self.S if s == self.minimum_matching_row(s)]
    
    def minimum_matching_row(self, t: str) -> str:
        """
        Find canonical representative for row equivalence class.
        
        Args:
            t: Word to find representative for
            
        Returns:
            Lexicographically minimal s ∈ S with row(s) = row(t)
        """
        # Use cache for efficiency
        for s in sorted(self.S):  # Ensure deterministic choice
            if (t, s) in self.equal_cache:
                return s
        
        # Fallback if not in cache (shouldn't happen)
        for s in sorted(self.S):
            if self._rows_are_same(t, s):
                self.equal_cache.add((t, s))
                return s
                
        raise ValueError(f"No matching row found for {t}")
    
    def find_and_handle_inconsistency(self) -> bool:
        """
        Find and resolve table inconsistency.
        
        Table is inconsistent if ∃s1,s2 ∈ S, a ∈ Σ:
        - row(s1) = row(s2) but row(s1·a) ≠ row(s2·a)
        
        Resolution: Add distinguishing suffix to E
        
        Returns:
            True if inconsistency found and resolved
        """
        # Find potentially inconsistent pairs
        # We need pairs (s1, s2) where s1 ∈ S and row(s1) = row(s2)
        maybe_inconsistent = [(s1, s2, a) for s1, s2 in self.equal_cache 
                              if s1 in self.S 
                              for a in self.A
                              if (s1 + a, s2 + a) not in self.equal_cache]
        
        # For each inconsistent pair, find the distinguishing suffix
        troublemakers = []
        for s1, s2, a in maybe_inconsistent:
            # Find e such that T[s1·a·e] ≠ T[s2·a·e]
            for e in self.E:
                if self.T.get(s1 + a + e) != self.T.get(s2 + a + e):
                    troublemakers.append(a + e)
                    break
        
        if len(troublemakers) == 0:
            return False
            
        # Add the first troublemaker to E
        new_exp = troublemakers[0]
        self.E.add(new_exp)
        self._fill_T(new_e_list=troublemakers)  # Optimistic batching
        self._update_row_equivalence_cache(new_e=new_exp)
        self._assert_not_timed_out()
        return True
    
    def find_and_close_row(self) -> bool:
        """
        Find and close unclosed row.
        
        Table is closed if ∀s ∈ S, a ∈ Σ: ∃t ∈ S with row(s·a) = row(t)
        
        Resolution: Add s·a to S when no such t exists
        
        Returns:
            True if unclosed row found and added
        """
        for s in self.S:
            for a in self.A:
                sa = s + a
                
                # Check if sa has representative in S
                has_match = any((sa, t) in self.equal_cache for t in self.S)
                
                if not has_match:
                    # Add sa to S
                    self.S.add(sa)
                    self._fill_T(new_s=sa)
                    self._update_row_equivalence_cache(new_s=sa)
                    self._assert_not_timed_out()
                    return True
                    
        return False
    
    def add_counterexample(self, ce: str, label: bool):
        """
        Process counterexample by adding all prefixes to S.
        
        Follows Rivest-Schapire optimization: add all prefixes at once
        rather than discovering them incrementally.
        
        Args:
            ce: Counterexample string
            label: True if RNN accepts, False otherwise
        """
        if ce in self.S:
            print(f"bad counterexample - already saved and classified in table!")
            # Still need to ensure table is properly refined
            return
            
        # Add counterexample classification
        self.T[ce] = label
        
        # Add all prefixes to S
        new_states = []
        for i in range(len(ce) + 1):
            prefix = ce[:i]
            if prefix not in self.S:
                new_states.append(prefix)
                self.S.add(prefix)
                
        # Fill table for new states
        self._fill_T()
        
        # Update equivalence cache for each new state
        for s in new_states:
            self._update_row_equivalence_cache(new_s=s)
            
        self._assert_not_timed_out()
        
    def is_closed(self) -> bool:
        """Check if table is closed."""
        for s in self.S:
            for a in self.A:
                sa = s + a
                if not any((sa, t) in self.equal_cache for t in self.S):
                    return False
        return True
    
    def is_consistent(self) -> bool:
        """Check if table is consistent."""
        for s1, s2 in self.equal_cache:
            if s1 in self.S:
                for a in self.A:
                    if (s1 + a, s2 + a) not in self.equal_cache:
                        return False
        return True
    
    def get_statistics(self) -> Dict[str, int]:
        """Return performance statistics."""
        return {
            "states": len(self.S),
            "experiments": len(self.E),
            "cached_words": len(self.T),
            "total_queries": self.query_count,
            "cache_hits": self.cache_hits,
            "cache_hit_rate": self.cache_hits / max(1, self.cache_hits + self.query_count)
        }
    
    def debug_table_state(self):
        """Print detailed debug information about the observation table."""
        print("\n=== Observation Table Debug Info ===")
        print(f"|S| = {len(self.S)}, |E| = {len(self.E)}")
        
        # Show S (prefixes)
        s_lengths = sorted([(len(s), s) for s in self.S])
        print(f"\nS (prefixes): {len(self.S)} elements")
        print(f"  Length distribution: min={s_lengths[0][0]}, max={s_lengths[-1][0]}")
        if s_lengths[-1][0] > 15:
            print("  Long prefixes:")
            for length, s in s_lengths[-5:]:
                print(f"    Length {length}: '{s[:20]}{'...' if len(s) > 20 else ''}'")
        
        # Show E (suffixes) 
        e_lengths = sorted([(len(e), e) for e in self.E])
        print(f"\nE (suffixes): {len(self.E)} elements")
        print(f"  Length distribution: min={e_lengths[0][0]}, max={e_lengths[-1][0]}")
        if e_lengths[-1][0] > 10:
            print("  Long suffixes:")
            for length, e in e_lengths[-5:]:
                print(f"    Length {length}: '{e[:20]}{'...' if len(e) > 20 else ''}'")
        
        # Show potential max query length
        max_s = max(len(s) for s in self.S)
        max_e = max(len(e) for e in self.E)
        print(f"\nMax potential query length: {max_s} + {max_e} = {max_s + max_e}")
        
        # Count queries by length
        query_lengths = {}
        for s in self.S:
            for e in self.E:
                length = len(s + e)
                query_lengths[length] = query_lengths.get(length, 0) + 1
        
        print(f"\nQuery length distribution:")
        for length in sorted(query_lengths.keys()):
            if length > 15 or length == 0:
                print(f"  Length {length}: {query_lengths[length]} queries")
        
        print("=== End Debug Info ===\n")
    
    def __str__(self) -> str:
        """String representation for debugging."""
        lines = ["Observation Table:"]
        lines.append(f"  |S| = {len(self.S)}, |E| = {len(self.E)}")
        lines.append(f"  Closed: {self.is_closed()}, Consistent: {self.is_consistent()}")
        
        # Table visualization
        if len(self.S) <= 10:  # Only for small tables
            lines.append("\n  " + " ".join(f"{e:>3}" for e in sorted(self.E)))
            for s in sorted(self.S):
                row = [str(int(self.T.get(s + e, False))) for e in sorted(self.E)]
                lines.append(f"{s:>3} " + " ".join(f"{v:>3}" for v in row))
                
        stats = self.get_statistics()
        lines.append(f"\n  Cache hit rate: {stats['cache_hit_rate']:.2%}")
        
        return "\n".join(lines)