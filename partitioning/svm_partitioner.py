"""
SVM-based partitioning following Weiss et al. (2018).

Implements a decision tree structure where:
- Initial splits use simple dimension-based partitioning
- Subsequent splits use SVM for better separation
- Each leaf node represents a partition/cluster
"""

import numpy as np
from typing import List, Optional
from sklearn import svm
from copy import deepcopy


class SVMDecisionTreeNode:
    """A node in the decision tree partitioning structure."""
    
    def __init__(self, id: int):
        self.id = id
        self.has_children = False
        self.is_dim_split = False
        
        # For dimension splits
        self.split_dim = None
        self.split_val = None
        self.high = None
        self.low = None
        
        # For SVM splits
        self.clf = None
        self.zero_child = None
        self.one_child = None
        
    def get_node(self, vector: np.ndarray):
        """Find the leaf node for a given vector."""
        if self.has_children:
            return self._choose_child(vector).get_node(vector)
        return self
        
    def _choose_child(self, vector: np.ndarray):
        """Choose which child to traverse."""
        if self.is_dim_split:
            return self._dim_choose_child(vector)
        
        # SVM split
        prediction = self.clf.predict([vector])[0]
        return self.zero_child if prediction == 0 else self.one_child
        
    def _dim_choose_child(self, vector: np.ndarray):
        """Choose child for dimension split."""
        if vector[self.split_dim] > self.split_val:
            return self.high
        return self.low
        
    def dim_split(self, agreeing_states: List[np.ndarray], 
                  conflicted_state: np.ndarray, new_id: int, 
                  split_depth: int) -> int:
        """
        Create dimension-based splits (used for initial partitioning).
        
        Args:
            agreeing_states: States that should be in same partition
            conflicted_state: State that needs to be separated
            new_id: Next available ID
            split_depth: How many dimensions to split
            
        Returns:
            Next available ID after creating all nodes
        """
        # Find dimensions with largest margins
        mean_agreeing = np.mean(agreeing_states, axis=0)
        margins = np.abs(mean_agreeing - conflicted_state)
        
        # Sort dimensions by margin size
        dim_order = np.argsort(margins)[::-1]
        split_vals = (mean_agreeing + conflicted_state) / 2.0
        
        # Create splits recursively
        return self._dim_split_recursive(
            dim_order[:split_depth], split_vals, new_id, split_depth)
        
    def _dim_split_recursive(self, dims_to_split: np.ndarray, 
                           split_vals: np.ndarray, new_id: int, 
                           depth: int) -> int:
        """Recursively create dimension splits."""
        if depth == 0 or len(dims_to_split) == 0:
            return new_id
            
        # Split on first dimension
        self.split_dim = dims_to_split[0]
        self.split_val = split_vals[self.split_dim]
        
        # Create children
        self.high = SVMDecisionTreeNode(self.id)  # Keep same ID
        self.low = SVMDecisionTreeNode(new_id)
        new_id += 1
        
        self.has_children = True
        self.is_dim_split = True
        
        # Recurse on children
        remaining_dims = dims_to_split[1:]
        new_id = self.high._dim_split_recursive(
            remaining_dims, split_vals, new_id, depth - 1)
        new_id = self.low._dim_split_recursive(
            remaining_dims, split_vals, new_id, depth - 1)
            
        return new_id
        
    def split(self, agreeing_states: List[np.ndarray], 
              conflicted_state: np.ndarray, new_id: int) -> int:
        """
        Create SVM-based split.
        
        Args:
            agreeing_states: States that should be in same partition
            conflicted_state: State that needs to be separated
            new_id: Next available ID
            
        Returns:
            Next available ID after creating children
        """
        # Prepare training data
        X = agreeing_states + [conflicted_state]
        y = [0] * len(agreeing_states) + [1]
        
        # Train SVM with RBF kernel
        self.clf = svm.SVC(kernel='rbf', C=10000, gamma='scale')
        self.clf.fit(X, y)
        
        # Verify perfect split
        predictions = self.clf.predict(X).tolist()
        if predictions != y:
            # Fall back to linear kernel if RBF fails
            self.clf = svm.SVC(kernel='linear', C=10000)
            self.clf.fit(X, y)
            predictions = self.clf.predict(X).tolist()
            if predictions != y:
                raise ValueError("SVM failed to achieve perfect split")
        
        # Create children
        self.zero_child = SVMDecisionTreeNode(self.id)  # Keep same ID
        self.one_child = SVMDecisionTreeNode(new_id)
        new_id += 1
        
        self.has_children = True
        self.is_dim_split = False
        
        return new_id


class SVMPartitioner:
    """
    Decision tree-based partitioning for RNN state space.
    
    Following Weiss et al. (2018):
    - Initial refinement uses dimension splits for efficiency
    - Subsequent refinements use SVM for accuracy
    - Tree structure ensures clean, local refinements
    """
    
    def __init__(self, initial_split_depth: int = 5):
        """
        Initialize partitioner.
        
        Args:
            initial_split_depth: Number of dimensions to split initially
        """
        self.initial_split_depth = initial_split_depth
        self.next_id = 1
        self.root = SVMDecisionTreeNode(self.next_id)
        self.initial_split_done = False
        
        # Property that whitebox algorithm relies on
        self.refinement_doesnt_hurt_other_clusters = True
        
    def get_partition(self, vector: np.ndarray) -> int:
        """
        Get partition ID for a state vector.
        
        Args:
            vector: State vector
            
        Returns:
            Partition ID
        """
        return self.root.get_node(vector).id
        
    def refine(self, agreeing_states: List[np.ndarray],
               conflicted_state: np.ndarray) -> bool:
        """
        Refine partitioning to separate conflicted state.
        
        Args:
            agreeing_states: States that should remain together
            conflicted_state: State that needs separate partition
            
        Returns:
            True if refinement succeeded
        """
        # Find the node containing the conflicted state
        node = self.root.get_node(conflicted_state)
        
        try:
            if not self.initial_split_done:
                # First refinement: use dimension splits
                self.next_id = node.dim_split(
                    agreeing_states, conflicted_state, 
                    self.next_id + 1, self.initial_split_depth)
                self.initial_split_done = True
            else:
                # Subsequent refinements: use SVM
                self.next_id = node.split(
                    agreeing_states, conflicted_state, self.next_id + 1)
                    
            return True
            
        except Exception as e:
            print(f"Refinement failed: {e}")
            return False
            
    def get_num_partitions(self) -> int:
        """Get current number of partitions."""
        return self.next_id
        
    def __str__(self) -> str:
        """String representation."""
        return (f"SVMPartitioner(partitions={self.next_id}, "
                f"initial_depth={self.initial_split_depth})")