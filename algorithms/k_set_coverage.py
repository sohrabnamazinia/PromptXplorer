"""
K-set coverage algorithm for selecting representative sequences.
"""

import sys
import os
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_model.data_models import PromptManager


class KSetCoverage:
    """Greedy k-set coverage algorithm."""
    
    def __init__(self, prompt_manager: PromptManager, k_large_class_sequences: list):
        """
        Args:
            prompt_manager: PromptManager object
            k_large_class_sequences: List of composite class sequences (each is [primary_class, secondary1, ...])
        """
        self.prompt_manager = prompt_manager
        self.k_large_class_sequences = k_large_class_sequences
    
    def run_greedy_coverage(self, k: int):
        """
        Runs greedy k-set coverage algorithm.
        
        Args:
            k: Target number of sequences to select
        
        Returns:
            List of k selected composite class sequences
        """
        if k >= len(self.k_large_class_sequences):
            # If k is greater than or equal to all sequences, return all
            self.prompt_manager.k_class_sequences = self.k_large_class_sequences
            return self.k_large_class_sequences
        
        # Initialize
        I = []  # Selected sequences
        U = set()  # Uncovered classes (all classes from all sequences)
        
        # Build set of all classes
        for sequence in self.k_large_class_sequences:
            U.update(sequence)
        
        # Greedy selection
        for _ in range(k):
            best_sequence = None
            best_coverage_gain = -1
            
            # Find sequence with maximum coverage gain
            for sequence in self.k_large_class_sequences:
                if sequence in I:
                    continue  # Skip already selected sequences
                
                # Calculate coverage gain (new classes this sequence adds)
                sequence_classes = set(sequence)
                covered_classes = set()
                for selected_seq in I:
                    covered_classes.update(selected_seq)
                
                new_classes = sequence_classes - covered_classes
                coverage_gain = len(new_classes)
                
                if coverage_gain > best_coverage_gain:
                    best_coverage_gain = coverage_gain
                    best_sequence = sequence
            
            # Add best sequence to selection
            if best_sequence:
                I.append(best_sequence)
            else:
                # If no sequence adds new coverage, just pick any remaining
                remaining = [seq for seq in self.k_large_class_sequences if seq not in I]
                if remaining:
                    I.append(remaining[0])
                else:
                    break
        
        # Store result in prompt manager
        self.prompt_manager.k_class_sequences = I
        return I

    def run_stochastic_coverage(self, k: int, sample_size: int, rng=None):
        """
        Greedy k-set coverage with stochastic candidate pool: at each step, sample
        ``sample_size`` sequences uniformly from remaining candidates and pick the
        one with maximum coverage gain.

        Args:
            k: Target number of sequences to select
            sample_size: Number of candidates to sample each iteration (>= 1)
            rng: numpy Generator or None for default_rng()

        Returns:
            List of k selected composite class sequences
        """
        if rng is None:
            rng = np.random.default_rng()
        sample_size = max(1, int(sample_size))

        if k >= len(self.k_large_class_sequences):
            self.prompt_manager.k_class_sequences = list(self.k_large_class_sequences)
            return self.prompt_manager.k_class_sequences

        I = []

        def coverage_gain(sequence):
            sequence_classes = set(sequence)
            covered = set()
            for s in I:
                covered.update(s)
            return len(sequence_classes - covered)

        for _ in range(k):
            candidates = [s for s in self.k_large_class_sequences if s not in I]
            if not candidates:
                break
            n_sample = min(sample_size, len(candidates))
            idx = rng.choice(len(candidates), size=n_sample, replace=False)
            pool = [candidates[i] for i in idx]
            best_sequence = max(pool, key=coverage_gain)
            if coverage_gain(best_sequence) == 0 and len(I) < k:
                best_sequence = candidates[rng.integers(len(candidates))]
            I.append(best_sequence)

        self.prompt_manager.k_class_sequences = I
        return I
