"""
Sequence construction algorithms for PromptXplorer.
"""

import itertools
import random
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_model.data_models import PromptManager
from llm.llm_interface import LLMInterface


def _select_primary_class_for_sequence(prompt_manager: PromptManager, llm_interface: LLMInterface, user_input: str):
    """
    Uses LLM to choose primary class from user input. Shared by RandomWalk and IPF.
    Returns primary class index or None.
    """
    primary_prompts = prompt_manager.get_all_primary_prompts()
    primary_classes_info = {}
    for prompt in primary_prompts:
        if prompt.class_obj:
            class_idx = prompt.class_obj.index
            if class_idx not in primary_classes_info:
                primary_classes_info[class_idx] = {
                    'description': prompt.class_obj.description,
                    'samples': []
                }
            primary_classes_info[class_idx]['samples'].append(prompt.text)
    classes_text = []
    for class_idx, info in sorted(primary_classes_info.items()):
        samples = info['samples'][:3]
        samples_text = "\n".join([f"  - {s}" for s in samples])
        classes_text.append(f"Class {class_idx}: {info['description']}\nSamples:\n{samples_text}")
    classes_context = "\n\n".join(classes_text)
    class_idx = llm_interface.select_primary_class(user_input, classes_context)
    if class_idx is not None and class_idx in primary_classes_info:
        return class_idx
    if primary_classes_info:
        return min(primary_classes_info.keys())
    return None


class RandomWalk:
    """Random walk algorithm for generating composite class sequences."""
    
    def __init__(self, prompt_manager: PromptManager):
        """
        Args:
            prompt_manager: PromptManager object with clustering results and support matrices
        """
        self.prompt_manager = prompt_manager
        self.llm_interface = LLMInterface()
    
    def walk(self, user_input: str, phi: int):
        """
        Main method to generate a composite class sequence.
        
        Args:
            user_input: User's input primary prompt
            phi: Number of secondary classes to generate
        
        Returns:
            List of class indices: [primary_class, secondary1_class, ..., secondary_phi_class]
        """
        # Step 1: Choose primary class
        primary_class = _select_primary_class_for_sequence(self.prompt_manager, self.llm_interface, user_input)
        
        # Step 2: Choose first secondary class from primary
        secondary_classes = []
        current_primary = primary_class
        
        # Step 3: Choose first secondary
        first_secondary = self.walk_primary_secondary(current_primary)
        secondary_classes.append(first_secondary)
        
        # Step 4: Choose remaining secondaries (phi - 1 more)
        current_secondary = first_secondary
        for _ in range(phi - 1):
            next_secondary = self.walk_secondary_secondary(current_secondary)
            secondary_classes.append(next_secondary)
            current_secondary = next_secondary
        
        return [primary_class] + secondary_classes
    
    def random_walk_iter(self, user_input: str, phi: int, large_k: int):
        """
        Runs random walk multiple times to generate multiple composite class sequences.
        
        Args:
            user_input: User's input primary prompt
            phi: Number of secondary classes to generate per sequence
            large_k: Number of times to run random walk
        
        Returns:
            List of composite class sequences, each is [primary_class, secondary1_class, ..., secondary_phi_class]
        """
        sequences = []
        for _ in range(large_k):
            sequence = self.walk(user_input, phi)
            sequences.append(sequence)
        return sequences
    
    def walk_primary_secondary(self, primary_class: int):
        """
        Chooses first secondary class from primary class using support values.
        Considers ALL secondary classes, using 0.1 for missing edges.
        
        Args:
            primary_class: Primary class index
        
        Returns:
            Secondary class index
        """
        # Get all secondary classes
        all_secondary_classes = set()
        for cp in self.prompt_manager.composite_prompts:
            for sec in cp.secondaries:
                if sec.class_obj:
                    all_secondary_classes.add(sec.class_obj.index)
        
        if not all_secondary_classes:
            return None
        
        # Build support values for all secondary classes
        # If edge exists, use support value; if not, use 0.1
        candidates = {}
        for sec_class in all_secondary_classes:
            if self.prompt_manager.primary_to_secondary_support:
                support = self.prompt_manager.primary_to_secondary_support.get((primary_class, sec_class), 0.1)
            else:
                support = 0.1
            candidates[sec_class] = support
        
        # Non-uniform sampling based on support values
        secondary_classes = list(candidates.keys())
        weights = [candidates[c] for c in secondary_classes]
        total_weight = sum(weights)
        probabilities = [w / total_weight for w in weights]
        
        chosen = np.random.choice(secondary_classes, p=probabilities)
        return chosen
    
    def walk_secondary_secondary(self, secondary_class: int):
        """
        Chooses next secondary class from current secondary class using support values.
        Considers ALL secondary classes, using 0.1 for missing edges.
        
        Args:
            secondary_class: Current secondary class index
        
        Returns:
            Next secondary class index
        """
        # Get all secondary classes
        all_secondary_classes = set()
        for cp in self.prompt_manager.composite_prompts:
            for sec in cp.secondaries:
                if sec.class_obj:
                    all_secondary_classes.add(sec.class_obj.index)
        
        if not all_secondary_classes:
            return None
        
        # Build support values for all secondary classes
        # If edge exists, use support value; if not, use 0.1
        candidates = {}
        for sec_class in all_secondary_classes:
            if self.prompt_manager.secondary_to_secondary_support:
                support = self.prompt_manager.secondary_to_secondary_support.get((secondary_class, sec_class), 0.1)
            else:
                support = 0.1
            candidates[sec_class] = support
        
        # Non-uniform sampling based on support values
        next_secondary_classes = list(candidates.keys())
        weights = [candidates[c] for c in next_secondary_classes]
        total_weight = sum(weights)
        probabilities = [w / total_weight for w in weights]
        
        chosen = np.random.choice(next_secondary_classes, p=probabilities)
        return chosen


class IPF:
    """
    Iterative Proportional Fitting (IPF) for generating composite class sequences.
    Builds pairwise (and higher-order) constraint targets from the data, then
    iteratively rescales outcome probabilities so constraints are satisfied.
    Returns top-large_k sequences by estimated probability.
    """

    def __init__(
        self,
        prompt_manager: PromptManager,
        degree: int = 2,
        max_iter: int = 50000,
        tol: float = 1e-6,
        use_degree1: bool = False,
    ):
        """
        Args:
            prompt_manager: PromptManager with clustering and support matrices.
            degree: Maximum constraint order (1=singleton counts, 2=consecutive pairs, 3=triples, ...). Default 2.
            max_iter: Maximum IPF iterations. Default 400 (often needed to reach tol with degree-2 constraints).
            tol: Convergence tolerance (max probability change). Default 1e-4 (use 1e-6 for stricter).
            use_degree1: If True, add degree-1 constraints (P(class appears)). Default False: use only
                degree-2 (and 3) so constraints are consistent with a distribution over sequences,
                avoiding limit cycles and non-convergence.
        """
        self.prompt_manager = prompt_manager
        self.degree = max(1, min(degree, 10))
        self.max_iter = int(max_iter)
        self.tol = tol
        self.use_degree1 = use_degree1
        self.llm_interface = LLMInterface()

    def _get_secondary_classes(self):
        """Set of secondary (satellite) class indices."""
        s = set()
        for cp in self.prompt_manager.composite_prompts:
            for sec in cp.secondaries:
                if sec.class_obj:
                    s.add(sec.class_obj.index)
        return sorted(s)

    def _build_outcomes(self, secondary_classes: list, phi: int):
        """All ordered sequences of length phi over secondary_classes (no repetition)."""
        if phi <= 0 or phi > len(secondary_classes):
            return []
        return [tuple(p) for p in itertools.permutations(secondary_classes, phi)]

    def _compute_constraints(self, secondary_classes: list, phi: int):
        """
        Build constraint list: each (target_prob, predicate).
        predicate(outcome) True iff outcome satisfies that constraint.
        Returns (constraints, n_degree1, n_degree2, n_degree3) for logging.
        """
        constraints = []
        n_degree1 = n_degree2 = n_degree3 = 0

        # Degree 1: P(class c appears in sequence) from marginal counts.
        # Optional: when combined with degree-2 and renormalization, these can be inconsistent
        # with a distribution over full sequences, causing limit cycles. Default off.
        if self.use_degree1:
            count_c = {}
            total_1 = 0
            for cp in self.prompt_manager.composite_prompts:
                for sec in cp.secondaries:
                    if sec.class_obj:
                        c = sec.class_obj.index
                        count_c[c] = count_c.get(c, 0) + 1
                        total_1 += 1
            if total_1 > 0:
                for c in secondary_classes:
                    target = count_c.get(c, 0) / total_1
                    constraints.append((target, lambda o, c=c: c in o))
                    n_degree1 += 1

        # Degree 2: P(c1 immediately followed by c2) from secondary_to_secondary_support.
        # Only include pairs with c1 != c2: outcomes are permutations (no repetition),
        # so same-class pairs (e.g. (0,0)) have no outcome and would get actual=0.
        if self.degree >= 2 and phi >= 2 and getattr(
            self.prompt_manager, "secondary_to_secondary_support", None
        ):
            sup = self.prompt_manager.secondary_to_secondary_support
            valid_pairs = [(c1, c2) for (c1, c2) in sup if c1 != c2]
            total_2 = sum(sup[(c1, c2)] for (c1, c2) in valid_pairs)
            if total_2 > 0:
                for (c1, c2) in valid_pairs:
                    cnt = sup[(c1, c2)]
                    target = cnt / total_2
                    constraints.append(
                        (
                            target,
                            lambda o, c1=c1, c2=c2: any(
                                o[i] == c1 and o[i + 1] == c2 for i in range(len(o) - 1)
                            ),
                        )
                    )
                    n_degree2 += 1

        # Degree 3: P(c1,c2,c3 consecutive) from consecutive triples in data.
        # Only include triples with c1, c2, c3 all distinct (outcomes are permutations).
        if self.degree >= 3 and phi >= 3:
            triple_count = {}
            for cp in self.prompt_manager.composite_prompts:
                secs = [s.class_obj.index for s in cp.secondaries if s.class_obj]
                for i in range(len(secs) - 2):
                    t = (secs[i], secs[i + 1], secs[i + 2])
                    triple_count[t] = triple_count.get(t, 0) + 1
            valid_triples = [t for t in triple_count if len(set(t)) == 3]
            total_3 = sum(triple_count[t] for t in valid_triples)
            if total_3 > 0:
                for (c1, c2, c3) in valid_triples:
                    cnt = triple_count[(c1, c2, c3)]
                    target = cnt / total_3
                    constraints.append(
                        (
                            target,
                            lambda o, c1=c1, c2=c2, c3=c3: any(
                                o[i] == c1 and o[i + 1] == c2 and o[i + 2] == c3
                                for i in range(len(o) - 2)
                            ),
                        )
                    )
                    n_degree3 += 1

        return constraints, n_degree1, n_degree2, n_degree3

    def _ipf_iterate(self, outcomes: list, constraints: list):
        """
        Run IPF: start uniform d[outcome], then repeatedly rescale so each
        constraint's total probability over satisfying outcomes matches target.
        Returns (d, num_iterations_used).
        """
        n_out = len(outcomes)
        if n_out == 0:
            return {}, 0
        d = {o: 1.0 / n_out for o in outcomes}
        num_iters = 0
        for iteration in range(self.max_iter):
            num_iters = iteration + 1
            max_diff = 0.0
            for target, predicate in constraints:
                satisfying = [o for o in outcomes if predicate(o)]
                if not satisfying:
                    continue
                current = sum(d[o] for o in satisfying)
                if current <= 0:
                    continue
                factor = target / current
                for o in satisfying:
                    new_val = d[o] * factor
                    max_diff = max(max_diff, abs(new_val - d[o]))
                    d[o] = new_val
            # Renormalize so d stays a proper distribution (sum=1). Without this,
            # overlapping constraints drain probability and sum(d) drifts away from 1.
            total_prob = sum(d.values())
            if total_prob > 0:
                for o in outcomes:
                    d[o] /= total_prob
            # Log only first 3 iters, then every 25th, then final (avoid 200 lines)
            if num_iters <= 3 or num_iters % 2500 == 0 or max_diff < self.tol:
                print(f"  IPF iter {num_iters}: max_change={max_diff:.2e}, sum(d)={sum(d.values()):.6f}")
            if max_diff < self.tol:
                print(f"  IPF converged at iteration {num_iters} (tol={self.tol}).")
                break
        else:
            print(f"  IPF stopped at max_iter={self.max_iter} (max_change={max_diff:.2e}, tol={self.tol}).")
        return d, num_iters

    def run(self, user_input: str, phi: int, large_k: int):
        """
        Run IPF and return top-large_k composite class sequences.

        Args:
            user_input: User's primary prompt (used to select primary class via LLM).
            phi: Number of secondary classes per sequence.
            large_k: Number of sequences to return.

        Returns:
            List of [primary_class, secondary_1, ..., secondary_phi] (same format as RandomWalk).
        """
        print("  IPF setup: selecting primary class via LLM...")
        primary_class = _select_primary_class_for_sequence(
            self.prompt_manager, self.llm_interface, user_input
        )
        print(f"  IPF primary class selected: {primary_class}")
        secondary_classes = self._get_secondary_classes()
        if not secondary_classes:
            print("  IPF: no secondary classes, returning []")
            return []
        outcomes = self._build_outcomes(secondary_classes, phi)
        if not outcomes:
            print("  IPF: no outcomes (phi > n_secondary?), returning []")
            return []
        constraints, n_d1, n_d2, n_d3 = self._compute_constraints(secondary_classes, phi)
        print(f"  IPF setup: secondary_classes={secondary_classes}, |outcomes|={len(outcomes)}, "
              f"constraints: degree1={n_d1}, degree2={n_d2}, degree3={n_d3} (total={len(constraints)})")
        print("  IPF iterations:")
        d, num_iters = self._ipf_iterate(outcomes, constraints)
        # Top outcomes by probability
        sorted_outcomes = sorted(outcomes, key=lambda o: d.get(o, 0.0), reverse=True)
        print(f"  IPF top-5 outcomes (sequence -> probability):")
        for i, o in enumerate(sorted_outcomes[:5], 1):
            print(f"    {i}. {list(o)} -> d={d.get(o, 0):.6f}")
        # Quick constraint check: first 3 constraints target vs actual
        print("  IPF constraint check (first 3): target vs actual")
        for idx, (target, predicate) in enumerate(constraints[:3]):
            actual = sum(d[o] for o in outcomes if predicate(o))
            print(f"    constraint {idx + 1}: target={target:.4f}, actual={actual:.4f}, diff={abs(actual - target):.2e}")
        top = sorted_outcomes[: large_k]
        return [[primary_class] + list(o) for o in top]
