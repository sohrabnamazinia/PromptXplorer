"""
Run this file with the IDE Run button (no pytest needed).

Builds a tiny synthetic PromptManager and compares RandomWalk vs WalkWithPartner on
transition steps only — no LLM calls (WalkWithPartner uses llm_usage_percent=0).
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from algorithms.sequence_construction import RandomWalk, WalkWithPartner
from data_model.data_models import (
    CompositePrompt,
    PrimaryPrompt,
    PromptClass,
    PromptManager,
    SecondaryPrompt,
)


def make_sample_prompt_manager():
    """Two primaries (0 = high outgoing support, 1 = low) and three secondaries."""
    pm = PromptManager()

    pc0 = PromptClass(0, "primary A")
    pc1 = PromptClass(1, "primary B")
    s0 = PromptClass(0, "sec style")
    s1 = PromptClass(1, "sec lighting")
    s2 = PromptClass(2, "sec mood")

    p0 = PrimaryPrompt("p0")
    p0.class_obj = pc0
    p1 = PrimaryPrompt("p1")
    p1.class_obj = pc1

    def sec(text, cls):
        sp = SecondaryPrompt(text)
        sp.class_obj = cls
        return sp

    pm.composite_prompts = [
        CompositePrompt(p0, [sec("a0", s0), sec("a1", s1)]),
        CompositePrompt(p1, [sec("b0", s0), sec("b2", s2)]),
    ]
    pm.primary_to_secondary_support = {
        (0, 0): 100,
        (0, 1): 0,
        (0, 2): 0,
        (1, 0): 5,
        (1, 1): 5,
        (1, 2): 0,
    }
    pm.secondary_to_secondary_support = {
        (0, 0): 50,
        (0, 1): 50,
        (0, 2): 10,
        (1, 0): 50,
        (1, 1): 50,
        (1, 2): 10,
        (2, 0): 50,
        (2, 1): 50,
        (2, 2): 50,
    }
    return pm


def main():
    pm = make_sample_prompt_manager()

    # Outgoing totals (what WalkWithPartner uses for the LLM vs sample decision)
    p0_out = sum(pm.primary_to_secondary_support.get((0, s), 0) for s in (0, 1, 2))
    p1_out = sum(pm.primary_to_secondary_support.get((1, s), 0) for s in (0, 1, 2))
    print("Sample PM: primary 0 total outgoing support =", p0_out)
    print("             primary 1 total outgoing support =", p1_out)
    assert p0_out == 100 and p1_out == 10

    rw = RandomWalk(pm)
    wwp = WalkWithPartner(pm, llm_usage_percent=0, rng=np.random.default_rng(42))

    print("\n--- Same transition logic (0% LLM on WalkWithPartner = support sample only) ---")
    print("Fixed seed 42 for both where possible.\n")

    np.random.seed(42)
    a_rw = rw.walk_primary_secondary(0)
    a_wwp = wwp.walk_primary_secondary("demo user text", 0)
    print(f"From primary 0 → first secondary: RandomWalk={a_rw}, WalkWithPartner={a_wwp}")

    np.random.seed(42)
    b_rw = rw.walk_secondary_secondary(a_rw)
    b_wwp = wwp.walk_secondary_secondary("demo user text", a_wwp)
    print(f"One more step:                  RandomWalk={b_rw}, WalkWithPartner={b_wwp}")

    for x in (a_rw, a_wwp, b_rw, b_wwp):
        assert x in (0, 1, 2)

    wwp_b = WalkWithPartner(pm, llm_usage_percent=0, rng=np.random.default_rng(42))
    wwp_c = WalkWithPartner(pm, llm_usage_percent=0, rng=np.random.default_rng(42))
    assert wwp_b.walk_primary_secondary("u", 0) == wwp_c.walk_primary_secondary("u", 0)

    print("\n--- When would WalkWithPartner ask the LLM? (no API call here) ---")
    wwp50 = WalkWithPartner(pm, llm_usage_percent=50.0, rng=np.random.default_rng(0))
    thresh_p = float(np.percentile(wwp50._primary_totals_list, 50.0))
    print(f"Primary nodes outgoing totals: {wwp50._primary_totals_list}")
    print(f"50th percentile threshold: {thresh_p:.2f}")
    print(f"Primary 0 outgoing {wwp50._outgoing_primary(0)} → use LLM? {wwp50._use_llm_for_primary_node(0)}")
    print(f"Primary 1 outgoing {wwp50._outgoing_primary(1)} → use LLM? {wwp50._use_llm_for_primary_node(1)}")
    assert wwp50._use_llm_for_primary_node(1) is True
    assert wwp50._use_llm_for_primary_node(0) is False

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
