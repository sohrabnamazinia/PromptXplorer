"""
Run with the IDE Run button.

BruteForceSelector with mock_llm + embed_fn: no OpenAI calls.
Uses fixed embeddings so the global winner is predictable.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from algorithms.prompt_selector import BruteForceSelector
from data_model.data_models import PromptManager


class FakeRag:
    """Minimal stand-in for RAG (only embeddings_db + top_l)."""

    def __init__(self, embeddings_db):
        self.embeddings_db = embeddings_db
        self.top_l = 5


def main():
    # Query direction e_q; candidates in R^4
    e_q = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    v_best = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    v_mid = np.array([0.5, 0.5, 0.0, 0.0], dtype=float)
    v_low = np.array([0.0, 1.0, 0.0, 0.0], dtype=float)
    v_low2 = np.array([0.0, 0.0, 1.0, 0.0], dtype=float)

    db = [
        {"class_label": 0, "text": "WINNER", "embedding": v_best},
        {"class_label": 0, "text": "mid", "embedding": v_mid},
        {"class_label": 0, "text": "low_a", "embedding": v_low},
        {"class_label": 0, "text": "low_b", "embedding": v_low2},
    ]
    rag = FakeRag(db)

    pm = PromptManager()
    pm.k_class_sequences = [[0, 0]]  # one sequence: primary 0, one secondary slot class 0

    def embed_fn(_text: str) -> np.ndarray:
        return e_q

    bf = BruteForceSelector(
        pm,
        rag,
        max_batch_size=2,
        mock_llm=True,
        mock_top_fraction=0.25,
        seed=123,
        embed_fn=embed_fn,
    )

    # Tournament: batch1 WINNER+mid -> top 25% of 2 => 1 => WINNER
    #             batch2 low_a+low_b -> one winner (low_a vs low_b, higher cos with e_q is low_a? dot 0 vs 0 tie - sort stable first)
    # Final: WINNER vs low_a -> WINNER
    out = bf.select_prompts("user query", phi=1)
    assert len(out) == 1
    assert "WINNER" in out[0], out
    print("OK: tournament picked highest-relevance prompt:", out[0])

    # Top-25%% bucket: 4 items -> k = ceil(1) = 1 per batch; single batch of 4 would pick top 1 only
    bf2 = BruteForceSelector(
        pm,
        rag,
        max_batch_size=10,
        mock_llm=True,
        mock_top_fraction=0.25,
        seed=999,
        embed_fn=embed_fn,
    )
    out2 = bf2.select_prompts("user query", phi=1)
    assert "WINNER" in out2[0], out2
    print("OK: single batch (all in class):", out2[0])

    print("\nAll brute-force selector checks passed.")


if __name__ == "__main__":
    main()
