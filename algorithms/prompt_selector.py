"""
Prompt Selector algorithm for selecting actual prompt instances.
"""

import sys
import os
from typing import Optional, List, Any

import numpy as np
from openai import OpenAI

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_model.data_models import PromptManager
from llm.rag import RAG
from llm.llm_interface import LLMInterface


class IndividualPromptSelector:
    """Selects actual prompt instances for composite class sequences."""
    
    def __init__(self, prompt_manager: PromptManager, rag: RAG):
        """
        Args:
            prompt_manager: PromptManager object with k_class_sequences
            rag: RAG object for selecting secondary prompts
        """
        self.prompt_manager = prompt_manager
        self.rag = rag
        self.llm_interface = LLMInterface()
        self.completed_prompts = []  # Store previously completed prompts
    
    def _select_secondary_prompt_with_context(self, current_prompt: str, secondary_class_index: int):
        """Select secondary prompt considering previously completed prompts."""
        # Get candidates from RAG (top-L similar prompts)
        # But we need to filter by class - RAG doesn't do this, so we'll need to modify
        
        # For now, get candidates from embeddings that match the class
        candidates = []
        if self.rag.embeddings_db:
            for item in self.rag.embeddings_db:
                if item['class_label'] == secondary_class_index:
                    candidates.append(item['text'])
        
        if not candidates:
            return None
        
        # Get top-L candidates by similarity to current prompt
        if len(candidates) > self.rag.top_l:
            api_key = os.getenv('OPENAI_API_KEY')
            client = OpenAI(api_key=api_key)
            response = client.embeddings.create(
                model="text-embedding-3-small",
                input=[current_prompt]
            )
            current_embedding = np.array(response.data[0].embedding)
            
            similarities = []
            for candidate in candidates:
                # Find embedding for this candidate
                for item in self.rag.embeddings_db:
                    if item['text'] == candidate:
                        similarity = np.dot(current_embedding, item['embedding']) / (
                            np.linalg.norm(current_embedding) * np.linalg.norm(item['embedding'])
                        )
                        similarities.append((similarity, candidate))
                        break
            
            similarities.sort(key=lambda x: x[0], reverse=True)
            candidates = [cand for _, cand in similarities[:self.rag.top_l]]
        
        # Call LLM with context of previously completed prompts
        result = self.llm_interface.select_next_prompt_rag(
            current_prompt, 
            candidates, 
            self.completed_prompts
        )
        
        return result
    
    def select_prompts(self, user_input: str, phi: int):
        """
        Select actual prompt instances for k sequences.
        
        Args:
            user_input: Initial user input prompt
            phi: Number of secondary prompts to add
        
        Returns:
            List of k complete composite prompts (strings)
        """
        if not self.prompt_manager.k_class_sequences:
            return []
        
        k_prompts = []
        self.completed_prompts = []  # Reset for each call
        
        for seq_idx, class_sequence in enumerate(self.prompt_manager.k_class_sequences):
            # class_sequence is [primary_class, secondary_class_1, ..., secondary_class_phi]
            secondary_classes = class_sequence[1:phi+1] if len(class_sequence) > 1 else []
            
            # Start with user input
            current_prompt = user_input
            
            # Iteratively add secondary prompts phi times
            for secondary_class in secondary_classes[:phi]:
                result = self._select_secondary_prompt_with_context(current_prompt, secondary_class)
                if result and 'updated_prompt' in result:
                    current_prompt = result['updated_prompt']
                else:
                    # Fallback: just append a candidate if available
                    candidates = [item['text'] for item in self.rag.embeddings_db 
                                if item['class_label'] == secondary_class]
                    if candidates:
                        current_prompt = f"{current_prompt}, {candidates[0]}"
            
            k_prompts.append(current_prompt)
            self.completed_prompts.append(current_prompt)
        
        # Store in PromptManager
        self.prompt_manager.final_composite_prompts = k_prompts
        
        return k_prompts


class SampledGreedySelector:
    """
    Greedy instance selector with sampling.

    For each secondary class in the selected class sequence, sample a subset of the
    candidate prompts for that class, pick the closest (cosine similarity) to the
    current prompt embedding, and append it (comma-separated). No LLM reranking.
    """

    def __init__(
        self,
        prompt_manager: PromptManager,
        rag: RAG,
        sample_size_per_class: int = 20,
        seed: int = 42,
    ):
        self.prompt_manager = prompt_manager
        self.rag = rag
        self.sample_size_per_class = max(1, int(sample_size_per_class))
        self.rng = np.random.default_rng(seed)
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.client = OpenAI(api_key=self.api_key)
        self.model = "text-embedding-3-small"
        self.completed_prompts = []

    def _embed_text(self, text: str) -> np.ndarray:
        resp = self.client.embeddings.create(model=self.model, input=[text])
        return np.array(resp.data[0].embedding, dtype=float)

    @staticmethod
    def _cosine(a: np.ndarray, b: np.ndarray) -> float:
        denom = (np.linalg.norm(a) * np.linalg.norm(b))
        if denom == 0:
            return 0.0
        return float(np.dot(a, b) / denom)

    def _select_for_class(self, current_prompt: str, secondary_class_index: int):
        if not self.rag.embeddings_db:
            return None

        # Candidates restricted to this class
        class_items = [
            item for item in self.rag.embeddings_db if item["class_label"] == secondary_class_index
        ]
        if not class_items:
            return None

        # Avoid exact reuse if possible
        avoid_texts = set()
        avoid_texts.update([p for p in self.completed_prompts])
        # Also avoid already-contained snippets
        contained = set()
        for item in class_items:
            if item["text"] in current_prompt:
                contained.add(item["text"])

        filtered_items = [it for it in class_items if it["text"] not in contained]
        pool = filtered_items if filtered_items else class_items

        # Sample subset
        n = min(self.sample_size_per_class, len(pool))
        idx = self.rng.choice(len(pool), size=n, replace=False)
        sampled = [pool[i] for i in idx]

        current_emb = self._embed_text(current_prompt)

        best_item = None
        best_sim = -1e9
        for it in sampled:
            sim = self._cosine(current_emb, it["embedding"])
            if sim > best_sim:
                best_sim = sim
                best_item = it

        if not best_item:
            return None

        selected_text = best_item["text"]
        updated_prompt = f"{current_prompt}, {selected_text}"
        return {"selected_prompt": selected_text, "updated_prompt": updated_prompt}

    def select_prompts(self, user_input: str, phi: int):
        if not self.prompt_manager.k_class_sequences:
            return []

        k_prompts = []
        self.completed_prompts = []

        for class_sequence in self.prompt_manager.k_class_sequences:
            secondary_classes = class_sequence[1 : phi + 1] if len(class_sequence) > 1 else []
            current_prompt = user_input

            for secondary_class in secondary_classes[:phi]:
                result = self._select_for_class(current_prompt, secondary_class)
                if result and "updated_prompt" in result:
                    current_prompt = result["updated_prompt"]
                else:
                    # Fallback: append first available candidate in class
                    fallback = next(
                        (
                            item["text"]
                            for item in (self.rag.embeddings_db or [])
                            if item["class_label"] == secondary_class
                        ),
                        None,
                    )
                    if fallback:
                        current_prompt = f"{current_prompt}, {fallback}"

            k_prompts.append(current_prompt)
            self.completed_prompts.append(current_prompt)

        self.prompt_manager.final_composite_prompts = k_prompts
        return k_prompts


class BruteForceSelector:
    """
    Considers every prompt in the target class, in batches (for context limits). Each batch
    yields one winner via LLM or via mock mode. Winners are merged in further rounds until
    one prompt remains (same batching + final merge).

    ``mock_llm=True`` (default): no LLM calls; each batch winner is drawn uniformly at random
    from the top ``mock_top_fraction`` of that batch by cosine similarity to the current prompt.
    ``mock_llm=False``: uses ``select_next_prompt_rag`` per batch and for the final merge.

    Pass ``embed_fn`` for tests or offline use; default uses OpenAI embeddings like other selectors.
    """

    def __init__(
        self,
        prompt_manager: PromptManager,
        rag: RAG,
        max_batch_size: int = 15,
        mock_llm: bool = True,
        mock_top_fraction: float = 0.25,
        seed: int = 42,
        embed_fn=None,
    ):
        self.prompt_manager = prompt_manager
        self.rag = rag
        self.max_batch_size = max(1, int(max_batch_size))
        self.mock_llm = bool(mock_llm)
        self.mock_top_fraction = float(mock_top_fraction)
        if not 0 < self.mock_top_fraction <= 1:
            raise ValueError("mock_top_fraction must be in (0, 1]")
        self.rng = np.random.default_rng(seed)
        self.llm_interface = LLMInterface()
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.client = OpenAI(api_key=self.api_key) if self.api_key else None
        self.model = "text-embedding-3-small"
        self.embed_fn = embed_fn  # optional callable(str) -> np.ndarray
        self.completed_prompts = []

    def _embed_text(self, text: str) -> np.ndarray:
        if self.embed_fn is not None:
            return np.asarray(self.embed_fn(text), dtype=float)
        if not self.client:
            raise RuntimeError("OPENAI_API_KEY required for embeddings unless embed_fn is set")
        resp = self.client.embeddings.create(model=self.model, input=[text])
        return np.array(resp.data[0].embedding, dtype=float)

    @staticmethod
    def _cosine(a: np.ndarray, b: np.ndarray) -> float:
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        if denom == 0:
            return 0.0
        return float(np.dot(a, b) / denom)

    def _pick_winner_from_batch(
        self,
        current_prompt: str,
        batch: list,
        current_emb: Optional[np.ndarray],
    ):
        """
        batch: list of embeddings_db items (dict with 'text', 'embedding', ...).
        Returns one item dict.
        """
        if not batch:
            return None
        if len(batch) == 1:
            return batch[0]

        if self.mock_llm:
            if current_emb is None:
                current_emb = self._embed_text(current_prompt)
            scored = [(self._cosine(current_emb, it["embedding"]), it) for it in batch]
            scored.sort(key=lambda x: x[0], reverse=True)
            k = max(1, int(np.ceil(self.mock_top_fraction * len(batch))))
            top_bucket = [it for _, it in scored[:k]]
            return self.rng.choice(top_bucket)

        texts = [it["text"] for it in batch]
        result = self.llm_interface.select_next_prompt_rag(
            current_prompt,
            texts,
            self.completed_prompts,
        )
        if result and result.get("selected_prompt"):
            sel = (result["selected_prompt"] or "").strip()
            for it in batch:
                if it["text"].strip() == sel:
                    return it
            for it in batch:
                if sel in it["text"] or it["text"] in sel:
                    return it
        return batch[0]

    def _tournament_reduce(
        self, current_prompt: str, items: List[Any], current_emb: Optional[np.ndarray]
    ):
        """Reduce a list of embeddings_db items to a single winner."""
        if not items:
            return None
        if len(items) == 1:
            return items[0]

        winners = []
        for i in range(0, len(items), self.max_batch_size):
            batch = items[i : i + self.max_batch_size]
            w = self._pick_winner_from_batch(current_prompt, batch, current_emb)
            if w is not None:
                winners.append(w)

        if not winners:
            return items[0]
        if len(winners) == 1:
            return winners[0]
        return self._tournament_reduce(current_prompt, winners, current_emb)

    def _select_for_class(self, current_prompt: str, secondary_class_index: int):
        if not self.rag.embeddings_db:
            return None

        class_items = [
            item
            for item in self.rag.embeddings_db
            if item["class_label"] == secondary_class_index
        ]
        if not class_items:
            return None

        avoid_texts = set(self.completed_prompts)
        contained = {it["text"] for it in class_items if it["text"] in current_prompt}
        filtered = [it for it in class_items if it["text"] not in contained and it["text"] not in avoid_texts]
        pool = filtered if filtered else class_items

        current_emb = None
        if self.mock_llm:
            current_emb = self._embed_text(current_prompt)

        best_item = self._tournament_reduce(current_prompt, pool, current_emb)
        if not best_item:
            return None

        selected_text = best_item["text"]
        updated_prompt = f"{current_prompt}, {selected_text}"
        return {"selected_prompt": selected_text, "updated_prompt": updated_prompt}

    def select_prompts(self, user_input: str, phi: int):
        if not self.prompt_manager.k_class_sequences:
            return []

        k_prompts = []
        self.completed_prompts = []

        for class_sequence in self.prompt_manager.k_class_sequences:
            secondary_classes = class_sequence[1 : phi + 1] if len(class_sequence) > 1 else []
            current_prompt = user_input

            for secondary_class in secondary_classes[:phi]:
                result = self._select_for_class(current_prompt, secondary_class)
                if result and "updated_prompt" in result:
                    current_prompt = result["updated_prompt"]
                else:
                    fallback = next(
                        (
                            item["text"]
                            for item in (self.rag.embeddings_db or [])
                            if item["class_label"] == secondary_class
                        ),
                        None,
                    )
                    if fallback:
                        current_prompt = f"{current_prompt}, {fallback}"

            k_prompts.append(current_prompt)
            self.completed_prompts.append(current_prompt)

        self.prompt_manager.final_composite_prompts = k_prompts
        return k_prompts
