"""
Data loader for PromptXplorer framework.
"""

import csv
from .data_models import PromptManager, CompositePrompt, PrimaryPrompt, SecondaryPrompt


class DataLoader:
    """Loads and parses prompt data from CSV files."""

    # LIAR-style CSV: metadata columns (not treated as secondary prompts)
    _METADATA_COLUMN_NAMES = frozenset({"label", "statement_id"})

    def __init__(self, separated: bool = True, n: int = None):
        """
        Args:
            separated: If True, CSV has primary and secondaries already separated.
                      If False, CSV has single column with full prompts (needs LLM decomposition).
            n: Number of rows to consider from CSV (None = all rows)
        """
        self.separated = separated
        self.n = n
        self._named_column_map = None  # dict lower_name -> index, set when header has primary+label
    
    def load_data(self, csv_path: str, batch_size: int = None):
        """
        Loads data from CSV file and returns PromptManager.
        
        Args:
            csv_path: Path to CSV file
            batch_size: Batch size for LLM processing (if separated=False)
        
        Returns:
            PromptManager object
        """
        pm = PromptManager()
        
        if self.separated:
            pm = self._load_separated(csv_path)
        else:
            import sys
            import os
            sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            from llm.llm_interface import LLMInterface
            llm_interface = LLMInterface()
            pm = self._load_with_decomposition(csv_path, llm_interface, batch_size)
        
        return pm
    
    def _load_separated(self, csv_path: str):
        """Loads CSV where primary and secondaries are already separated."""
        pm = PromptManager()
        self._named_column_map = None

        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)

            first_row = next(reader, None)
            if not first_row:
                return pm

            fr0 = first_row[0].strip().lower()
            if fr0 == "prompt":
                start_with_named = False
            elif fr0 == "primary" and any(
                c.strip().lower() == "label" for c in first_row
            ):
                self._named_column_map = {
                    first_row[i].strip().lower(): i for i in range(len(first_row))
                }
                start_with_named = True
            else:
                start_with_named = False

            if start_with_named:
                count = 0
                for row in reader:
                    if self.n is not None and count >= self.n:
                        break
                    self._process_row_named(row, pm)
                    count += 1
                return pm

            # Legacy: first column primary, all other non-empty cells are secondaries
            if fr0 != "prompt":
                if first_row:
                    self._process_row_legacy(first_row, pm)

            count = 0
            for row in reader:
                if self.n is not None and count >= self.n:
                    break
                self._process_row_legacy(row, pm)
                count += 1

        return pm

    def _process_row_named(self, row: list, pm: PromptManager):
        """Row from liar.csv-style header: primary, secondary_*, label (optional statement_id)."""
        h = self._named_column_map
        if not row or "primary" not in h:
            return
        pi = h["primary"]
        if pi >= len(row) or not row[pi].strip():
            return

        primary_text = row[pi].strip()
        primary = PrimaryPrompt(primary_text)
        secondaries = []

        sec_keys = sorted(
            (k for k in h if k.startswith("secondary")),
            key=lambda k: h[k],
        )
        for name in sec_keys:
            idx = h[name]
            if idx >= len(row):
                continue
            sec_text = row[idx].strip()
            if sec_text:
                secondaries.append(SecondaryPrompt(sec_text))

        label = None
        statement_id = None
        if "label" in h and h["label"] < len(row):
            label = row[h["label"]].strip() or None
        if "statement_id" in h and h["statement_id"] < len(row):
            statement_id = row[h["statement_id"]].strip() or None

        cp = CompositePrompt(primary, secondaries, label=label, statement_id=statement_id)
        pm.composite_prompts.append(cp)

    def _process_row_legacy(self, row: list, pm: PromptManager):
        """First column is primary; every other non-empty column is a secondary."""
        if not row or not row[0].strip():
            return

        primary_text = row[0].strip()
        primary = PrimaryPrompt(primary_text)

        secondaries = []
        for i in range(1, len(row)):
            sec_text = row[i].strip()
            if sec_text:
                secondaries.append(SecondaryPrompt(sec_text))

        cp = CompositePrompt(primary, secondaries)
        pm.composite_prompts.append(cp)

    def _process_row(self, row: list, pm: PromptManager):
        """Backward-compatible alias."""
        self._process_row_legacy(row, pm)
    
    def _load_with_decomposition(self, csv_path: str, llm_interface, batch_size: int):
        """Loads CSV with single column and uses LLM to decompose prompts."""
        pm = PromptManager()
        
        # Read all prompts - each line is one complete prompt
        prompts = []
        with open(csv_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
            # Skip header if it's "prompt"
            start_idx = 0
            if lines and lines[0].strip().lower() == 'prompt':
                start_idx = 1
            
            count = 0
            for line in lines[start_idx:]:
                if self.n and count >= self.n:
                    break
                prompt = line.strip()
                if prompt:  # Skip empty lines
                    prompts.append(prompt)
                count += 1
        
        # Decompose using LLM
        decomposed = llm_interface.decompose_prompts(prompts, batch_size)
        
        # Create CompositePrompts from decomposed data
        for item in decomposed:
            primary = PrimaryPrompt(item['primary'])
            secondaries = [SecondaryPrompt(sec) for sec in item['secondaries']]
            cp = CompositePrompt(primary, secondaries)
            pm.composite_prompts.append(cp)
        
        return pm
