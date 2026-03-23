# PromptXplorer Implementation Plan

## Overview

This repository implements the PromptXplorer framework, which constructs ordered sequences of composite prompts that maximize relevance, coverage, and diversity. The framework processes prompts as **sequences** (order matters), not sets.

## Stepwise Implementation Plan

### Phase 1: Data Loading & Preprocessing

**1.1. Data Models (`data_model/data_models.py`)**
- `PromptClass` class:
  - Attributes: `index` (int), `description` (str)
- `PrimaryPrompt` class:
  - Attributes: `text` (str), `class` (PromptClass, initially None)
- `SecondaryPrompt` class:
  - Attributes: `text` (str), `class` (PromptClass, initially None)
- `CompositePrompt` class:
  - Attributes: `primary` (PrimaryPrompt), `secondaries` (ordered list of SecondaryPrompt objects)
  - Method: `get_composite_class()` → returns ordered list [primary.class, secondary1.class, secondary2.class, ...]
- `PromptManager` class:
  - Attributes: 
    - `composite_prompts` (list of CompositePrompt objects)
    - `composite_classes` (2D matrix, initially None) - same dimensions as loaded data, each cell contains class instead of text
    - `support_matrix` (2D matrix, initially None) - support values for composite classes
  - Methods:
    - `get_all_primary_prompts()` → returns list of all PrimaryPrompt objects
    - `get_all_composite_prompts()` → returns list of all CompositePrompt objects
    - `get_composite_classes()` → computes and sets `composite_classes` matrix by calling `get_composite_class()` on each CompositePrompt, returns the matrix
    - `save(filename_prefix: str)` → saves PromptManager to `prompt_manager_objects/{filename_prefix}/` subfolder:
      - CSV file 1: `prompts.csv` - each row = one prompt, columns: primary, secondary1, secondary2, ..., primary_class, secondary1_class, secondary2_class, ...
      - CSV file 2: `classes.csv` - each row = class index and description
      - CSV file 3: `support_matrix.csv` - first two columns = combination of possible pairs of class indices (directed way), third column = support value
    - `load(filename_prefix: str) → PromptManager` → loads PromptManager from `prompt_manager_objects/{filename_prefix}/` subfolder using the three CSV files

**1.2. Data Loader (`data_model/load_data.py`)**
- `DataLoader` class with parameters:
  - `separated` (bool): whether data is already separated
  - `n` (int): number of rows to consider from CSV
  - Method: `load_data(csv_path: str, batch_size: int = None) → PromptManager`
    - **If `separated=True`**:
      - Reads CSV file (first `n` rows) where each row is one prompt
      - First column is primary prompt, remaining columns (comma-separated) are satellite/secondary prompts
      - Parses into primary + ordered secondary sequences (order matters)
      - Handles edge cases (empty secondaries, malformed data)
    - **If `separated=False`**:
      - Reads CSV file (first `n` rows) with single column containing full prompts
      - Uses LLM decomposition to split each prompt into primary + satellite prompts
      - Processes data in batches (max possible length) to LLM
      - Returns decomposed data in same format as separated=True
    - Returns PromptManager object containing all composite prompts

**1.3. LLM Interface (`llm/llm_interface.py`)**
- `LLMInterface` class (will be extended with more functions later)
- Method: `decompose_prompts(prompts: List[str], batch_size: int) → List[Dict]`
  - Takes list of full prompts
  - Processes in batches (respecting max token length)
  - Calls LLM to decompose each prompt into primary + satellite prompts
  - Returns list of dictionaries with 'primary' and 'secondaries' keys
  - Handles batching logic and LLM API calls
- Method: `generate_class_description(prompts: List[str], max_tokens: int = None) → str`
  - Takes list of prompts belonging to a class (below threshold to avoid max length)
  - Calls LLM to generate a short description (few words) for the class
  - Returns description string

---

### Phase 2: Clustering Module

**2.1. Clustering Class (`preprocessing/clusterer.py`)**
- `Clustering` class with parameters:
  - `prompt_manager` (PromptManager object): the prompt manager to cluster
  - `algorithm` (str): clustering algorithm to use (e.g., 'kmeans', 'dbscan', etc.)
- Clustering algorithm methods (each as a separate method):
  - `kmeans_clustering(texts, n_clusters)` → returns cluster labels (descriptions generated via LLM)
  - `dbscan_clustering(texts, eps, min_samples)` → returns cluster labels (descriptions generated via LLM)
  - `hdbscan_clustering(texts, min_cluster_size, min_samples)` → returns cluster labels (descriptions generated via LLM)
  - Additional algorithms can be added as methods
- Main workflow method: `cluster(algorithm_params: dict) → PromptManager`
  - Vectorize prompts (TF-IDF or embeddings)
  - Cluster primary prompts separately using selected algorithm
  - Cluster secondary prompts separately using selected algorithm
  - For each cluster, generate class description using LLM (given prompts below threshold to avoid max length)
  - Assigns cluster labels and descriptions to prompts:
    - Creates `PromptClass` objects with index and description (description from LLM)
    - Sets `class` attribute for all PrimaryPrompt and SecondaryPrompt objects
  - Call `compute_support()` and `build_support_matrix()` to populate support matrix
  - Returns updated PromptManager object (ready to be saved)
- `compute_support() → dict`
  - Computes support for each composite class
  - Returns dictionary where:
    - Key: ordered tuple of classes (e.g., (primary_class, secondary1_class, secondary2_class, ...))
    - Value: support count for that composite class sequence
- `build_support_matrix()`
  - Takes the output dictionary from `compute_support()`
  - Builds support matrix from the computed support values
  - Sets `support_matrix` attribute of PromptManager object
- **Note**: Clustering is a preprocessing step that can be done without knowing the user query. After clustering, call `PromptManager.save()` to persist everything (prompts, classes, support matrix) in a subfolder within `prompt_manager_objects/`.

---

### Phase 3: Algorithms (`algorithms/` folder)

**3.1. Sequence Construction (`algorithms/sequence_construction.py`)**
- Three alternatives; choose via `--sequence_algorithm` in the runner.
- **`RandomWalk`**: LLM selects primary class; then support-weighted sampling over consecutive secondary classes to generate `large_k` class sequences.
- **`WalkWithPartner`**: Same as RandomWalk for primary choice and for transitions, but if the current node’s **total outgoing support** is in the bottom `llm_usage_percent` of all nodes of that type (primary vs secondary), the next secondary class is chosen by LLM instead of sampling. Control with `--walk_partner_llm_percent` (0 = never LLM on transitions, 100 = always; default 25). Run: `--sequence_algorithm walk_with_partner`.
- **`IPF` (Iterative Proportional Fitting)**: Fits a distribution over all length-φ permutations of secondary classes to match observed constraint marginals from the data (degree-2 = consecutive pairs; degree-3 = consecutive triples when `--ipf_degree 3`). After convergence, returns top-`large_k` sequences by probability. Same-class pairs are excluded (outcomes are permutations). Run with IPF: `--sequence_algorithm ipf`; optional: `--ipf_degree 2` (or 3).

**3.2. Representative Selection (`algorithms/representative_selection.py`)**
- `KSetCoverage` class:
  - Compute coverage (distinct complementary classes) for a set of sequences
  - Track which classes are covered by which sequences
  - Greedy algorithm to select k sequences maximizing coverage
  - Sampling-based variant: at each iteration, sample k candidate sequences and select best
  - **Note**: Applied to composite class sequences (not prompt instances) to reduce from many sequences to k before instance selection

**3.3. Prompt Selector (`algorithms/prompt_selector.py`)**
- `IndividualPromptSelector` class:
  - Takes PromptManager with k selected composite class sequences
  - Uses RAG to select actual prompt instances for each sequence
  - For each composite class sequence, incrementally builds the prompt:
    - Starts with user input (primary prompt)
    - For each secondary class in sequence, uses RAG to select best secondary prompt instance
    - RAG retrieves top-L similar prompts and uses LLM to select one
  - **Second LLM Integration**: LLM selects best instances based on current prompt context
  - Generate final composite prompt sequences (primary + ordered complementary instances)
  - **Note**: Applied after k-set coverage to reduce LLM costs by only selecting instances for k sequences
- `SampledGreedySelector`: sample a subset per class, pick nearest neighbor to the current prompt (cosine); no LLM.
- `NaiveSelector` (naive batched tournament): uses **all** prompts in the class pool. Splits into batches (`max_batch_size`), picks one winner per batch (LLM or mock), then runs further rounds on winners until one remains. **Mock mode** (`mock_llm=True`, default): no LLM; each batch winner is chosen uniformly at random from the top `mock_top_fraction` (default 0.15) by cosine similarity to the current prompt. Runner: `--prompt_selector naive`, plus `--naive_batch_size`, `--naive_mock_llm`, `--naive_top_fraction`. Optional `embed_fn` in code for tests/offline embeddings.

**Preprocessing Components:**
- `preprocessing/embedding.py`: `Embedding` class that computes and stores embeddings for all secondary prompts in `embeddings_db/secondary_embeddings.csv`
- `llm/rag.py`: `RAG` class that uses embeddings and LLM to select next prompt to add

**3.4. Sequence Ordering (`algorithms/sequence_ordering.py`)**
- `OrderSequence` class:
  - Compute order of priority (weights) for complementary classes given primary class
  - Compute pairwise diversity vectors between sequences
  - Compute weighted Hamming distance between sequences
  - Greedy ordering algorithm: start with most diverse sequence, iteratively select most diverse from previously selected
  - Maximize overall diversity vector

---

### Phase 4: Main Pipeline (`promptxplorer.py`)

**4.1. PromptXplorer Class**
- Main end-to-end framework class
- Input parameters:
  - `prompt_manager` (PromptManager object): the prompt repository (preprocessed with clustering)
  - `primary_prompt` (str): user's new input primary prompt
  - `length` (int): desired composite prompt length
  - `k` (int): number of composite prompts to return
  - Additional configuration parameters (algorithm choices, thresholds, etc.)
- Method: `run() → List[CompositePrompt]`
  - Orchestrates all algorithm phases:
    1. Sequence construction (IPF or RandomWalk)
    2. Prompt selection (IndividualPromptSelector)
    3. Representative selection (KSetCoverage)
    4. Sequence ordering (OrderSequence)
  - Returns ordered sequence of k composite prompts

---

## Proposed File Structure

```
PromptXplorer-/
├── README.md
├── requirements.txt
├── config.py                    # Configuration parameters
├── data_model/
│   ├── data_models.py          # Phase 1.1: Data structures (PrimaryPrompt, SecondaryPrompt, CompositePrompt, PromptManager)
│   └── load_data.py            # Phase 1.2: Data loading (DataLoader class + load_data function)
├── preprocessing/
│   ├── clusterer.py            # Phase 2: Clustering (Clustering class with multiple algorithms, support computation)
│   └── embedding.py            # Embedding computation for secondary prompts
├── embeddings_db/             # Stored embeddings (CSV file)
├── prompt_manager_objects/     # Saved/loaded PromptManager objects (subfolders with CSV files)
├── algorithms/
│   ├── sequence_construction.py  # Phase 3.1: RandomWalk, WalkWithPartner, IPF
│   ├── prompt_selector.py        # Phase 3.2: IndividualPromptSelector class (RAG + LLM)
│   ├── representative_selection.py # Phase 3.3: KSetCoverage class
│   └── sequence_ordering.py      # Phase 3.4: OrderSequence class
├── llm/
│   ├── llm_interface.py        # LLM integration (OpenAI/other) - Phase 1.3: decompose_prompts() + future functions
│   ├── rag.py                  # RAG class for prompt selection
│   └── prompts.py              # LLM prompt templates
├── promptxplorer.py            # Phase 4: PromptXplorer main class (end-to-end framework)
└── utils/
    ├── logger.py
    └── metrics.py
```

## Key Design Principles

1. **Modularity**: Each phase is a separate module
2. **Configurability**: All parameters in `config.py`
3. **Sequence-aware**: All data structures preserve order
4. **LLM Integration**: Two clear integration points
5. **Extensibility**: Easy to swap algorithms (e.g., different clustering, embeddings)

## Algorithmic Pipeline Summary

1. **Cluster satellite prompts** (complementary prompts)
2. **Create sequences of clusters (classes)** (choose one):
   - **IPF**: Fit distribution to degree-2 (and optionally degree-3) pair/triple marginals; return top-k sequences by probability.
   - **Random Walk**: Assign class to primary (LLM) → support-weighted random walk → get sequences.
   - **WalkWithPartner**: Like random walk, but LLM picks the next secondary when the current node is in the lowest-X% by outgoing support (`--walk_partner_llm_percent`).
     - Gelman-Rubin for convergence
     - **First LLM use**: If confidence below threshold
3. **Convert cluster sequences → individual prompt sequences** (each sequence independently)
   - RAG-based approach
   - **Second LLM use**
4. **k-set coverage** (e.g., 100 sequences → 10):
   - Maximize coverage of distinct classes
   - **Improved**: Sampling combined k-set coverage
5. **Order sequences**:
   - Weighted Hamming distance between prompts
   - Greedy: most diverse from all → most diverse from first → most diverse from previous → ...

---

## Remaining Tasks

1. **Fix DBSCAN and HDBSCAN clustering algorithms**: Currently implemented but may need debugging/refinement
2. **Implement confidence-based LLM helper in Random Walk**: Track confidence values for transitions (primary→secondary and secondary→secondary). When confidence is below a threshold, use LLM to help decide which node to go to next, rather than relying solely on support values
3. **Implement stochastic coverage in k-set coverage**: In each iteration of the greedy algorithm, instead of considering all possible sequences, sample p of them and select the best from the sampled set. This improves efficiency for large sets of sequences