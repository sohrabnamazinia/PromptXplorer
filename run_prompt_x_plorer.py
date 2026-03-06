"""
End-to-end PromptXplorer framework runner.
"""

import argparse
import os
import sys
import time
from datetime import datetime
from data_model.load_data import DataLoader
from preprocessing.clusterer import Clustering
from preprocessing.embedding import Embedding
from algorithms.sequence_construction import RandomWalk, IPF
from algorithms.k_set_coverage import KSetCoverage
from algorithms.prompt_selector import IndividualPromptSelector
from algorithms.sequence_ordering import OrderSequence
from llm.rag import RAG
from llm.llm_interface import LLMInterface


class TeeOutput:
    """Write to both file and stdout."""
    def __init__(self, file):
        self.file = file
        self.stdout = __import__('sys').stdout
    
    def write(self, text):
        self.file.write(text)
        self.file.flush()
        self.stdout.write(text)
        self.stdout.flush()
    
    def flush(self):
        self.file.flush()
        self.stdout.flush()


def main():
    # Small-scale run (quick test):
    #   python run_prompt_x_plorer.py --n 50 --n_clusters_primary 3 --n_clusters_secondary 5 --phi 2 --large_k 10 --small_k 3 --top_l 3
    # With IPF: add --sequence_algorithm ipf (optional: --ipf_degree 2).
    parser = argparse.ArgumentParser(description="Run PromptXplorer end-to-end")
    
    # Dataset parameters
    parser.add_argument("--dataset_path", type=str, default="data/diffusion_db.csv",
                        help="Path to dataset CSV file")
    parser.add_argument("--n", type=int, default=None,
                        help="Number of prompts from dataset to consider (None = all)")
    parser.add_argument("--separated", type=lambda x: x.lower() == 'true', default=True,
                        help="Whether the dataset is already separated (primary, secondary_1, ...). Default: True")
    
    # Clustering parameters
    parser.add_argument("--clustering_algorithm", type=str, default="kmeans",
                        choices=["kmeans", "dbscan", "hdbscan"],
                        help="Clustering algorithm to use")
    parser.add_argument("--n_clusters_primary", type=int, default=5,
                        help="Number of clusters for primary prompts")
    parser.add_argument("--n_clusters_secondary", type=int, default=10,
                        help="Number of clusters for secondary prompts")
    
    # Sequence construction parameters
    parser.add_argument("--sequence_algorithm", type=str, default="random_walk",
                        choices=["random_walk", "ipf"],
                        help="Sequence construction algorithm: random_walk or ipf. Default: random_walk")
    parser.add_argument("--ipf_degree", type=int, default=2,
                        help="IPF constraint degree (1=singletons, 2=pairs, 3=triples, ...). Used only if sequence_algorithm=ipf. Default: 2")
    parser.add_argument("--user_input", type=str, default="Create a portrait of a famous person",
                        help="User input query")
    parser.add_argument("--phi", type=int, default=4,
                        help="Number of secondary classes in each sequence")
    parser.add_argument("--large_k", type=int, default=20,
                        help="Number of sequences to generate (random walk or IPF top-k)")
    parser.add_argument("--small_k", type=int, default=5,
                        help="Number of sequences to select in k-set coverage")
    
    # RAG parameters
    parser.add_argument("--top_l", type=int, default=5,
                        help="Number of top candidates for RAG retrieval")
    
    # Output parameters
    parser.add_argument("--save_prompt_manager", type=lambda x: x.lower() == 'true', default=True,
                        help="Whether to save the PromptManager object. Default: True")
    parser.add_argument("--output_prefix", type=str, default="run",
                        help="Prefix for saved PromptManager folder")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs("prompt_xplorer_outputs", exist_ok=True)
    
    # Create log file with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"prompt_xplorer_outputs/run_{timestamp}.txt"
    log_file = open(log_filename, 'w', encoding='utf-8')
    
    # Create TeeOutput to write to both file and stdout
    tee = TeeOutput(log_file)
    
    # Save original stdout and redirect to tee
    original_stdout = sys.stdout
    sys.stdout = tee
    
    # Start total execution timer
    total_start_time = time.time()
    execution_times = {}
    
    try:
        # Write parameters at the beginning
        tee.write("=" * 80 + "\n")
        tee.write("PROMPTXPLORER RUN - PARAMETERS\n")
        tee.write("=" * 80 + "\n")
        tee.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        tee.write(f"Dataset path: {args.dataset_path}\n")
        tee.write(f"Number of prompts (n): {args.n}\n")
        tee.write(f"Separated: {args.separated}\n")
        tee.write(f"Clustering algorithm: {args.clustering_algorithm}\n")
        tee.write(f"Primary clusters: {args.n_clusters_primary}\n")
        tee.write(f"Secondary clusters: {args.n_clusters_secondary}\n")
        tee.write(f"Sequence algorithm: {args.sequence_algorithm}\n")
        tee.write(f"IPF degree: {args.ipf_degree}\n")
        tee.write(f"User input: {args.user_input}\n")
        tee.write(f"Phi (secondary classes per sequence): {args.phi}\n")
        tee.write(f"Large K (sequences): {args.large_k}\n")
        tee.write(f"Small K (selected sequences): {args.small_k}\n")
        tee.write(f"Top L (RAG candidates): {args.top_l}\n")
        tee.write(f"Save PromptManager: {args.save_prompt_manager}\n")
        tee.write(f"Output prefix: {args.output_prefix}\n")
        tee.write("=" * 80 + "\n\n")
    
        # Phase 1: Load data
        phase_start = time.time()
        print("=" * 80)
        print("Phase 1: Loading data...")
        print("=" * 80)
        loader = DataLoader(separated=args.separated, n=args.n)
        pm = loader.load_data(args.dataset_path)
        print(f"Loaded {len(pm.composite_prompts)} composite prompts")
        execution_times['Phase 1: Load data'] = time.time() - phase_start
        
        # Phase 2: Cluster
        phase_start = time.time()
        print("\n" + "=" * 80)
        print("Phase 2: Clustering...")
        print("=" * 80)
        clusterer = Clustering(pm, algorithm=args.clustering_algorithm)
        algorithm_params = {
            'primary': {'n_clusters': args.n_clusters_primary},
            'secondary': {'n_clusters': args.n_clusters_secondary}
        }
        pm = clusterer.cluster(algorithm_params)
        print("Clustering completed")
        execution_times['Phase 2: Clustering'] = time.time() - phase_start
        
        # Save PromptManager (optional)
        if args.save_prompt_manager:
            phase_start = time.time()
            print("\nSaving PromptManager...")
            folder_name = pm.save(args.output_prefix, algorithm=args.clustering_algorithm, csv_filename=args.dataset_path)
            print(f"Saved to prompt_manager_objects/{folder_name}/")
            execution_times['Save PromptManager'] = time.time() - phase_start
        
        # Embedding (needed for RAG)
        phase_start = time.time()
        print("\n" + "=" * 80)
        print("Embedding secondary prompts...")
        print("=" * 80)
        embedding = Embedding(pm)
        embedding.embed()
        execution_times['Embedding'] = time.time() - phase_start
        
        # Phase 3.1: Sequence construction (Random Walk or IPF)
        phase_start = time.time()
        phase_name = "IPF" if args.sequence_algorithm == "ipf" else "Random Walk"
        print("\n" + "=" * 80)
        print(f"Phase 3.1: Sequence construction ({phase_name})...")
        print("=" * 80)
        if args.sequence_algorithm == "ipf":
            ipf = IPF(pm, degree=args.ipf_degree)
            composite_class_sequences = ipf.run(args.user_input, args.phi, args.large_k)
        else:
            random_walk = RandomWalk(pm)
            composite_class_sequences = random_walk.random_walk_iter(args.user_input, args.phi, args.large_k)
        print(f"Generated {len(composite_class_sequences)} composite class sequences:")
        for i, sequence in enumerate(composite_class_sequences, 1):
            print(f"  Sequence {i}: Primary={sequence[0]}, Secondaries={sequence[1:]}")
        execution_times[f'Phase 3.1: {phase_name}'] = time.time() - phase_start
        
        # Phase 3.2: K-Set Coverage
        phase_start = time.time()
        print("\n" + "=" * 80)
        print("Phase 3.2: K-Set Coverage...")
        print("=" * 80)
        k_set_coverage = KSetCoverage(pm, composite_class_sequences)
        selected_sequences = k_set_coverage.run_greedy_coverage(args.small_k)
        print(f"Selected {len(selected_sequences)} sequences:")
        for i, sequence in enumerate(selected_sequences, 1):
            print(f"  Sequence {i}: Primary={sequence[0]}, Secondaries={sequence[1:]}")
        execution_times['Phase 3.2: K-Set Coverage'] = time.time() - phase_start
        
        # Phase 3.3: Prompt Selector
        phase_start = time.time()
        print("\n" + "=" * 80)
        print("Phase 3.3: Prompt Selector...")
        print("=" * 80)
        llm_interface = LLMInterface()
        rag = RAG(embedding, llm_interface, top_l=args.top_l)
        prompt_selector = IndividualPromptSelector(pm, rag)
        prompt_selector.select_prompts(args.user_input, args.phi)
        execution_times['Phase 3.3: Prompt Selector'] = time.time() - phase_start
        
        # Read final prompts from PromptManager
        final_prompts = pm.final_composite_prompts
        print(f"\n✓ Generated {len(final_prompts)} composite prompts (before ordering):")
        for i, prompt in enumerate(final_prompts, 1):
            print(f"\n  Prompt {i}:")
            print(f"  {prompt}")
        
        # Phase 3.4: Sequence Ordering
        phase_start = time.time()
        print("\n" + "=" * 80)
        print("Phase 3.4: Sequence Ordering...")
        print("=" * 80)
        order_sequence = OrderSequence(pm)
        ordered_prompts = order_sequence.order_sequences()
        execution_times['Phase 3.4: Sequence Ordering'] = time.time() - phase_start
        
        # Calculate total execution time
        total_execution_time = time.time() - total_start_time
        
        # Output Metrics Section
        print("\n" + "=" * 80)
        print("PROMPTXPLORER OUTPUT - PARAMETERS")
        print("=" * 80)
        print("\nExecution Time Metrics:")
        print("-" * 80)
        for phase, exec_time in execution_times.items():
            print(f"  {phase}: {exec_time:.2f} seconds ({exec_time/60:.2f} minutes)")
        print("-" * 80)
        print(f"  Total Execution Time: {total_execution_time:.2f} seconds ({total_execution_time/60:.2f} minutes)")
        print("=" * 80)
        
        # Final Results
        print("\n" + "=" * 80)
        print("FINAL RESULTS")
        print("=" * 80)
        print(f"✓ Final ordered {len(ordered_prompts)} composite prompts:")
        for i, prompt in enumerate(ordered_prompts, 1):
            print(f"\n  Prompt {i}:")
            print(f"  {prompt}")
        
        print("\n" + "=" * 80)
        print("✓ End-to-end run completed successfully!")
        print(f"✓ Log file saved to: {log_filename}")
        print("=" * 80)
    
    finally:
        # Restore original stdout
        sys.stdout = original_stdout
        # Close log file
        log_file.close()


if __name__ == "__main__":
    main()
