"""
Clustering module for PromptXplorer framework.
"""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans, DBSCAN
try:
    from hdbscan import HDBSCAN
    HDBSCAN_AVAILABLE = True
except ImportError:
    HDBSCAN_AVAILABLE = False

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_model.data_models import PromptClass, PromptManager


class Clustering:
    """Clusters prompts into classes."""
    
    def __init__(self, prompt_manager: PromptManager, algorithm: str = 'kmeans'):
        """
        Args:
            prompt_manager: PromptManager object to cluster
            algorithm: Clustering algorithm ('kmeans', 'dbscan', 'hdbscan')
        """
        self.prompt_manager = prompt_manager
        self.algorithm = algorithm
    
    def cluster(self, algorithm_params: dict):
        """
        Main clustering workflow.
        
        Args:
            algorithm_params: Dictionary of parameters for the selected algorithm
        
        Returns:
            Updated PromptManager object
        """
        # Cluster primary prompts
        primary_prompts = self.prompt_manager.get_all_primary_prompts()
        primary_texts = [p.text for p in primary_prompts]
        primary_labels, primary_descriptions = self._cluster_texts(
            primary_texts, 
            algorithm_params.get('primary', {})
        )
        
        # Cluster secondary prompts
        all_secondaries = []
        for cp in self.prompt_manager.composite_prompts:
            all_secondaries.extend([sec.text for sec in cp.secondaries])
        
        secondary_labels, secondary_descriptions = self._cluster_texts(
            all_secondaries,
            algorithm_params.get('secondary', {})
        )
        
        # Assign classes to prompts
        self._assign_classes(
            primary_labels, primary_descriptions,
            secondary_labels, secondary_descriptions
        )
        
        # Compute support and build support matrices
        primary_to_secondary, secondary_to_secondary = self.compute_support()
        self.build_support_matrix(primary_to_secondary, secondary_to_secondary)
        
        return self.prompt_manager
    
    def _cluster_texts(self, texts: list, params: dict):
        """Clusters texts using the selected algorithm."""
        if not texts:
            return [], {}
        
        # Vectorize
        vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        X = vectorizer.fit_transform(texts)
        
        # Cluster
        if self.algorithm == 'kmeans':
            labels = self.kmeans_clustering(texts, params.get('n_clusters', 10))
        elif self.algorithm == 'dbscan':
            labels = self.dbscan_clustering(texts, params.get('eps', 0.5), params.get('min_samples', 5))
        elif self.algorithm == 'hdbscan':
            if not HDBSCAN_AVAILABLE:
                raise ImportError("HDBSCAN not available. Install with: pip install hdbscan")
            labels = self.hdbscan_clustering(
                texts, 
                params.get('min_cluster_size', 5), 
                params.get('min_samples', 3)
            )
        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")
        
        # Generate descriptions (LLM by default; can be disabled for offline/benchmark runs)
        use_llm_descriptions = True
        if isinstance(params, dict) and "use_llm_descriptions" in params:
            use_llm_descriptions = bool(params.get("use_llm_descriptions"))
        descriptions = self._generate_class_descriptions(
            texts, labels, use_llm_descriptions=use_llm_descriptions
        )
        
        return labels, descriptions
    
    def kmeans_clustering(self, texts: list, n_clusters: int = 10):
        """KMeans clustering."""
        vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        X = vectorizer.fit_transform(texts)
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)
        
        return labels.tolist()
    
    def dbscan_clustering(self, texts: list, eps: float = 0.5, min_samples: int = 5):
        """DBSCAN clustering."""
        vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        X = vectorizer.fit_transform(texts)
        
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels = dbscan.fit_predict(X)
        
        return labels.tolist()
    
    def hdbscan_clustering(self, texts: list, min_cluster_size: int = 5, min_samples: int = 3):
        """HDBSCAN clustering."""
        if not HDBSCAN_AVAILABLE:
            raise ImportError("HDBSCAN not available. Install with: pip install hdbscan")
        
        vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        X = vectorizer.fit_transform(texts)
        
        hdbscan = HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples)
        labels = hdbscan.fit_predict(X)
        
        return labels.tolist()
    
    def _generate_class_descriptions(
        self, texts: list, labels: list, use_llm_descriptions: bool = True
    ):
        """Generates class descriptions (LLM by default; fallback is deterministic placeholders)."""
        if not use_llm_descriptions:
            return {int(lbl): f"cluster_{int(lbl)}" for lbl in set(labels) if int(lbl) >= 0}

        import sys
        import os
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        try:
            from llm.llm_interface import LLMInterface
        except Exception:
            return {int(lbl): f"cluster_{int(lbl)}" for lbl in set(labels) if int(lbl) >= 0}

        try:
            llm_interface = LLMInterface()
        except Exception:
            return {int(lbl): f"cluster_{int(lbl)}" for lbl in set(labels) if int(lbl) >= 0}
        descriptions = {}
        
        # Group texts by label
        label_to_texts = {}
        for text, label in zip(texts, labels):
            if label not in label_to_texts:
                label_to_texts[label] = []
            label_to_texts[label].append(text)
        
        # Generate description for each class
        for label, class_texts in label_to_texts.items():
            # Use subset of texts to avoid max length
            max_texts = min(10, len(class_texts))
            subset_texts = class_texts[:max_texts]
            description = llm_interface.generate_class_description(subset_texts)
            descriptions[label] = description
        
        return descriptions
    
    def _assign_classes(self, primary_labels, primary_descriptions, 
                       secondary_labels, secondary_descriptions):
        """Assigns PromptClass objects to prompts."""
        # Create PromptClass objects for primaries
        primary_classes = {}
        for label, desc in primary_descriptions.items():
            # Handle negative labels (noise in DBSCAN/HDBSCAN)
            if label >= 0:
                primary_classes[label] = PromptClass(label, desc)
        
        # Create PromptClass objects for secondaries
        secondary_classes = {}
        for label, desc in secondary_descriptions.items():
            # Handle negative labels (noise in DBSCAN/HDBSCAN)
            if label >= 0:
                secondary_classes[label] = PromptClass(label, desc)
        
        # Assign to primary prompts
        primary_prompts = self.prompt_manager.get_all_primary_prompts()
        for i, prompt in enumerate(primary_prompts):
            if i < len(primary_labels):
                label = primary_labels[i]
                if label in primary_classes:
                    prompt.class_obj = primary_classes[label]
        
        # Assign to secondary prompts
        sec_idx = 0
        for cp in self.prompt_manager.composite_prompts:
            for sec in cp.secondaries:
                if sec_idx < len(secondary_labels):
                    label = secondary_labels[sec_idx]
                    if label in secondary_classes:
                        sec.class_obj = secondary_classes[label]
                    sec_idx += 1
    
    def _compute_primary_to_secondary_support(self):
        """
        Computes support from primary class to secondary class.
        
        Returns:
            Dictionary where key is (primary_class_index, secondary_class_index), value is support count
        """
        support_dict = {}
        
        for cp in self.prompt_manager.composite_prompts:
            primary_class = cp.primary.class_obj
            if primary_class and cp.secondaries:
                first_secondary = cp.secondaries[0]
                if first_secondary.class_obj:
                    pair = (primary_class.index, first_secondary.class_obj.index)
                    support_dict[pair] = support_dict.get(pair, 0) + 1
        
        return support_dict
    
    def _compute_secondary_to_secondary_support(self):
        """
        Computes support from secondary class to secondary class.
        
        Returns:
            Dictionary where key is (secondary_class1_index, secondary_class2_index), value is support count
        """
        support_dict = {}
        
        for cp in self.prompt_manager.composite_prompts:
            # For each consecutive pair of secondaries
            for i in range(len(cp.secondaries) - 1):
                sec1 = cp.secondaries[i]
                sec2 = cp.secondaries[i + 1]
                if sec1.class_obj and sec2.class_obj:
                    pair = (sec1.class_obj.index, sec2.class_obj.index)
                    support_dict[pair] = support_dict.get(pair, 0) + 1
        
        return support_dict
    
    def compute_support(self):
        """
        Computes support for both primary-to-secondary and secondary-to-secondary.
        
        Returns:
            Tuple of (primary_to_secondary_dict, secondary_to_secondary_dict)
        """
        primary_to_secondary = self._compute_primary_to_secondary_support()
        secondary_to_secondary = self._compute_secondary_to_secondary_support()
        return primary_to_secondary, secondary_to_secondary
    
    def build_support_matrix(self, primary_to_secondary_dict: dict, secondary_to_secondary_dict: dict):
        """
        Builds support matrices from computed support values.
        
        Args:
            primary_to_secondary_dict: Dictionary from _compute_primary_to_secondary_support()
            secondary_to_secondary_dict: Dictionary from _compute_secondary_to_secondary_support()
        """
        self.prompt_manager.primary_to_secondary_support = primary_to_secondary_dict
        self.prompt_manager.secondary_to_secondary_support = secondary_to_secondary_dict
