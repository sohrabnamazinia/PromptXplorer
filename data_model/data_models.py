"""
Data models for PromptXplorer framework.
"""


class PromptClass:
    """Represents a prompt class with index and description."""
    
    def __init__(self, index: int, description: str = ""):
        self.index = index
        self.description = description
    
    def __repr__(self):
        return f"PromptClass(index={self.index}, description='{self.description}')"


class PrimaryPrompt:
    """Represents a primary prompt."""
    
    def __init__(self, text: str):
        self.text = text
        self.class_obj = None  # PromptClass object
    
    def __repr__(self):
        return f"PrimaryPrompt(text='{self.text[:50]}...', class={self.class_obj})"


class SecondaryPrompt:
    """Represents a secondary (complementary) prompt."""
    
    def __init__(self, text: str):
        self.text = text
        self.class_obj = None  # PromptClass object
    
    def __repr__(self):
        return f"SecondaryPrompt(text='{self.text[:50]}...', class={self.class_obj})"


class CompositePrompt:
    """Represents a composite prompt (primary + ordered secondaries)."""

    def __init__(
        self,
        primary: PrimaryPrompt,
        secondaries: list,
        label: str = None,
        statement_id: str = None,
    ):
        self.primary = primary
        self.secondaries = secondaries  # ordered list of SecondaryPrompt objects
        self.label = label  # e.g. LIAR veracity label
        self.statement_id = statement_id  # e.g. Politifact statement id
    
    def get_composite_class(self):
        """Returns ordered list of classes: [primary.class, secondary1.class, ...]"""
        classes = [self.primary.class_obj]
        classes.extend([sec.class_obj for sec in self.secondaries])
        return classes
    
    def __repr__(self):
        return f"CompositePrompt(primary='{self.primary.text[:30]}...', {len(self.secondaries)} secondaries)"


class PromptManager:
    """Manages prompts, classes, and support matrix."""
    
    def __init__(self):
        self.composite_prompts = []  # list of CompositePrompt objects
        self.composite_classes = None  # 2D matrix
        self.primary_to_secondary_support = None  # dict: (primary_class, secondary_class) -> support
        self.secondary_to_secondary_support = None  # dict: (secondary_class1, secondary_class2) -> support
        self.k_class_sequences = None  # list of k selected composite class sequences
        self.final_composite_prompts = None  # list of k final composite prompt strings
    
    def get_all_primary_prompts(self):
        """Returns list of all PrimaryPrompt objects."""
        return [cp.primary for cp in self.composite_prompts]
    
    def get_all_composite_prompts(self):
        """Returns list of all CompositePrompt objects."""
        return self.composite_prompts
    
    def get_composite_classes(self):
        """Computes and sets composite_classes matrix."""
        if not self.composite_prompts:
            return None
        
        classes_matrix = []
        for cp in self.composite_prompts:
            classes_matrix.append(cp.get_composite_class())
        
        self.composite_classes = classes_matrix
        return self.composite_classes
    
    def save(self, filename_prefix: str = None, algorithm: str = None, csv_filename: str = None):
        """Saves PromptManager to prompt_manager_objects/{timestamp_based_name}/ folder."""
        import os
        import csv
        from datetime import datetime
        
        # Generate unique timestamp-based folder name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        parts = []
        
        if filename_prefix:
            parts.append(filename_prefix)
        if csv_filename:
            # Extract just the filename without extension
            csv_name = os.path.splitext(os.path.basename(csv_filename))[0]
            parts.append(csv_name)
        if algorithm:
            parts.append(algorithm)
        
        parts.append(timestamp)
        folder_name = "_".join(parts) if parts else f"pm_{timestamp}"
        
        save_dir = f"prompt_manager_objects/{folder_name}"
        os.makedirs(save_dir, exist_ok=True)
        
        # Save prompts.csv
        prompts_path = f"{save_dir}/prompts.csv"
        with open(prompts_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # Write header
            max_secondaries = max(len(cp.secondaries) for cp in self.composite_prompts) if self.composite_prompts else 0
            header = ['primary']
            header.extend([f'secondary{i+1}' for i in range(max_secondaries)])
            header.append('primary_class')
            header.extend([f'secondary{i+1}_class' for i in range(max_secondaries)])
            writer.writerow(header)
            
            # Write data
            for cp in self.composite_prompts:
                row = [cp.primary.text]
                row.extend([sec.text for sec in cp.secondaries])
                # Pad with empty strings if needed
                while len(row) < max_secondaries + 1:
                    row.append('')
                
                # Add classes
                row.append(cp.primary.class_obj.index if cp.primary.class_obj else '')
                row.extend([sec.class_obj.index if sec.class_obj else '' for sec in cp.secondaries])
                # Pad class columns
                while len(row) < 2 * (max_secondaries + 1) + 1:
                    row.append('')
                
                writer.writerow(row)
        
        # Save primary_classes.csv
        primary_classes_path = f"{save_dir}/primary_classes.csv"
        primary_classes_set = set()
        for cp in self.composite_prompts:
            if cp.primary.class_obj:
                primary_classes_set.add(cp.primary.class_obj)
        
        with open(primary_classes_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['index', 'description'])
            for cls in sorted(primary_classes_set, key=lambda x: x.index):
                writer.writerow([cls.index, cls.description])
        
        # Save secondary_classes.csv
        secondary_classes_path = f"{save_dir}/secondary_classes.csv"
        secondary_classes_set = set()
        for cp in self.composite_prompts:
            for sec in cp.secondaries:
                if sec.class_obj:
                    secondary_classes_set.add(sec.class_obj)
        
        with open(secondary_classes_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['index', 'description'])
            for cls in sorted(secondary_classes_set, key=lambda x: x.index):
                writer.writerow([cls.index, cls.description])
        
        # Save primary_to_secondary_support.csv
        primary_to_secondary_path = f"{save_dir}/primary_to_secondary_support.csv"
        with open(primary_to_secondary_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['primary_class', 'secondary_class', 'support'])
            
            if self.primary_to_secondary_support is not None:
                for (class1, class2), support in self.primary_to_secondary_support.items():
                    writer.writerow([class1, class2, support])
        
        # Save secondary_to_secondary_support.csv
        secondary_to_secondary_path = f"{save_dir}/secondary_to_secondary_support.csv"
        with open(secondary_to_secondary_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['secondary_class1', 'secondary_class2', 'support'])
            
            if self.secondary_to_secondary_support is not None:
                for (class1, class2), support in self.secondary_to_secondary_support.items():
                    writer.writerow([class1, class2, support])
        
        return folder_name  # Return the folder name for reference
    
    @staticmethod
    def load(folder_name: str):
        """Loads PromptManager from prompt_manager_objects/{folder_name}/ folder."""
        import os
        import csv
        
        load_dir = f"prompt_manager_objects/{folder_name}"
        
        pm = PromptManager()
        
        # Load primary_classes.csv and secondary_classes.csv to build class mapping
        class_map = {}  # index -> PromptClass
        
        primary_classes_path = f"{load_dir}/primary_classes.csv"
        if os.path.exists(primary_classes_path):
            with open(primary_classes_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    idx = int(row['index'])
                    desc = row['description']
                    class_map[idx] = PromptClass(idx, desc)
        
        secondary_classes_path = f"{load_dir}/secondary_classes.csv"
        if os.path.exists(secondary_classes_path):
            with open(secondary_classes_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    idx = int(row['index'])
                    desc = row['description']
                    class_map[idx] = PromptClass(idx, desc)
        
        # Load prompts.csv
        prompts_path = f"{load_dir}/prompts.csv"
        if os.path.exists(prompts_path):
            with open(prompts_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    primary_text = row['primary']
                    primary = PrimaryPrompt(primary_text)
                    
                    # Get primary class
                    primary_class_idx = row.get('primary_class', '')
                    if primary_class_idx and primary_class_idx != '':
                        primary.class_obj = class_map.get(int(primary_class_idx))
                    
                    # Get secondaries
                    secondaries = []
                    i = 1
                    while f'secondary{i}' in row:
                        sec_text = row[f'secondary{i}']
                        if sec_text and sec_text.strip():
                            sec = SecondaryPrompt(sec_text)
                            sec_class_idx = row.get(f'secondary{i}_class', '')
                            if sec_class_idx and sec_class_idx != '':
                                sec.class_obj = class_map.get(int(sec_class_idx))
                            secondaries.append(sec)
                        i += 1
                    
                    cp = CompositePrompt(primary, secondaries)
                    pm.composite_prompts.append(cp)
        
        # Load primary_to_secondary_support.csv
        primary_to_secondary_path = f"{load_dir}/primary_to_secondary_support.csv"
        if os.path.exists(primary_to_secondary_path):
            support_dict = {}
            with open(primary_to_secondary_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    class1 = int(row['primary_class'])
                    class2 = int(row['secondary_class'])
                    support = int(row['support'])
                    support_dict[(class1, class2)] = support
            pm.primary_to_secondary_support = support_dict
        
        # Load secondary_to_secondary_support.csv
        secondary_to_secondary_path = f"{load_dir}/secondary_to_secondary_support.csv"
        if os.path.exists(secondary_to_secondary_path):
            support_dict = {}
            with open(secondary_to_secondary_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    class1 = int(row['secondary_class1'])
                    class2 = int(row['secondary_class2'])
                    support = int(row['support'])
                    support_dict[(class1, class2)] = support
            pm.secondary_to_secondary_support = support_dict
        
        return pm
