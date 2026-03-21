"""
LLM interface for PromptXplorer framework.
"""

import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate


class LLMInterface:
    """Interface for LLM operations using OpenAI."""
    
    def __init__(self, api_key: str = None, model: str = "gpt-4o"):
        """
        Args:
            api_key: API key for OpenAI (if None, uses OPENAI_API_KEY environment variable)
            model: Model name to use
        """
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        self.model = model
        self.llm = ChatOpenAI(
            api_key=self.api_key,
            model=model,
            temperature=0
        )
    
    def decompose_prompts(self, prompts: list, batch_size: int = None):
        """
        Decomposes full prompts into primary + satellite prompts using LLM.
        
        Args:
            prompts: List of full prompt strings
            batch_size: Batch size for processing (if None, processes all at once)
        
        Returns:
            List of dictionaries with 'primary' and 'secondaries' keys
        """
        if batch_size is None:
            batch_size = len(prompts)
        
        results = []
        
        # Process in batches
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i:i + batch_size]
            batch_results = self._decompose_batch(batch)
            results.extend(batch_results)
        
        return results
    
    def _decompose_batch(self, prompts: list):
        """Decomposes a batch of prompts."""
        import json
        
        results = []
        
        for prompt in prompts:
            # Create prompt template
            template = ChatPromptTemplate.from_messages([
                ("system", "You are a helpful assistant that decomposes prompts into a primary prompt and secondary prompts."),
                ("user", """Given the following prompt, decompose it into:
1. A primary prompt (the core objective)
2. Secondary prompts (auxiliary details that enrich it)

Return a JSON object with two keys: "primary" and "secondaries".
- "primary": string (the primary prompt)
- "secondaries": array of strings (the secondary prompts)

Example JSON:
{{"primary": "Create a portrait of Elon Musk", "secondaries": ["digital art", "high quality", "trending"]}}

Prompt: {prompt}

Return only the JSON object:""")
            ])
            
            # Invoke LLM
            chain = template | self.llm
            response = chain.invoke({"prompt": prompt})
            
            # Parse JSON response
            try:
                response_text = response.content.strip()
                # Remove markdown code blocks if present
                if response_text.startswith("```"):
                    response_text = response_text.split("```")[1]
                    if response_text.startswith("json"):
                        response_text = response_text[4:]
                response_text = response_text.strip()
                
                decomposed = json.loads(response_text)
                results.append({
                    'primary': decomposed['primary'],
                    'secondaries': decomposed['secondaries']
                })
            except (json.JSONDecodeError, KeyError):
                # Fallback: use original prompt as primary
                results.append({
                    'primary': prompt,
                    'secondaries': []
                })
        
        return results
    
    def generate_class_description(self, prompts: list, max_tokens: int = None):
        """
        Generates a short description for a class given its prompts.
        
        Args:
            prompts: List of prompt strings belonging to the class
            max_tokens: Maximum tokens to use (if None, uses default)
        
        Returns:
            Description string (few words)
        """
        if not prompts:
            return "unknown"
        
        # Limit prompts to avoid exceeding token limits
        max_prompts = min(10, len(prompts))
        sample_prompts = prompts[:max_prompts]
        
        # Create prompt template
        prompts_text = "\n".join([f"- {p}" for p in sample_prompts])
        
        template = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant that generates concise class descriptions."),
            ("user", """Given the following prompts that belong to the same class, generate a short description (2-4 words) that captures the common theme or category.

Prompts:
{prompts}

Provide a concise description (2-4 words only):""")
        ])
        
        # Invoke LLM
        chain = template | self.llm
        response = chain.invoke({"prompts": prompts_text})
        
        description = response.content.strip()
        
        # Clean up description (remove quotes, extra words)
        description = description.strip('"\'')
        if len(description.split()) > 5:
            # If too long, take first few words
            description = " ".join(description.split()[:4])
        
        return description.lower()
    
    def select_primary_class(self, user_input: str, classes_context: str):
        """
        Selects the most appropriate primary class for a user input prompt.
        
        Args:
            user_input: User's input primary prompt
            classes_context: Formatted string with available primary classes, descriptions, and samples
        
        Returns:
            Selected primary class index (int) or None if parsing fails
        """
        template = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant that selects the most appropriate class for a user prompt."),
            ("user", """Given the user input prompt, choose the most appropriate primary class.

User input: {user_input}

Available primary classes:
{classes_context}

Return only the class index (just the number) that best matches the user input:""")
        ])
        
        chain = template | self.llm
        response = chain.invoke({"user_input": user_input, "classes_context": classes_context})
        response_text = response.content.strip()
        
        # Extract class index from response
        try:
            import re
            numbers = re.findall(r'\d+', response_text)
            if numbers:
                return int(numbers[0])
        except:
            pass
        
        return None

    def select_next_secondary_class_walk_partner(
        self,
        user_input: str,
        from_kind: str,
        current_class_index: int,
        candidates_context: str,
    ):
        """
        Pick the next secondary class index for WalkWithPartner when support is low.

        Args:
            user_input: User's primary intent text
            from_kind: "primary" or "secondary" (where the walk is stepping from)
            current_class_index: Current primary or secondary class index
            candidates_context: Formatted list of candidate secondary classes (index, support, description)

        Returns:
            Selected secondary class index (int) or None
        """
        template = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You choose the next secondary (satellite) class index for a composite prompt chain. "
                    "Reply with only the chosen class index as an integer.",
                ),
                (
                    "user",
                    """User input (overall goal): {user_input}

We are stepping from {from_kind} class index {current_class_index}.
Choose the next secondary class index from the candidates below (use only listed indices).

Candidates (secondary class index, observed support count, short description):
{candidates}

Return only the integer class index you choose:""",
                ),
            ]
        )
        chain = template | self.llm
        response = chain.invoke(
            {
                "user_input": user_input,
                "from_kind": from_kind,
                "current_class_index": current_class_index,
                "candidates": candidates_context,
            }
        )
        response_text = response.content.strip()
        try:
            import re

            numbers = re.findall(r"\d+", response_text)
            if numbers:
                return int(numbers[0])
        except Exception:
            pass
        return None

    def select_next_prompt_rag(self, current_prompt: str, candidate_prompts: list, completed_prompts: list = None):
        """
        Selects the next prompt to add to current prompt using RAG.
        
        Args:
            current_prompt: Current prompt (user input + any already added prompts)
            candidate_prompts: List of candidate prompt strings to choose from
            completed_prompts: Optional list of previously completed prompts to avoid repetition
        
        Returns:
            Dictionary with 'selected_prompt' and 'updated_prompt' keys, or None if parsing fails
        """
        import json
        
        # Format candidate prompts
        candidates_text = "\n".join([f"- {p}" for p in candidate_prompts])
        
        # Format completed prompts context
        completed_context = ""
        if completed_prompts:
            completed_text = "\n".join([f"- {p}" for p in completed_prompts])
            completed_context = f"\n\nPreviously completed prompts (avoid exact repetition with these):\n{completed_text}"
        
        template = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant that selects the best prompt to add to an existing prompt. Your selection should make the most sense and should not be repetitive within the current prompt."),
            ("user", """Given the current prompt and a list of candidate prompts, select the best one to add to it.

Current prompt: {current_prompt}

Candidate prompts to choose from:
{candidates}{completed_context}

Return a JSON object with two keys:
- "selected_prompt": the prompt you selected from the candidates
- "updated_prompt": the current prompt with the selected prompt added (comma-separated)

Make sure your selection:
1. Makes the most sense with the current prompt
2. Is not repetitive within the current prompt
3. Avoids exact repetition with previously completed prompts if any

Return only the JSON object:""")
        ])
        
        chain = template | self.llm
        response = chain.invoke({
            "current_prompt": current_prompt,
            "candidates": candidates_text,
            "completed_context": completed_context
        })
        
        response_text = response.content.strip()
        
        # Parse JSON response
        try:
            # Remove markdown code blocks if present
            if response_text.startswith("```"):
                response_text = response_text.split("```")[1]
                if response_text.startswith("json"):
                    response_text = response_text[4:]
            response_text = response_text.strip()
            
            result = json.loads(response_text)
            return {
                'selected_prompt': result['selected_prompt'],
                'updated_prompt': result['updated_prompt']
            }
        except (json.JSONDecodeError, KeyError):
            return None
    
    def decompose_prompts_batch(self, prompts: list):
        """
        Decomposes a batch of prompts into primary and secondary prompts.
        
        Args:
            prompts: List of prompt strings to decompose
        
        Returns:
            Dictionary mapping each prompt to {"primary": str, "secondaries": list}
            If a prompt has no secondaries, it will be marked with "ignore": True
        """
        import json
        
        # Format prompts for the template
        prompts_text = "\n".join([f"{i+1}. {p}" for i, p in enumerate(prompts)])
        
        template = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant that decomposes prompts into a primary objective and secondary details. The primary is the core subject or style, and secondaries are additional modifiers, details, or context."),
            ("user", """Given the following prompts, decompose each into a primary prompt (the core objective) and secondary prompts (auxiliary details that enrich it).

Examples:

1. "a painting by edward hopper of scenes from the mad max movie universe."
Primary: "a painting by edward hopper"
Secondaries: ["of scenes from the mad max movie universe"]

2. "a painting by edward hopper of a bird with a staff:"
Primary: "a painting by edward hopper"
Secondaries: ["of a bird with a staff"]

3. "a painting by edward hopper of an eerie scene on a rainy day, illuminated by the cityscape."
Primary: "a painting by edward hopper"
Secondaries: ["of an eerie scene on a rainy day", "illuminated by the cityscape"]

4. "a painting by edward hopper of a woman in the streets of an apocalyptic city, surrounded by the darkness."
Primary: "a painting by edward hopper"
Secondaries: ["of a woman in the streets of an apocalyptic city", "surrounded by the darkness"]

5. "a painting by edward hopper of a group of four friends laughing hysterically in front of a castle."
Primary: "a painting by edward hopper"
Secondaries: ["of a group of four friends", "friends laughing hysterically in front of a castle"]

6. "geodesic landscape, john chamberlain, christopher balaskas, tadao ando, 4 k"
Primary: "geodesic landscape"
Secondaries: ["john chamberlain", "christopher balaskas", "tadao ando", "4 k"]

7. "a painting by edward hopper"
This prompt has no secondaries, so it should be ignored.

Prompts to decompose:
{prompts}

Return a JSON object where each key is the original prompt (exactly as given) and the value is an object with:
- "primary": the primary prompt string
- "secondaries": a list of secondary prompt strings (can be empty)
- "ignore": true if the prompt has no secondaries or cannot be decomposed, false otherwise

Return only the JSON object:""")
        ])
        
        chain = template | self.llm
        response = chain.invoke({
            "prompts": prompts_text
        })
        
        response_text = response.content.strip()
        
        # Parse JSON response
        try:
            # Remove markdown code blocks if present
            if response_text.startswith("```"):
                response_text = response_text.split("```")[1]
                if response_text.startswith("json"):
                    response_text = response_text[4:]
            response_text = response_text.strip()
            
            result = json.loads(response_text)
            return result
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Error parsing LLM response: {e}")
            return {}
