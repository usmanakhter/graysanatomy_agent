"""
LLM Hub - Manages GPT and Anthropic models
Slice 3: Simplified to just OpenAI and Anthropic
"""
import os
from typing import Dict, Any, Optional
from config import LLM_OPTIONS, STANDARD_PROMPT, GRAPH_RAG_PROMPT


class LLMHub:
    """
    Unified interface for OpenAI and Anthropic LLMs
    """
    
    def __init__(self, model_name: str = "gpt-4o-mini", **kwargs):
        """
        Initialize LLM hub
        
        Args:
            model_name: Key from LLM_OPTIONS in config.py
            **kwargs: Additional settings including API keys
        """
        if model_name not in LLM_OPTIONS:
            raise ValueError(f"Unknown model: {model_name}. Choose from {list(LLM_OPTIONS.keys())}")
        
        self.model_name = model_name
        self.model_config = LLM_OPTIONS[model_name]
        self.provider = self.model_config["provider"]
        self.model_id = self.model_config["model"]
        self.api_key = kwargs.get(self.model_config["requires_key"])
        
        # Initialize appropriate client
        if self.provider == "openai":
            self._init_openai()
        elif self.provider == "anthropic":
            self._init_anthropic()
        else:
            raise ValueError(f"Unknown provider: {self.provider}")
    
    def _init_openai(self):
        """Initialize OpenAI client"""
        from openai import OpenAI
        
        api_key = self.api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY not found. "
                "Get one at https://platform.openai.com/api-keys"
            )
        
        self.client = OpenAI(api_key=api_key)
        print(f"✓ Initialized OpenAI: {self.model_id}")
    
    def _init_anthropic(self):
        """Initialize Anthropic client"""
        from anthropic import Anthropic
        
        api_key = self.api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY not found. "
                "Get one at https://console.anthropic.com/settings/keys"
            )
        
        self.client = Anthropic(api_key=api_key)
        print(f"✓ Initialized Anthropic: {self.model_id}")
    
    def generate(
        self, 
        question: str, 
        context: str, 
        graph_context: Optional[Dict] = None,
        max_tokens: int = 1000,
        temperature: float = 0.0  # Changed to 0 for deterministic output
    ) -> str:
        """
        Generate answer using selected LLM
        
        Args:
            question: User's question
            context: Retrieved text context
            graph_context: Optional knowledge graph context
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            
        Returns:
            Generated answer
        """
        # Choose prompt based on whether we have graph context
        if graph_context:
            prompt = self._format_graph_prompt(question, context, graph_context)
        else:
            prompt = self._format_standard_prompt(question, context)
        
        # Generate using appropriate provider
        if self.provider == "openai":
            return self._generate_openai(prompt, max_tokens, temperature)
        elif self.provider == "anthropic":
            return self._generate_anthropic(prompt, max_tokens, temperature)
    
    def _format_standard_prompt(self, question: str, context: str) -> str:
        """
        Format standard RAG prompt
        
        Args:
            question: User's question
            context: Retrieved context
            
        Returns:
            Formatted prompt
        """
        # Truncate context to fit model's context window
        max_context_length = 6000  # Conservative limit
        if len(context) > max_context_length:
            context = context[:max_context_length] + "..."
        
        prompt = STANDARD_PROMPT.format(context=context, question=question)
        return prompt
    
    def _format_graph_prompt(
        self, 
        question: str, 
        context: str, 
        graph_context: Dict
    ) -> str:
        """
        Format prompt with knowledge graph context
        
        Args:
            question: User's question
            context: Retrieved text context
            graph_context: Knowledge graph information
            
        Returns:
            Formatted prompt
        """
        # Format graph context as text
        graph_text = self._format_graph_as_text(graph_context)
        
        prompt = GRAPH_RAG_PROMPT.format(
            context=context[:4000],
            graph_context=graph_text,
            question=question
        )
        
        return prompt
    
    def _format_graph_as_text(self, graph_context: Dict) -> str:
        """
        Convert graph context to readable text
        
        Args:
            graph_context: Graph data
            
        Returns:
            Formatted text description
        """
        # SLICE 4: Will implement when we add knowledge graph
        return "Graph context not yet implemented"
    
    def _generate_openai(
        self, 
        prompt: str, 
        max_tokens: int, 
        temperature: float
    ) -> str:
        """
        Generate using OpenAI API
        
        Args:
            prompt: Formatted prompt
            max_tokens: Max tokens to generate
            temperature: Sampling temperature (0 for deterministic)
            
        Returns:
            Generated text
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model_id,
                messages=[
                    {
                        "role": "system", 
                        "content": "You are an expert on Gray's Anatomy (1918 edition). Answer questions ONLY using the provided text excerpts. Do not use any external knowledge, web searches, or modern medical information."
                    },
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,
                temperature=0.0  # Always 0 for deterministic, context-only answers
            )
            
            return response.choices[0].message.content.strip()
        
        except Exception as e:
            raise Exception(f"OpenAI API Error: {str(e)}")
    
    def _generate_anthropic(
        self, 
        prompt: str, 
        max_tokens: int, 
        temperature: float
    ) -> str:
        """
        Generate using Anthropic API
        
        Args:
            prompt: Formatted prompt
            max_tokens: Max tokens to generate
            temperature: Sampling temperature (0 for deterministic)
            
        Returns:
            Generated text
        """
        try:
            message = self.client.messages.create(
                model=self.model_id,
                max_tokens=max_tokens,
                temperature=0.0,  # Always 0 for deterministic, context-only answers
                system="You are an expert on Gray's Anatomy (1918 edition). Answer questions ONLY using the provided text excerpts. Do not use any external knowledge, web searches, or modern medical information. If the provided context doesn't contain the answer, say so explicitly.",
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            return message.content[0].text.strip()
        
        except Exception as e:
            raise Exception(f"Anthropic API Error: {str(e)}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about current model
        
        Returns:
            Model configuration
        """
        return {
            "name": self.model_name,
            "model_id": self.model_id,
            "provider": self.provider,
            **self.model_config
        }