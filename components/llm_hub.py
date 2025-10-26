"""
LLM Hub - Manages multiple LLM providers
Supports OpenAI GPT and Anthropic Claude models
"""
import os
import requests
from typing import Dict, Any, Optional
import openai
from anthropic import Anthropic
from config import LLM_OPTIONS, STANDARD_PROMPT, GRAPH_RAG_PROMPT


class LLMHub:
    """
    Unified interface for multiple LLM providers
    """
    
    def __init__(self, model_name: str = "gpt-3.5"):
        """
        Initialize LLM hub
        
        Args:
            model_name: Key from LLM_OPTIONS in config.py
        """
        if model_name not in LLM_OPTIONS:
            raise ValueError(f"Unknown model: {model_name}. Choose from {list(LLM_OPTIONS.keys())}")
        
        self.model_name = model_name
        self.model_config = LLM_OPTIONS[model_name]
        self.model = self.model_config["model"]
        self.provider = self.model_config["provider"]
        
        # Get API token
        key_name = self.model_config["requires_key"]
        self.api_token = os.environ.get(key_name)
        if not self.api_token:
            provider_name = "OpenAI" if self.provider == "openai" else "Anthropic"
            raise ValueError(
                f"{key_name} not found. Get one at: " +
                ("https://platform.openai.com/api-keys" if self.provider == "openai"
                 else "https://console.anthropic.com/settings/keys")
            )
            
        # Initialize client
        if self.provider == "openai":
            self.client = openai.Client(api_key=self.api_token)
        else:  # anthropic
            self.client = Anthropic(api_key=self.api_token)
    
    def generate(
        self, 
        question: str, 
        context: str, 
        graph_context: Optional[Dict] = None,
        max_tokens: int = 500,
        temperature: float = 0.1
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
        
        # Generate using appropriate API
        if self.provider == "openai":
            answer = self._call_openai_api(prompt, max_tokens, temperature)
        else:  # anthropic
            answer = self._call_anthropic_api(prompt, max_tokens, temperature)
        
        return answer
        
    def _call_openai_api(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float
    ) -> str:
        """Call OpenAI API"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            if "Rate limit" in str(e):
                raise Exception("Rate limit reached. Please wait a minute and try again.")
            elif "Invalid authentication" in str(e):
                raise Exception("Invalid OpenAI API key. Please check your OPENAI_API_KEY.")
            else:
                raise Exception(f"OpenAI API error: {str(e)}")
                
    def _call_anthropic_api(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float
    ) -> str:
        """Call Anthropic API"""
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text.strip()
        except Exception as e:
            if "rate_limit" in str(e):
                raise Exception("Rate limit reached. Please wait a minute and try again.")
            elif "invalid_api_key" in str(e):
                raise Exception("Invalid Anthropic API key. Please check your ANTHROPIC_API_KEY.")
            else:
                raise Exception(f"Anthropic API error: {str(e)}")
    
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
        max_context_length = self.model_config["context_length"] - 1000  # Leave room for response
        if len(context) > max_context_length:
            context = context[:max_context_length] + "..."
        
        return STANDARD_PROMPT.format(context=context, question=question)
    
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
        
        # Calculate safe context length
        max_context_length = (self.model_config["context_length"] - 1000) // 2
        
        return GRAPH_RAG_PROMPT.format(
            context=context[:max_context_length],
            graph_context=graph_text[:max_context_length],
            question=question
        )
    
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
    
    def _call_huggingface_api(
        self, 
        prompt: str, 
        max_tokens: int, 
        temperature: float
    ) -> str:
        """
        Call HuggingFace Inference API
        
        Args:
            prompt: Formatted prompt
            max_tokens: Max tokens to generate
            temperature: Sampling temperature
            
        Returns:
            Generated text
        """
        api_url = f"https://api-inference.huggingface.co/models/{self.repo_id}"
        headers = {"Authorization": f"Bearer {self.api_token}"}
        
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": max_tokens,
                "temperature": temperature,
                "top_p": 0.9,
                "return_full_text": False
            }
        }
        
        try:
            response = requests.post(
                api_url, 
                headers=headers, 
                json=payload, 
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                
                # Handle different response formats
                if isinstance(result, list) and len(result) > 0:
                    return result[0].get("generated_text", "").strip()
                elif isinstance(result, dict):
                    return result.get("generated_text", "").strip()
                else:
                    return str(result)
            
            else:
                error_msg = f"API Error {response.status_code}: {response.text}"
                
                # Provide helpful error messages
                if response.status_code == 503:
                    error_msg += "\n\nModel is loading. Please wait 20 seconds and try again."
                elif response.status_code == 401:
                    error_msg += "\n\nInvalid API token. Check HUGGINGFACEHUB_API_TOKEN."
                elif response.status_code == 429:
                    error_msg += "\n\nRate limit exceeded. Wait a few minutes."
                
                raise Exception(error_msg)
        
        except requests.exceptions.Timeout:
            raise Exception("Request timed out. Model may be overloaded. Try again.")
        except requests.exceptions.RequestException as e:
            raise Exception(f"Network error: {str(e)}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about current model
        
        Returns:
            Model configuration
        """
        return {
            "name": self.model_name,
            "model": self.model,
            "provider": self.provider,
            **self.model_config
        }