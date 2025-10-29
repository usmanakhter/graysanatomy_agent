"""
Orchestrator - Routes requests based on user configuration
Follows vertical slice architecture for incremental development
"""
from typing import Dict, List, Any
from config import DEFAULT_SETTINGS


class Orchestrator:
    """
    Central orchestrator that routes queries through selected components
    """
    
    def __init__(self, settings: Dict[str, Any] = None):
        """
        Initialize orchestrator with user settings
        
        Args:
            settings: User-selected configuration (LLM, search, graph options)
        """
        self.settings = settings or DEFAULT_SETTINGS.copy()
        
        # Lazy loading - only import what we need
        self.search_engine = None
        self.llm_engine = None
        self.graph_engine = None
        
    def query(self, question: str) -> Dict[str, Any]:
        """
        Main query method - orchestrates the full RAG pipeline
        
        Args:
            question: User's question
            
        Returns:
            Dict containing answer, sources, metadata
        """
        # Step 1: Search for relevant context
        search_results = self._search(question)
        
        # Step 2: Enhance with knowledge graph (if enabled)
        graph_context = self._get_graph_context(question, search_results)
        
        # Step 3: Generate answer with LLM
        answer = self._generate_answer(question, search_results, graph_context)
        
        # Step 4: Return structured response
        return {
            "answer": answer,
            "sources": search_results,
            "graph_context": graph_context,
            "settings": self.settings,
            "metadata": {
                "num_sources": len(search_results),
                "search_strategy": self.settings["search_strategy"],
                "llm_model": self.settings["llm"],
                "graph_enabled": self.settings["knowledge_graph"] != "none"
            }
        }
    
    def _search(self, question: str) -> List[Dict[str, Any]]:
        """
        Route to appropriate search strategy
        
        Args:
            question: User's question
            
        Returns:
            List of retrieved document chunks
        """
        # Lazy import search module
        if self.search_engine is None:
            from components.search import SearchEngine
            self.search_engine = SearchEngine(
                strategy=self.settings["search_strategy"],
                embedding_model=self.settings["embedding"],
                top_k=self.settings["top_k"]
            )
        
        results = self.search_engine.search(question)
        return results
    
    def _get_graph_context(self, question: str, search_results: List[Dict]) -> Dict[str, Any]:
        """
        Get knowledge graph context if enabled

        Args:
            question: User's question
            search_results: Retrieved document chunks

        Returns:
            Graph context or None
        """
        if self.settings["knowledge_graph"] == "none":
            return None

        # Lazy import graph module
        if self.graph_engine is None:
            import os
            from components.graph_rag import GraphRAG
            from config import GRAPH_STORE_FILE
            from data.loader import load_and_chunk

            self.graph_engine = GraphRAG(
                mode=self.settings["knowledge_graph"]
            )

            # Load or build graph
            if os.path.exists(GRAPH_STORE_FILE):
                print(f"Loading knowledge graph from {GRAPH_STORE_FILE}...")
                self.graph_engine.load(GRAPH_STORE_FILE)
            else:
                print("Building knowledge graph (this may take 2-5 minutes)...")
                chunks = load_and_chunk()

                # Progress callback
                def progress(current, total):
                    if current % 500 == 0:
                        print(f"  Processed {current}/{total} chunks...")

                self.graph_engine.build_graph(chunks, progress_callback=progress)
                self.graph_engine.save(GRAPH_STORE_FILE)
                print(f"✓ Knowledge graph built and saved")
                stats = self.graph_engine.get_stats()
                print(f"  Entities: {stats['num_entities']}, Relationships: {stats['num_relationships']}")

        graph_context = self.graph_engine.get_context(question, search_results)
        return graph_context
    
    def _generate_answer(
        self, 
        question: str, 
        search_results: List[Dict], 
        graph_context: Dict = None
    ) -> str:
        """
        Generate answer using selected LLM
        
        Args:
            question: User's question
            search_results: Retrieved document chunks
            graph_context: Optional graph context
            
        Returns:
            Generated answer text
        """
        # Lazy import LLM module
        if self.llm_engine is None:
            from components.llm_hub import LLMHub
            # Get API keys from settings
            llm_config = {"model_name": self.settings["llm"]}
            if "openai_api_key" in self.settings:
                llm_config["openai_api_key"] = self.settings["openai_api_key"]
            if "anthropic_api_key" in self.settings:
                llm_config["anthropic_api_key"] = self.settings["anthropic_api_key"]
            
            self.llm_engine = LLMHub(**llm_config)
        
        # Format context from search results
        context = "\n\n".join([
            result.get("text", "") for result in search_results
        ])
        
        # Generate answer
        answer = self.llm_engine.generate(
            question=question,
            context=context,
            graph_context=graph_context
        )
        
        return answer
    
    def update_settings(self, new_settings: Dict[str, Any]):
        """
        Update settings and reset engines if needed
        
        Args:
            new_settings: New configuration settings
        """
        # Check if we need to reload components
        if new_settings.get("llm") != self.settings.get("llm"):
            self.llm_engine = None
            
        if new_settings.get("search_strategy") != self.settings.get("search_strategy"):
            self.search_engine = None
            
        if new_settings.get("knowledge_graph") != self.settings.get("knowledge_graph"):
            self.graph_engine = None
        
        self.settings.update(new_settings)