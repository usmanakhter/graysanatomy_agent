"""
Search Engine - Implements lexical, semantic, and hybrid search
Vertical Slice 1: Lexical (BM25) only
"""
import os
import pickle
import numpy as np
from typing import List, Dict, Any
from rank_bm25 import BM25Okapi


class SearchEngine:
    """
    Handles all search strategies: lexical, semantic, hybrid
    """
    
    def __init__(self, strategy: str = "lexical", embedding_model: str = None, top_k: int = 5):
        """
        Initialize search engine
        
        Args:
            strategy: "lexical", "semantic", or "hybrid"
            embedding_model: Embedding model name (for semantic/hybrid)
            top_k: Number of results to return
        """
        self.strategy = strategy
        self.embedding_model = embedding_model
        self.top_k = top_k
        
        # Lazy loading
        self.bm25_index = None
        self.chunks = None
        self.vector_store = None
        
        # Load based on strategy
        if strategy in ["lexical", "hybrid"]:
            self._load_bm25_index()
        
        if strategy in ["semantic", "hybrid"]:
            self._load_vector_store()
    
    def search(self, query: str) -> List[Dict[str, Any]]:
        """
        Search using selected strategy
        
        Args:
            query: User's search query
            
        Returns:
            List of search results with text and metadata
        """
        if self.strategy == "lexical":
            return self._search_lexical(query)
        elif self.strategy == "semantic":
            return self._search_semantic(query)
        elif self.strategy == "hybrid":
            return self._search_hybrid(query)
        else:
            raise ValueError(f"Unknown search strategy: {self.strategy}")
    
    def _load_bm25_index(self):
        """Load or create BM25 index"""
        from data.loader import load_and_chunk
        from config import BM25_INDEX_FILE
        
        # Create indexes directory if needed
        os.makedirs(os.path.dirname(BM25_INDEX_FILE), exist_ok=True)
        
        if os.path.exists(BM25_INDEX_FILE):
            print(f"Loading BM25 index from {BM25_INDEX_FILE}...")
            with open(BM25_INDEX_FILE, "rb") as f:
                self.bm25_index, self.chunks = pickle.load(f)
            print(f"✓ Loaded BM25 index with {len(self.chunks)} chunks")
        else:
            print("Creating BM25 index (first time only)...")
            self.chunks = load_and_chunk()
            
            # Tokenize chunks
            tokenized_chunks = [chunk.lower().split() for chunk in self.chunks]
            
            # Create BM25 index
            self.bm25_index = BM25Okapi(tokenized_chunks)
            
            # Save for next time
            with open(BM25_INDEX_FILE, "wb") as f:
                pickle.dump((self.bm25_index, self.chunks), f)
            
            print(f"✓ Created BM25 index with {len(self.chunks)} chunks")
    
    def _load_vector_store(self):
        """Load or create FAISS vector store"""
        # SLICE 3: Will implement semantic search here
        print("Note: Semantic search not yet implemented (Slice 3)")
        pass
    
    def _search_lexical(self, query: str) -> List[Dict[str, Any]]:
        """
        BM25 keyword search
        
        Args:
            query: Search query
            
        Returns:
            Top K results
        """
        # Tokenize query
        tokenized_query = query.lower().split()
        
        # Get BM25 scores
        scores = self.bm25_index.get_scores(tokenized_query)
        
        # Get top K indices
        top_indices = np.argsort(scores)[-self.top_k:][::-1]
        
        # Format results
        results = []
        for idx in top_indices:
            results.append({
                "text": self.chunks[idx],
                "score": float(scores[idx]),
                "index": int(idx),
                "method": "lexical"
            })
        
        return results
    
    def _search_semantic(self, query: str) -> List[Dict[str, Any]]:
        """
        Vector similarity search
        
        Args:
            query: Search query
            
        Returns:
            Top K results
        """
        # SLICE 3: Will implement here
        raise NotImplementedError("Semantic search coming in Slice 3!")
    
    def _search_hybrid(self, query: str) -> List[Dict[str, Any]]:
        """
        Combined BM25 + Vector search with Reciprocal Rank Fusion
        
        Args:
            query: Search query
            
        Returns:
            Top K results
        """
        # SLICE 3: Will implement here
        raise NotImplementedError("Hybrid search coming in Slice 3!")
    
    def _reciprocal_rank_fusion(
        self, 
        lexical_results: List[Dict], 
        semantic_results: List[Dict],
        k: int = 60
    ) -> List[Dict[str, Any]]:
        """
        Combine lexical and semantic results using RRF
        
        Args:
            lexical_results: BM25 results
            semantic_results: Vector search results
            k: RRF constant (default 60)
            
        Returns:
            Fused and re-ranked results
        """
        # SLICE 3: Will implement here
        pass