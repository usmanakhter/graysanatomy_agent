"""
Search Engine - Implements lexical, semantic, and hybrid search
Slice 3: ALL search strategies implemented
"""
import os
import pickle
import numpy as np
from typing import List, Dict, Any
from rank_bm25 import BM25Okapi
import faiss


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
        self.embedding_model_name = embedding_model
        self.top_k = top_k
        
        # Lazy loading
        self.bm25_index = None
        self.chunks = None
        self.vector_store = None
        self.embedding_manager = None
        
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
        from data.loader import load_and_chunk
        from components.embeddings import EmbeddingManager
        from config import VECTOR_STORE_DIR
        
        # Create directory
        os.makedirs(VECTOR_STORE_DIR, exist_ok=True)
        
        # File paths
        index_file = os.path.join(VECTOR_STORE_DIR, f"{self.embedding_model_name}_index.faiss")
        chunks_file = os.path.join(VECTOR_STORE_DIR, f"{self.embedding_model_name}_chunks.pkl")
        
        # Initialize embedding manager
        self.embedding_manager = EmbeddingManager(self.embedding_model_name)
        
        if os.path.exists(index_file) and os.path.exists(chunks_file):
            print(f"Loading FAISS index from {index_file}...")
            
            # Load FAISS index
            self.vector_store = faiss.read_index(index_file)
            
            # Load chunks
            with open(chunks_file, "rb") as f:
                self.chunks = pickle.load(f)
            
            print(f"✓ Loaded FAISS index with {len(self.chunks)} chunks")
        else:
            print("Creating FAISS vector store (first time only)...")
            print("This may take 5-10 minutes depending on embedding model...")
            
            # Load chunks
            self.chunks = load_and_chunk()
            
            # Generate embeddings
            print(f"Generating embeddings for {len(self.chunks)} chunks...")
            embeddings = self.embedding_manager.embed_documents(self.chunks)
            
            # Create FAISS index
            dimension = embeddings.shape[1]
            print(f"Creating FAISS index (dimension: {dimension})...")
            
            # Use IndexFlatIP for inner product (cosine similarity after normalization)
            self.vector_store = faiss.IndexFlatIP(dimension)
            
            # Normalize embeddings for cosine similarity
            faiss.normalize_L2(embeddings)
            
            # Add to index
            self.vector_store.add(embeddings.astype('float32'))
            
            # Save index
            faiss.write_index(self.vector_store, index_file)
            
            # Save chunks
            with open(chunks_file, "wb") as f:
                pickle.dump(self.chunks, f)
            
            print(f"✓ Created FAISS index with {len(self.chunks)} chunks")
    
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
        # Generate query embedding
        query_embedding = self.embedding_manager.embed_query(query)
        
        # Normalize for cosine similarity
        query_embedding = query_embedding.reshape(1, -1).astype('float32')
        faiss.normalize_L2(query_embedding)
        
        # Search
        scores, indices = self.vector_store.search(query_embedding, self.top_k)
        
        # Format results
        results = []
        for score, idx in zip(scores[0], indices[0]):
            results.append({
                "text": self.chunks[idx],
                "score": float(score),
                "index": int(idx),
                "method": "semantic"
            })
        
        return results
    
    def _search_hybrid(self, query: str) -> List[Dict[str, Any]]:
        """
        Combined BM25 + Vector search with Reciprocal Rank Fusion
        
        Args:
            query: Search query
            
        Returns:
            Top K results
        """
        from config import RRF_K
        
        # Get more results from each method for fusion
        retrieve_k = self.top_k * 2
        
        # Save original top_k
        original_top_k = self.top_k
        self.top_k = retrieve_k
        
        # Get results from both methods
        lexical_results = self._search_lexical(query)
        semantic_results = self._search_semantic(query)
        
        # Restore original top_k
        self.top_k = original_top_k
        
        # Apply Reciprocal Rank Fusion
        fused_results = self._reciprocal_rank_fusion(
            lexical_results, 
            semantic_results,
            k=RRF_K
        )
        
        # Return top K
        return fused_results[:self.top_k]
    
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
        from config import HYBRID_WEIGHT_LEXICAL, HYBRID_WEIGHT_SEMANTIC
        
        # Build score map: chunk_index -> score
        scores = {}
        chunk_map = {}  # index -> full chunk data
        
        # Process lexical results
        for rank, result in enumerate(lexical_results, 1):
            idx = result["index"]
            rrf_score = HYBRID_WEIGHT_LEXICAL / (k + rank)
            scores[idx] = scores.get(idx, 0) + rrf_score
            chunk_map[idx] = result
        
        # Process semantic results
        for rank, result in enumerate(semantic_results, 1):
            idx = result["index"]
            rrf_score = HYBRID_WEIGHT_SEMANTIC / (k + rank)
            scores[idx] = scores.get(idx, 0) + rrf_score
            
            # Update chunk_map if not already present
            if idx not in chunk_map:
                chunk_map[idx] = result
        
        # Sort by fused score
        sorted_indices = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
        
        # Build final results
        fused_results = []
        for idx in sorted_indices:
            result = chunk_map[idx].copy()
            result["score"] = float(scores[idx])
            result["method"] = "hybrid"
            fused_results.append(result)
        
        return fused_results