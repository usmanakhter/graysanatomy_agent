"""
Lightweight Search Engine - NO PyTorch or FAISS!
Works on 8GB RAM without heavy ML frameworks
"""
import os
import pickle
import numpy as np
from typing import List, Dict, Any
from rank_bm25 import BM25Okapi
from sklearn.metrics.pairwise import cosine_similarity


class SearchEngine:
    """
    Lightweight search: BM25 + TF-IDF (no heavy frameworks)
    """
    
    def __init__(self, strategy: str = "lexical", embedding_model: str = None, top_k: int = 5):
        """
        Initialize search engine
        
        Args:
            strategy: "lexical", "semantic", or "hybrid"
            embedding_model: "tfidf" or "openai"
            top_k: Number of results to return
        """
        self.strategy = strategy
        self.embedding_model_name = embedding_model
        self.top_k = top_k
        
        # Lazy loading
        self.bm25_index = None
        self.chunks = None
        self.document_embeddings = None
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
        
        os.makedirs(os.path.dirname(BM25_INDEX_FILE), exist_ok=True)
        
        if os.path.exists(BM25_INDEX_FILE):
            print(f"Loading BM25 index...")
            with open(BM25_INDEX_FILE, "rb") as f:
                self.bm25_index, self.chunks = pickle.load(f)
            print(f"✓ Loaded BM25 index with {len(self.chunks)} chunks")
        else:
            print("Creating BM25 index...")
            self.chunks = load_and_chunk()
            
            tokenized_chunks = [chunk.lower().split() for chunk in self.chunks]
            self.bm25_index = BM25Okapi(tokenized_chunks)
            
            with open(BM25_INDEX_FILE, "wb") as f:
                pickle.dump((self.bm25_index, self.chunks), f)
            
            print(f"✓ Created BM25 index with {len(self.chunks)} chunks")
    
    def _load_vector_store(self):
        """Load or create lightweight vector store (TF-IDF or OpenAI)"""
        from data.loader import load_and_chunk
        from components.embeddings_lightweight import EmbeddingManager
        from config import VECTOR_STORE_DIR
        
        os.makedirs(VECTOR_STORE_DIR, exist_ok=True)
        
        # File paths
        embeddings_file = os.path.join(VECTOR_STORE_DIR, f"{self.embedding_model_name}_embeddings.pkl")
        chunks_file = os.path.join(VECTOR_STORE_DIR, f"{self.embedding_model_name}_chunks.pkl")
        tfidf_model_file = os.path.join(VECTOR_STORE_DIR, f"{self.embedding_model_name}_model.pkl")
        
        # Initialize embedding manager
        self.embedding_manager = EmbeddingManager(self.embedding_model_name)
        
        if os.path.exists(embeddings_file) and os.path.exists(chunks_file):
            print(f"Loading vector store...")
            
            # Load embeddings
            with open(embeddings_file, "rb") as f:
                self.document_embeddings = pickle.load(f)
            
            # Load chunks
            with open(chunks_file, "rb") as f:
                self.chunks = pickle.load(f)
            
            # Load TF-IDF model if exists
            if self.embedding_model_name == "tfidf" and os.path.exists(tfidf_model_file):
                self.embedding_manager.load(tfidf_model_file)
            
            print(f"✓ Loaded vector store with {len(self.chunks)} chunks")
        else:
            print("Creating vector store (lightweight, no PyTorch)...")
            print("This will take 2-5 minutes...")
            
            # Load chunks
            self.chunks = load_and_chunk()
            
            # Generate embeddings
            print(f"Generating {self.embedding_model_name} embeddings...")
            if self.embedding_model_name == "tfidf":
                # Fit and transform in one go
                self.document_embeddings = self.embedding_manager.fit_transform(self.chunks)
                # Save TF-IDF model
                self.embedding_manager.save(tfidf_model_file)
            else:
                # OpenAI embeddings
                self.document_embeddings = self.embedding_manager.embed_documents(self.chunks)
            
            print(f"✓ Generated embeddings: {self.document_embeddings.shape}")
            
            # Save embeddings
            with open(embeddings_file, "wb") as f:
                pickle.dump(self.document_embeddings, f)
            
            # Save chunks
            with open(chunks_file, "wb") as f:
                pickle.dump(self.chunks, f)
            
            print(f"✓ Created vector store with {len(self.chunks)} chunks")
    
    def _search_lexical(self, query: str) -> List[Dict[str, Any]]:
        """BM25 keyword search"""
        tokenized_query = query.lower().split()
        scores = self.bm25_index.get_scores(tokenized_query)
        top_indices = np.argsort(scores)[-self.top_k:][::-1]
        
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
        """Semantic search using cosine similarity"""
        # Generate query embedding
        query_embedding = self.embedding_manager.embed_query(query)
        
        # Reshape for cosine similarity
        query_embedding = query_embedding.reshape(1, -1)
        
        # Calculate cosine similarities
        similarities = cosine_similarity(query_embedding, self.document_embeddings)[0]
        
        # Get top K indices
        top_indices = np.argsort(similarities)[-self.top_k:][::-1]
        
        # Format results
        results = []
        for idx in top_indices:
            results.append({
                "text": self.chunks[idx],
                "score": float(similarities[idx]),
                "index": int(idx),
                "method": "semantic"
            })
        
        return results
    
    def _search_hybrid(self, query: str) -> List[Dict[str, Any]]:
        """Hybrid search with Reciprocal Rank Fusion"""
        from config import RRF_K
        
        # Get more results for fusion
        retrieve_k = self.top_k * 2
        original_top_k = self.top_k
        self.top_k = retrieve_k
        
        # Get results from both methods
        lexical_results = self._search_lexical(query)
        semantic_results = self._search_semantic(query)
        
        # Restore original top_k
        self.top_k = original_top_k
        
        # Apply RRF
        fused_results = self._reciprocal_rank_fusion(
            lexical_results, 
            semantic_results,
            k=RRF_K
        )
        
        return fused_results[:self.top_k]
    
    def _reciprocal_rank_fusion(
        self, 
        lexical_results: List[Dict], 
        semantic_results: List[Dict],
        k: int = 60
    ) -> List[Dict[str, Any]]:
        """Combine lexical and semantic results using RRF"""
        from config import HYBRID_WEIGHT_LEXICAL, HYBRID_WEIGHT_SEMANTIC
        
        scores = {}
        chunk_map = {}
        
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