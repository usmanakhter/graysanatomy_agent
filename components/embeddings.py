"""
Lightweight Embedding Manager - NO PyTorch Required!
Works on 8GB RAM MacBook Air
Uses TF-IDF or OpenAI API embeddings
"""
import os
import numpy as np
from typing import List
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle


class EmbeddingManager:
    """
    Lightweight embeddings without heavy ML frameworks
    """
    
    def __init__(self, embedding_name: str = "tfidf"):
        """
        Initialize lightweight embedding manager
        
        Args:
            embedding_name: "tfidf" (local, light) or "openai" (API)
        """
        self.embedding_name = embedding_name
        self.model = None
        self.client = None
        
        if embedding_name == "tfidf":
            self.dimension = 1000  # Configurable
            self._init_tfidf()
        elif embedding_name == "openai":
            self.dimension = 1536
            self._init_openai()
        else:
            raise ValueError(f"Unknown embedding: {embedding_name}")
    
    def _init_tfidf(self):
        """Initialize TF-IDF (very lightweight, ~10MB)"""
        print("Initializing TF-IDF embeddings (lightweight, no PyTorch)...")
        
        # TF-IDF parameters optimized for anatomical text
        self.model = TfidfVectorizer(
            max_features=self.dimension,  # Dimension of vectors
            ngram_range=(1, 2),  # Unigrams and bigrams
            min_df=2,  # Ignore very rare terms
            max_df=0.8,  # Ignore very common terms
            sublinear_tf=True,  # Use log scaling
            strip_accents='unicode',
            lowercase=True,
            stop_words='english'
        )
        
        print("✓ TF-IDF initialized (RAM usage: ~10MB)")
    
    def _init_openai(self):
        """Initialize OpenAI client (API-based, no local model)"""
        from openai import OpenAI
        
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment")
        
        self.client = OpenAI(api_key=api_key)
        print("✓ Initialized OpenAI embeddings (API-based, 0MB local)")
    
    def fit_transform(self, texts: List[str]) -> np.ndarray:
        """
        Fit TF-IDF model and transform texts (only for TF-IDF)
        
        Args:
            texts: List of document texts
            
        Returns:
            numpy array of embeddings
        """
        if self.embedding_name != "tfidf":
            raise ValueError("fit_transform only for TF-IDF")
        
        print(f"Fitting TF-IDF on {len(texts)} documents...")
        embeddings = self.model.fit_transform(texts).toarray()
        print(f"✓ Generated TF-IDF vectors: {embeddings.shape}")
        
        return embeddings
    
    def embed_documents(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for multiple documents
        
        Args:
            texts: List of text strings
            
        Returns:
            numpy array of shape (len(texts), dimension)
        """
        if self.client:  # OpenAI
            return self._embed_openai(texts)
        else:  # TF-IDF
            if not hasattr(self.model, 'vocabulary_'):
                # Model not fitted yet, fit first
                return self.fit_transform(texts)
            else:
                # Model already fitted, just transform
                return self.model.transform(texts).toarray()
    
    def embed_query(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single query
        
        Args:
            text: Query text
            
        Returns:
            numpy array of shape (dimension,)
        """
        if self.client:  # OpenAI
            embeddings = self._embed_openai([text])
            return embeddings[0]
        else:  # TF-IDF
            if not hasattr(self.model, 'vocabulary_'):
                raise ValueError("TF-IDF model not fitted yet. Call fit_transform first.")
            return self.model.transform([text]).toarray()[0]
    
    def _embed_openai(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings using OpenAI API
        
        Args:
            texts: List of texts
            
        Returns:
            numpy array of embeddings
        """
        # OpenAI has a limit of ~8000 texts per request
        batch_size = 2000
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            response = self.client.embeddings.create(
                input=batch,
                model="text-embedding-3-small"
            )
            
            batch_embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(batch_embeddings)
        
        return np.array(all_embeddings)
    
    def save(self, filepath: str):
        """Save TF-IDF model to disk"""
        if self.embedding_name == "tfidf":
            with open(filepath, 'wb') as f:
                pickle.dump(self.model, f)
            print(f"✓ Saved TF-IDF model to {filepath}")
    
    def load(self, filepath: str):
        """Load TF-IDF model from disk"""
        if self.embedding_name == "tfidf":
            with open(filepath, 'rb') as f:
                self.model = pickle.load(f)
            print(f"✓ Loaded TF-IDF model from {filepath}")
    
    def get_info(self):
        """Get information about current embedding model"""
        return {
            "name": self.embedding_name,
            "dimension": self.dimension,
            "provider": "local" if self.embedding_name == "tfidf" else "openai",
            "ram_usage": "~10MB" if self.embedding_name == "tfidf" else "0MB (API)"
        }