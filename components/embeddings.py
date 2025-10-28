"""
Embedding Manager - Handles embeddings using Sentence Transformers
"""
from typing import List
import numpy as np
from config import EMBEDDING_OPTIONS


class EmbeddingManager:
    """
    Manages embedding generation using Sentence Transformers
    """
    
    def __init__(self, embedding_name: str = "bge-base"):
        """
        Initialize embedding manager
        
        Args:
            embedding_name: Key from EMBEDDING_OPTIONS in config.py
        """
        if embedding_name not in EMBEDDING_OPTIONS:
            raise ValueError(f"Unknown embedding: {embedding_name}")
        
        self.embedding_name = embedding_name
        self.config = EMBEDDING_OPTIONS[embedding_name]
        self.dimension = self.config["dimension"]
        
        # Lazy loading
        self.model = None
        
        # Initialize the model
        self._init_sentence_transformer()
    
    def _init_sentence_transformer(self):
        """Initialize Sentence Transformer model (local)"""
        from sentence_transformers import SentenceTransformer
        
        print(f"Loading embedding model: {self.config['model_name']}")
        print(f"Size: {self.config['size_mb']}MB - This may take a minute...")
        
        self.model = SentenceTransformer(self.config['model_name'])
        
        print(f"✓ Loaded {self.embedding_name} embeddings")
    
    def embed_documents(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for multiple documents
        
        Args:
            texts: List of text strings
            
        Returns:
            numpy array of shape (len(texts), dimension)
        """
        return self._embed_sentence_transformer(texts)
    
    def embed_query(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single query
        
        Args:
            text: Query text
            
        Returns:
            numpy array of shape (dimension,)
        """
        return self.model.encode(text, convert_to_numpy=True)
    
    def _embed_sentence_transformer(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings using Sentence Transformers
        
        Args:
            texts: List of texts
            
        Returns:
            numpy array of embeddings
        """
        # Show progress for large batches
        show_progress = len(texts) > 100
        
        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=show_progress,
            batch_size=32
        )
        
        return embeddings
    
    def get_info(self):
        """Get information about current embedding model"""
        return {
            "name": self.embedding_name,
            "dimension": self.dimension,
            "provider": "local",
            **self.config
        }