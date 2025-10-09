"""
Search functionality using BM25
"""
import os
import pickle
import numpy as np
from rank_bm25 import BM25Okapi
from data_loader import load_and_chunk
from config import BM25_INDEX_FILE, TOP_K_RESULTS


def create_search_index():
    """Create BM25 search index"""
    print("Building BM25 search index...")
    
    # Load chunks
    chunks = load_and_chunk()
    
    # Tokenize for BM25
    print("Tokenizing chunks...")
    tokenized_chunks = [chunk.lower().split() for chunk in chunks]
    
    # Create BM25 index
    print("Creating BM25 index...")
    bm25 = BM25Okapi(tokenized_chunks)
    
    # Save to disk
    with open(BM25_INDEX_FILE, "wb") as f:
        pickle.dump((bm25, chunks), f)
    
    print(f"✓ Index saved to {BM25_INDEX_FILE}")
    return bm25, chunks


def load_search_index():
    """Load existing search index or create new one"""
    if os.path.exists(BM25_INDEX_FILE):
        print(f"Loading index from {BM25_INDEX_FILE}...")
        with open(BM25_INDEX_FILE, "rb") as f:
            bm25, chunks = pickle.load(f)
        print(f"✓ Loaded index with {len(chunks):,} chunks")
        return bm25, chunks
    else:
        return create_search_index()


def search(query, k=TOP_K_RESULTS):
    """Search for relevant chunks"""
    bm25, chunks = load_search_index()
    
    # Tokenize query
    tokenized_query = query.lower().split()
    
    # Get scores
    scores = bm25.get_scores(tokenized_query)
    
    # Get top k indices
    top_indices = np.argsort(scores)[-k:][::-1]
    
    # Return top chunks
    results = [chunks[i] for i in top_indices]
    
    return results