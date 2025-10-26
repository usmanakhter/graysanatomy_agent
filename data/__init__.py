"""
Data package - Text loading and preprocessing
"""
from .loader import download_text, load_text, chunk_text, load_and_chunk

__all__ = ['download_text', 'load_text', 'chunk_text', 'load_and_chunk']