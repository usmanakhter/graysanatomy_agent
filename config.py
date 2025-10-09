"""
Configuration file for Gray's Anatomy AI Agent
"""

# Data source
GRAYS_ANATOMY_URL = "https://archive.org/stream/anatomyofhumanbo1918gray/anatomyofhumanbo1918gray_djvu.txt"
TEXT_FILE = "grays_anatomy.txt"

# Chunking parameters
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150

# Search parameters
TOP_K_RESULTS = 5

# Model selection
USE_OPENAI = True  # Set to True to use OpenAI instead of HuggingFace
LLM_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"

# File paths
BM25_INDEX_FILE = "bm25_index.pkl"

# Prompt template
PROMPT_TEMPLATE = """Based on the following excerpts from Gray's Anatomy (1918), answer the question accurately and educationally.

Context from Gray's Anatomy:
{context}

Question: {question}

Instructions:
- Provide a clear, accurate answer based on the context
- Use proper anatomical terminology
- If the context doesn't fully answer the question, acknowledge this
- Be concise but educational

Answer:"""