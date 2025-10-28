"""
Configuration file for Gray's Anatomy AI Agent
Slice 3: Semantic + Hybrid Search with GPT/Anthropic
"""

# ============================================
# LLM OPTIONS (GPT + Anthropic only)
# ============================================
LLM_OPTIONS = {
    "gpt-4o-mini": {
        "provider": "openai",
        "model": "gpt-4o-mini",
        "description": "Fast, cost-effective ($0.15/1M tokens)",
        "context_length": 128000,
        "speed": "fast"
    },
    "gpt-4o": {
        "provider": "openai",
        "model": "gpt-4o",
        "description": "Most capable ($2.50/1M tokens)",
        "context_length": 128000,
        "speed": "medium"
    },
    "claude-3-5-sonnet": {
        "provider": "anthropic",
        "model": "claude-3-5-sonnet-20241022",
        "description": "Excellent reasoning ($3/1M tokens)",
        "context_length": 200000,
        "speed": "medium"
    },
    "claude-3-5-haiku": {
        "provider": "anthropic",
        "model": "claude-3-5-haiku-20241022",
        "description": "Fast and affordable ($0.80/1M tokens)",
        "context_length": 200000,
        "speed": "fast"
    }
}

DEFAULT_LLM = "gpt-4o-mini"

# ============================================
# EMBEDDING OPTIONS (for semantic search)
# LIGHTWEIGHT - NO PYTORCH!
# ============================================
EMBEDDING_OPTIONS = {
    "tfidf": {
        "model_name": "sklearn-tfidf",
        "description": "TF-IDF vectors (LIGHTWEIGHT - 10MB, no PyTorch!)",
        "dimension": 1000,
        "speed": "very fast",
        "size_mb": 10
    },
    "openai-small": {
        "model_name": "text-embedding-3-small",
        "description": "OpenAI API ($0.02/1M tokens, no local model)",
        "dimension": 1536,
        "speed": "fast",
        "size_mb": 0,  # API-based
        "provider": "openai"
    }
}

DEFAULT_EMBEDDING = "tfidf"  # FREE and super lightweight!

# ============================================
# SEARCH STRATEGY OPTIONS
# ============================================
SEARCH_STRATEGIES = {
    "lexical": {
        "name": "Lexical (BM25)",
        "description": "Keyword-based, fast, exact matches",
        "requires_embeddings": False,
        "speed": "very fast"
    },
    "semantic": {
        "name": "Semantic (Vector)",
        "description": "Meaning-based, conceptual matches",
        "requires_embeddings": True,
        "speed": "medium"
    },
    "hybrid": {
        "name": "Hybrid (BM25 + Vector)",
        "description": "Best of both worlds, balanced",
        "requires_embeddings": True,
        "speed": "medium"
    }
}

DEFAULT_SEARCH_STRATEGY = "hybrid"

# ============================================
# KNOWLEDGE GRAPH OPTIONS
# ============================================
KNOWLEDGE_GRAPH_OPTIONS = {
    "none": {
        "name": "None",
        "description": "Standard RAG (no graph)",
        "enabled": False
    },
    # Slice 4 will add more options
}

DEFAULT_KNOWLEDGE_GRAPH = "none"

# ============================================
# DATA & CHUNKING
# ============================================
GRAYS_ANATOMY_URL = "https://archive.org/stream/anatomyofhumanbo1918gray/anatomyofhumanbo1918gray_djvu.txt"
TEXT_FILE = "grays_anatomy.txt"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# ============================================
# RETRIEVAL PARAMETERS
# ============================================
TOP_K_RESULTS = 5
HYBRID_WEIGHT_LEXICAL = 0.5  # 50% weight to lexical
HYBRID_WEIGHT_SEMANTIC = 0.5  # 50% weight to semantic

# Reciprocal Rank Fusion constant
RRF_K = 60

# ============================================
# FILE PATHS
# ============================================
BM25_INDEX_FILE = "indexes/bm25_index.pkl"
VECTOR_STORE_DIR = "indexes/vector_store"
GRAPH_STORE_FILE = "indexes/knowledge_graph.pkl"

# ============================================
# PROMPTS
# ============================================
STANDARD_PROMPT = """You are an expert on Gray's Anatomy (1918 edition). Answer questions ONLY using the provided excerpts. Do not use any outside knowledge or information from web searches.

Context from Gray's Anatomy:
{context}

Question: {question}

Instructions:
- Base your answer EXCLUSIVELY on the context provided above
- If the context doesn't contain enough information, say "This information is not available in the provided excerpts from Gray's Anatomy"
- Use proper anatomical terminology from the text
- Be precise and educational
- Do NOT supplement with modern knowledge or external sources

Answer:"""

GRAPH_RAG_PROMPT = """You are an expert anatomist analyzing Gray's Anatomy (1918). Answer ONLY using the provided information. Do NOT use external knowledge.

Text Context from Gray's Anatomy:
{context}

Knowledge Graph Information:
{graph_context}

Question: {question}

Instructions:
- Use ONLY the text context and graph information provided
- If information is insufficient, state this clearly
- Do NOT add modern medical knowledge
- Do NOT use web information or external sources
- Synthesize only from the given context

Answer:"""

# ============================================
# UI DEFAULTS
# ============================================
DEFAULT_SETTINGS = {
    "llm": DEFAULT_LLM,
    "embedding": DEFAULT_EMBEDDING,
    "search_strategy": DEFAULT_SEARCH_STRATEGY,
    "knowledge_graph": DEFAULT_KNOWLEDGE_GRAPH,
    "top_k": TOP_K_RESULTS
}