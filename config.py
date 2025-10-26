"""
Configuration file for Gray's Anatomy AI Agent
Multi-choice architecture with toggleable options
"""

# ============================================
# LLM OPTIONS
# ============================================

LLM_OPTIONS = {
    "gpt-4": {
        "model": "gpt-4-1106-preview",
        "description": "Most capable GPT model, best for medical knowledge",
        "context_length": 128000,
        "speed": "fast",
        "provider": "openai",
        "requires_key": "OPENAI_API_KEY"
    },
    "gpt-3.5": {
        "model": "gpt-3.5-turbo-1106",
        "description": "Fast and cost-effective",
        "context_length": 16385,
        "speed": "very fast",
        "provider": "openai",
        "requires_key": "OPENAI_API_KEY"
    },
    "claude-3": {
        "model": "claude-3-opus-20240229",
        "description": "Anthropic's most capable model",
        "context_length": 200000,
        "speed": "fast",
        "provider": "anthropic",
        "requires_key": "ANTHROPIC_API_KEY"
    },
    "claude-instant": {
        "model": "claude-3-sonnet-20240229",
        "description": "Fast, efficient Anthropic model",
        "context_length": 200000,
        "speed": "very fast",
        "provider": "anthropic",
        "requires_key": "ANTHROPIC_API_KEY"
    }
}

DEFAULT_LLM = "gpt-3.5"  # Default to GPT-3.5 for balance of cost/performance


# ============================================
# EMBEDDING OPTIONS (for semantic search)
# ============================================
EMBEDDING_OPTIONS = {
    "minilm": {
        "model_name": "sentence-transformers/all-MiniLM-L6-v2",
        "description": "Fast, lightweight (80MB)",
        "dimension": 384,
        "speed": "very fast"
    },
    "bge-small": {
        "model_name": "BAAI/bge-small-en-v1.5",
        "description": "Better quality, small (133MB)",
        "dimension": 384,
        "speed": "fast"
    },
    "bge-base": {
        "model_name": "BAAI/bge-base-en-v1.5",
        "description": "High quality (438MB)",
        "dimension": 768,
        "speed": "medium"
    },
    "bge-large": {
        "model_name": "BAAI/bge-large-en-v1.5",
        "description": "Best quality (1.34GB)",
        "dimension": 1024,
        "speed": "slow"
    },
    "biobert": {
        "model_name": "pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb",
        "description": "Medical domain specialist",
        "dimension": 768,
        "speed": "medium"
    }
}

DEFAULT_EMBEDDING = "bge-base"

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
        "description": "Meaning-based, slower, conceptual matches",
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
    "simple": {
        "name": "Simple Graph",
        "description": "Entity extraction + relationships",
        "enabled": True,
        "method": "spacy"  # or "llm"
    },
    "graph-rag": {
        "name": "Full GraphRAG",
        "description": "Community detection + hierarchical summaries",
        "enabled": True,
        "method": "microsoft"  # Microsoft's GraphRAG approach
    }
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

# ============================================
# GRAPH-RAG PARAMETERS
# ============================================
GRAPH_ENTITY_TYPES = [
    "ORGAN", "BONE", "MUSCLE", "NERVE", "ARTERY", 
    "VEIN", "TISSUE", "CELL", "SYSTEM", "REGION"
]

GRAPH_RELATIONSHIP_TYPES = [
    "CONNECTED_TO", "PART_OF", "INNERVATES", 
    "SUPPLIES_BLOOD_TO", "LOCATED_IN", "FUNCTIONS_AS"
]

# ============================================
# FILE PATHS
# ============================================
BM25_INDEX_FILE = "indexes/bm25_index.pkl"
VECTOR_STORE_DIR = "indexes/vector_store"
GRAPH_STORE_FILE = "indexes/knowledge_graph.pkl"

# ============================================
# PROMPTS
# ============================================
STANDARD_PROMPT = """Based on the following excerpts from Gray's Anatomy (1918), answer the question accurately.

Context:
{context}

Question: {question}

Provide a clear, educational answer using proper anatomical terminology:"""

GRAPH_RAG_PROMPT = """You are an expert anatomist. Use the following information to answer the question:

Text Context:
{context}

Knowledge Graph Information:
{graph_context}

Question: {question}

Synthesize information from both the text and the knowledge graph to provide a comprehensive answer:"""

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