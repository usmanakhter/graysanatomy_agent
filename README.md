# 🧠 Gray's Anatomy AI Assistant

Intelligent anatomy question-answering system using RAG (Retrieval-Augmented Generation) architecture.

## 🚀 Quick Start

### 1. Clone Repository
```bash
git clone https://github.com/yourusername/graysanatomy_agent.git
cd graysanatomy_agent
```

### 2. Install Dependencies

**Option A: Quick Setup (Recommended)**
```bash
chmod +x setup_slice4.sh
./setup_slice4.sh
```

**Option B: Manual Setup**
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python test_graph.py  # Verify installation
```

### 3. Set API Keys
Create a `.env` file in the project root OR add it in the app UI:

```bash
# Required: Choose at least one
OPENAI_API_KEY=sk-...              # Get at: https://platform.openai.com/api-keys
ANTHROPIC_API_KEY=sk-ant-...       # Get at: https://console.anthropic.com/settings/keys
```

### 4. Run Application
```bash
streamlit run app.py
```

### 5. First Run
- Downloads Gray's Anatomy text (~30 seconds)
- Builds BM25 search index (~1 minute)
- If using knowledge graph: builds graph (~2-5 minutes)
- Subsequent runs are instant!

## 📁 Project Structure

```
graysanatomy_agent/
├── app.py                 # Streamlit UI (main entry point)
├── orchestrator.py        # Central routing logic
├── config.py             # All configuration options
├── requirements.txt      # Python dependencies
│
├── components/
│   ├── search.py         # Search strategies (BM25, FAISS, hybrid)
│   ├── llm_hub.py        # LLM provider management
│   └── graph_rag.py      # Knowledge graph (Slice 4)
│
├── data/
│   └── loader.py         # Text loading and chunking
│
└── indexes/              # Generated indexes (auto-created)
    ├── bm25_index.pkl
    ├── vector_store/
    └── knowledge_graph.pkl
```

## 💡 Example Usage

### Standard RAG (Knowledge Graph: None)
```python
# Ask a question
"What are the main bones of the skull?"

# System automatically:
1. Searches Gray's Anatomy text (lexical/semantic/hybrid)
2. Retrieves top 5 relevant chunks
3. Sends to LLM (GPT/Claude) with context
4. Returns comprehensive answer
5. Shows sources used
```

### Graph RAG (Knowledge Graph: Entity/Community)
```python
# Ask a question
"What connects to the temporal bone?"

# System automatically:
1. Searches Gray's Anatomy text (lexical/semantic/hybrid)
2. Retrieves top 5 relevant chunks
3. Extracts entities from question and results
4. Traverses knowledge graph to find related entities
5. Adds graph context (entities, relationships)
6. Sends to LLM with both text + graph context
7. Returns enriched answer
8. Shows sources AND graph relationships
```

## 🔧 Configuration

Edit `config.py` to customize:

```python
# Chunking
CHUNK_SIZE = 1000          # Characters per chunk
CHUNK_OVERLAP = 200        # Overlap between chunks

# Retrieval
TOP_K_RESULTS = 5          # Number of chunks to retrieve

# LLM Options
DEFAULT_LLM = "gpt-4o-mini"           # or "gpt-4o", "claude-3-5-sonnet", "claude-3-5-haiku"

# Search Strategy
DEFAULT_SEARCH_STRATEGY = "hybrid"     # or "lexical", "semantic"
DEFAULT_EMBEDDING = "tfidf"            # or "openai-small"

# Knowledge Graph (Slice 4)
DEFAULT_KNOWLEDGE_GRAPH = "none"       # or "entity", "community"
```

See [KNOWLEDGE_GRAPH_SETUP.md](KNOWLEDGE_GRAPH_SETUP.md) for detailed knowledge graph documentation.

## 📊 Performance

**Slice 4 Metrics:**
- **Query time**: 1-3 seconds (after indexes built)
- **Index build time** (one-time):
  - BM25: ~1 minute
  - TF-IDF vectors: ~2-3 minutes
  - Knowledge graph: ~2-5 minutes
- **Memory usage**:
  - Base system: ~200 MB
  - With graph: ~300-400 MB
- **Disk storage**:
  - BM25 index: ~10 MB
  - TF-IDF vectors: ~15 MB
  - Knowledge graph: ~20-50 MB


## 🙏 Acknowledgments

- **Gray's Anatomy (1918)** - Public domain medical text
- **OpenAI** - GPT models
- **Anthropic** - Claude models
- **spaCy** - Lightweight NLP for entity extraction
- **NetworkX** - Graph algorithms
- **Streamlit** - Excellent Python web framework
