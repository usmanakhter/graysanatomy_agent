# 🧠 Gray's Anatomy AI Assistant

Intelligent anatomy question-answering system using RAG (Retrieval-Augmented Generation) architecture.

## 🎯 Current Status: **Slice 4 Complete** ✅

### What's Working:
- ✅ **Question Answering**: Ask any anatomy question
- ✅ **Multiple Search Strategies**: Lexical (BM25), Semantic (TF-IDF/OpenAI), Hybrid (RRF)
- ✅ **Multiple LLMs**: OpenAI (GPT-4o, GPT-4o-mini), Anthropic (Claude 3.5 Sonnet/Haiku)
- ✅ **Knowledge Graph**: Entity extraction, relationship detection, graph traversal
- ✅ **Source Attribution**: See which text chunks were used
- ✅ **Performance Metrics**: Track response times
- ✅ **Lightweight Design**: Works on 8GB RAM, no PyTorch

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
Create a `.env` file in the project root:

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

## 🗺️ Roadmap

### ✅ Slice 1: Core System
- [x] Basic RAG pipeline
- [x] BM25 lexical search
- [x] Single LLM (Mistral-7B)
- [x] Streamlit UI
- [x] Source attribution

### ✅ Slice 2: Multiple LLMs
- [x] OpenAI (GPT-4o, GPT-4o-mini)
- [x] Anthropic (Claude 3.5 Sonnet, Claude 3.5 Haiku)
- [x] Dropdown selector in UI
- [x] Temperature=0 for deterministic answers

### ✅ Slice 3: Advanced Search
- [x] Semantic search (TF-IDF/OpenAI embeddings)
- [x] Hybrid search (BM25 + Vector + RRF)
- [x] Search strategy selector
- [x] Lightweight (no PyTorch)

### ✅ Slice 4: Knowledge Graph (CURRENT)
- [x] Entity extraction (spaCy + pattern-based)
- [x] Relationship detection (co-occurrence)
- [x] Graph construction (NetworkX)
- [x] GraphRAG (entity & community modes)
- [x] Toggle-able (3 modes: none/entity/community)
- [ ] Graph visualization (future enhancement)

## 🎨 Features by Slice

| Feature | Slice 1 | Slice 2 | Slice 3 | Slice 4 |
|---------|---------|---------|---------|---------|
| **Question Answering** | ✅ | ✅ | ✅ | ✅ |
| **Lexical Search** | ✅ | ✅ | ✅ | ✅ |
| **Semantic Search** | ❌ | ❌ | ✅ | ✅ |
| **Hybrid Search** | ❌ | ❌ | ✅ | ✅ |
| **Single LLM** | ✅ | ❌ | ❌ | ❌ |
| **Multiple LLMs** | ❌ | ✅ | ✅ | ✅ |
| **Knowledge Graph** | ❌ | ❌ | ❌ | ✅ |
| **Graph Visualization** | ❌ | ❌ | ❌ | ✅ |

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

## 🌐 Deployment

### Streamlit Cloud (Free)
1. Push to GitHub
2. Connect at share.streamlit.io
3. Add `HUGGINGFACEHUB_API_TOKEN` in Secrets
4. Deploy!

**Limitations:**
- 1GB RAM (fine for Slice 1-2)
- May struggle with Slice 3-4
- Consider HuggingFace Spaces for advanced features

### HuggingFace Spaces (Better for ML)
1. Create Space at huggingface.co/spaces
2. Upload code
3. Add `HUGGINGFACEHUB_API_TOKEN` secret
4. Automatic deployment

**Benefits:**
- 16GB RAM
- Better for embeddings & graphs
- Free tier available

## 🤝 Contributing

This is a vertical slice architecture:
1. Each slice is fully functional
2. New features add horizontal choices
3. Easy to understand and extend

To add a new feature:
1. Determine which slice it belongs to
2. Update relevant component
3. Add option to `config.py`
4. Update UI dropdown in `app.py`

## 📝 License

MIT License - Feel free to use and modify!

## 🙏 Acknowledgments

- **Gray's Anatomy (1918)** - Public domain medical text
- **OpenAI** - GPT models
- **Anthropic** - Claude models
- **spaCy** - Lightweight NLP for entity extraction
- **NetworkX** - Graph algorithms
- **Streamlit** - Excellent Python web framework

## 📧 Support

Questions? Issues?
- Open a GitHub issue
- Check [KNOWLEDGE_GRAPH_SETUP.md](KNOWLEDGE_GRAPH_SETUP.md) for graph documentation
- Review [config.py](config.py) for all configuration options

---

**Built with ❤️ using Vertical Slice Architecture**