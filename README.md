# 🧠 Gray's Anatomy AI Assistant

Intelligent anatomy question-answering system using RAG (Retrieval-Augmented Generation) architecture.

## 🎯 Current Status: **Slice 1 Complete** ✅

### What's Working:
- ✅ **Question Answering**: Ask any anatomy question
- ✅ **BM25 Search**: Fast keyword-based retrieval
- ✅ **Mistral-7B LLM**: Free HuggingFace API
- ✅ **Source Attribution**: See which text chunks were used
- ✅ **Performance Metrics**: Track response times
- ✅ **100% Free**: No paid APIs

## 🚀 Quick Start

### 1. Clone Repository
```bash
git clone https://github.com/yourusername/graysanatomy_agent.git
cd graysanatomy_agent
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Set API Token
Get a free token at https://huggingface.co/settings/tokens

**Local Development:**
```bash
```

**Streamlit Cloud:**
Add to Settings → Secrets:
```toml
```

### 4. Run Application
```bash
streamlit run app.py
```

### 5. First Run
- Downloads Gray's Anatomy text (~30 seconds)
- Builds BM25 search index (~2 minutes)
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

### ✅ Slice 1: Core System (CURRENT)
- [x] Basic RAG pipeline
- [x] BM25 lexical search
- [x] Single LLM (Mistral-7B)
- [x] Streamlit UI
- [x] Source attribution

### 🔄 Slice 2: Multiple LLMs (NEXT)
- [ ] Add 5 more LLM options
- [ ] Mistral-7B, Mixtral-8x7B
- [ ] Zephyr-7B, Llama-3.1-8B
- [ ] Phi-3-Mini, BioMedLM
- [ ] Dropdown selector in UI

### 📋 Slice 3: Advanced Search
- [ ] Semantic search (FAISS + embeddings)
- [ ] 5 embedding model options
- [ ] Hybrid search (BM25 + Vector + RRF)
- [ ] Search strategy selector

### 🕸️ Slice 4: Knowledge Graph
- [ ] Entity extraction (spaCy)
- [ ] Relationship detection
- [ ] Simple graph (NetworkX)
- [ ] Full GraphRAG (community detection)
- [ ] Graph visualization

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

```python
# Ask a question
"What are the main bones of the skull?"

# System automatically:
1. Searches Gray's Anatomy text (BM25)
2. Retrieves top 5 relevant chunks
3. Sends to Mistral-7B with context
4. Returns comprehensive answer
5. Shows sources used
```

## 🔧 Configuration

Edit `config.py` to customize:

```python
# Chunking
CHUNK_SIZE = 1000          # Characters per chunk
CHUNK_OVERLAP = 200        # Overlap between chunks

# Retrieval
TOP_K_RESULTS = 5          # Number of chunks to retrieve

# LLM (Slice 2+)
DEFAULT_LLM = "mistral-7b"

# Search (Slice 3+)
DEFAULT_SEARCH_STRATEGY = "hybrid"
```

## 📊 Performance

**Slice 1 Metrics:**
- First query: ~5-7 seconds (including model warmup)
- Subsequent queries: ~2-3 seconds
- Index build time: ~2 minutes (one-time)
- Memory usage: ~300 MB

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
- **HuggingFace** - Free LLM inference API
- **Streamlit** - Excellent Python web framework

## 📧 Support

Questions? Issues? 
- Open a GitHub issue
- Check the [ARCHITECTURE.md](ARCHITECTURE.md) for design details

---

**Built with ❤️ using Vertical Slice Architecture**