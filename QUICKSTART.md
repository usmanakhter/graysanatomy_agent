# Quick Start Guide - Slice 4 (Knowledge Graph)

## 5-Minute Setup

### 1. Install Dependencies

```bash
# Option A: Automated (Recommended)
chmod +x setup_slice4.sh
./setup_slice4.sh

# Option B: Manual
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python test_graph.py
```

### 2. Set API Keys

Create `.env` file:
```bash
# Choose at least one:
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
```

Get keys:
- OpenAI: https://platform.openai.com/api-keys
- Anthropic: https://console.anthropic.com/settings/keys

### 3. Run App

```bash
streamlit run app.py
```

## First Use

### Step 1: Configure in Sidebar

**Language Model**:
- gpt-4o-mini (fast, cheap)
- gpt-4o (best quality)
- claude-3-5-sonnet (excellent reasoning)
- claude-3-5-haiku (fast, affordable)

**Search Strategy**:
- Lexical (BM25 - keyword matching)
- Semantic (TF-IDF - meaning-based)
- Hybrid (Best of both)

**Knowledge Graph** ⭐ NEW:
- **None** - Standard RAG (fastest)
- **Entity Graph** - Find related anatomical terms
- **Community Graph** - Hierarchical concept grouping

### Step 2: Ask a Question

Try these examples:

**Good for Standard RAG (None)**:
- "What are the main bones of the skull?"
- "Describe the function of ligaments"
- "How does the digestive system work?"

**Good for Entity Graph**:
- "What connects to the temporal bone?"
- "Which muscles attach to the humerus?"
- "What vessels supply the liver?"

**Good for Community Graph**:
- "Explain the skeletal system of the hand"
- "What are the components of the cardiovascular system?"
- "Describe the nervous pathways in the leg"

### Step 3: First Query Timing

**Standard RAG (None)**:
- First time: ~1-3 min (builds BM25 index)
- After: 1-3 seconds

**Entity/Community Graph**:
- First time: ~3-8 min (builds BM25 + TF-IDF + Graph)
- After: 1-3 seconds

All indexes cached to disk - subsequent runs instant!

## Understanding the Results

### Standard RAG Output

```
Question: What are the bones of the skull?

Answer: [Based on retrieved text chunks]

Sources: 5 text excerpts shown
```

### Graph RAG Output

```
Question: What connects to the temporal bone?

Answer: [Based on text chunks + graph relationships]

Sources: 5 text excerpts shown

Graph Context (visible in logs):
- Key Entities: Temporal Bone, Mandible, Zygomatic Arch
- Relationships:
  - Temporal Bone ↔ Mandible (15 co-occurrences)
  - Temporal Bone ↔ Zygomatic Arch (8 co-occurrences)
```

The LLM sees both the text chunks AND the graph relationships!

## Troubleshooting

### Import Errors

```bash
# spaCy not found
pip install spacy>=3.7.0

# NetworkX not found
pip install networkx>=3.1

# spaCy model not found
python -m spacy download en_core_web_sm
```

### API Key Issues

```bash
# Missing key error
echo 'OPENAI_API_KEY=sk-...' >> .env

# Or enter in UI sidebar (temporary)
```

### Graph Building Slow

Expected! First time only:
- ~10,000 text chunks
- Entity extraction + graph construction
- ~2-5 minutes
- Cached to `indexes/knowledge_graph.pkl`
- Subsequent loads: <1 second

### Graph Not Helping

- Try entity-rich questions (mentions specific anatomy)
- Graph works best for relationship queries
- May not help for abstract/conceptual questions
- Try different graph modes (entity vs community)

## Performance Tips

### Fastest Setup
- LLM: gpt-4o-mini or claude-3-5-haiku
- Search: Lexical (BM25)
- Embedding: N/A (lexical doesn't need it)
- Graph: None

### Best Quality
- LLM: gpt-4o or claude-3-5-sonnet
- Search: Hybrid
- Embedding: tfidf (free, fast) or openai-small (better)
- Graph: Entity or Community

### Balanced
- LLM: gpt-4o-mini
- Search: Hybrid
- Embedding: tfidf
- Graph: Entity

## Understanding Graph Modes

### None (Default)
- Standard RAG
- No graph overhead
- Use for general questions

### Entity Graph
- Extracts entities from question + results
- Finds related entities via graph traversal
- Shows co-occurrence relationships
- Best for: "What connects to X?" questions

### Community Graph
- Same as Entity Graph
- Plus: Louvain community detection
- Groups related concepts hierarchically
- Best for: "Explain the X system" questions

## File Structure After Setup

```
graysanatomy_agent/
├── grays_anatomy.txt          ← Downloaded (~5MB)
├── indexes/
│   ├── bm25_index.pkl        ← BM25 (~10MB)
│   ├── vector_store/
│   │   ├── tfidf_embeddings.pkl  ← TF-IDF (~10MB)
│   │   └── tfidf_chunks.pkl      ← Chunks (~5MB)
│   └── knowledge_graph.pkl   ← Graph (~20-50MB) ⭐ NEW
└── .env                       ← Your API keys
```

## Common Workflows

### Workflow 1: Quick Answer (No Graph)
1. Select "None" for Knowledge Graph
2. Use "Lexical" or "Hybrid" search
3. Pick fast LLM (gpt-4o-mini)
4. Ask question
5. Get answer in 1-2 seconds

### Workflow 2: Entity Research (With Graph)
1. Select "Entity Graph" for Knowledge Graph
2. Use "Hybrid" search
3. Pick quality LLM (claude-3-5-sonnet)
4. Ask entity-rich question
5. Get enriched answer with relationships

### Workflow 3: System Exploration (Community Graph)
1. Select "Community Graph" for Knowledge Graph
2. Use "Hybrid" search
3. Pick quality LLM (gpt-4o)
4. Ask system-level question
5. Get hierarchical context

## Next Steps

- Read [KNOWLEDGE_GRAPH_SETUP.md](KNOWLEDGE_GRAPH_SETUP.md) for details
- Review [SLICE4_SUMMARY.md](SLICE4_SUMMARY.md) for architecture
- Check [config.py](config.py) for customization options
- Try different combinations of settings
- Compare results with/without graph

## Getting Help

- Run `python test_graph.py` to verify setup
- Check console logs for graph building progress
- Review [README.md](README.md) for full documentation
- Open GitHub issue for bugs

---

**You're ready to go!** 🚀

Try asking: "What are the sympathetic efferent fibers?" with Entity Graph enabled!
