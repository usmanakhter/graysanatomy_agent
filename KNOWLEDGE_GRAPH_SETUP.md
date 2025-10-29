# Knowledge Graph Setup Guide

## Overview

Slice 4 adds a **toggle-able knowledge graph component** that enhances the existing RAG system with entity and relationship extraction from Gray's Anatomy text.

## Features

### 3 Graph Modes

1. **None** (default)
   - Standard RAG (no graph)
   - Fastest performance
   - Uses existing search strategies (lexical/semantic/hybrid)

2. **Entity Graph**
   - Extracts anatomical entities from text
   - Builds co-occurrence graph
   - Traverses graph to find related entities
   - Adds entity context to LLM prompts

3. **Community Graph**
   - Same as Entity Graph
   - Plus: Community detection using Louvain algorithm
   - Groups related anatomical concepts hierarchically
   - Useful for understanding broader anatomical systems

## Installation

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

New dependencies for knowledge graph:
- **spaCy** (>=3.7.0): Lightweight NLP for entity extraction (~15MB)
- **NetworkX** (>=3.1): Pure Python graph library

### 2. Download spaCy Model

```bash
python -m spacy download en_core_web_sm
```

This downloads a small English model (~15MB) for entity recognition.

## Usage

### Via Streamlit UI

1. Start the app:
   ```bash
   streamlit run app.py
   ```

2. In the sidebar, under "🕸️ Knowledge Graph", select:
   - **None** - Standard RAG
   - **Entity Graph** - Entity-based graph traversal
   - **Community Graph** - Community detection mode

3. First query with graph enabled will:
   - Build knowledge graph (~2-5 minutes)
   - Save to `indexes/knowledge_graph.pkl`
   - Subsequent queries use cached graph (instant)

### How It Works

#### Entity Extraction

The graph component extracts anatomical entities using two methods:

1. **spaCy NER**: Uses trained model to identify entities
2. **Pattern-based**: Regex patterns for anatomical terms
   - Capitalized multi-word terms (e.g., "Temporal Bone")
   - Anatomical suffixes (-al, -ous, -ic, -oid)
   - Medical prefixes (inter-, intra-, sub-, supra-)
   - Common anatomical keywords (bone, muscle, nerve, etc.)

#### Graph Construction

- **Nodes**: Unique anatomical entities
- **Edges**: Co-occurrence relationships (entities in same text chunk)
- **Weights**: Frequency of co-occurrence

Example:
```
"The temporal bone articulates with the mandible"
→ Entities: ["Temporal Bone", "Mandible"]
→ Edge: Temporal Bone ←→ Mandible (weight: 1)
```

#### Graph Querying

When a user asks a question:

1. Extract entities from question
2. Extract entities from top search results
3. Find these entities in graph
4. Traverse to neighboring entities (connected by co-occurrence)
5. Return entity relationships as additional context
6. LLM synthesizes answer using both text chunks + graph context

#### Community Detection (Community mode only)

Uses Louvain algorithm to detect communities of closely related entities:
- Groups entities that frequently co-occur
- Helps identify anatomical systems (e.g., cardiovascular, skeletal)
- Provides hierarchical understanding

## Architecture Integration

### Files Modified

1. **config.py**
   - Added `KNOWLEDGE_GRAPH_OPTIONS` with 3 modes
   - Added `GRAPH_STORE_FILE` path

2. **app.py**
   - Added knowledge graph selector dropdown
   - Added graph status indicator
   - Updated footer to reflect Slice 4

3. **orchestrator.py**
   - Added `_get_graph_context()` method
   - Lazy loads graph engine
   - Builds/loads graph on first use

### New Files

4. **components/graph_rag.py**
   - `GraphRAG` class
   - Entity extraction (spaCy + patterns)
   - Graph construction (NetworkX)
   - Graph querying and context retrieval
   - Pickle-based persistence

5. **KNOWLEDGE_GRAPH_SETUP.md**
   - This documentation file

## Performance

### Build Time (One-time)
- ~2-5 minutes on first query with graph enabled
- ~10,000 chunks × entity extraction + graph construction

### Query Time (After build)
- Graph loading: <1 second (from pickle)
- Context retrieval: <100ms
- Negligible overhead vs. standard RAG

### Memory Usage
- Graph size: ~20-50MB (stored on disk)
- Runtime memory: ~50-100MB additional
- Total system: Still works on 8GB RAM

### Disk Storage
```
indexes/
├── bm25_index.pkl              (~10MB)
├── vector_store/
│   ├── tfidf_embeddings.pkl   (~10MB)
│   └── tfidf_chunks.pkl       (~5MB)
└── knowledge_graph.pkl         (~20-50MB)  ← NEW
```

## Example Queries

### Standard RAG (None)
```
Q: What are the bones of the skull?
A: [Uses only search results]
```

### Entity Graph
```
Q: What are the bones of the skull?

Search Results: [Text chunks about skull bones]

Graph Context:
  Key Entities: Temporal Bone, Parietal Bone, Frontal Bone
  Related Entities:
    - Temporal Bone → Mandible (co-occurs 15 times)
    - Temporal Bone → Zygomatic Arch (co-occurs 8 times)
    - Parietal Bone → Occipital Bone (co-occurs 12 times)

A: [Synthesizes from both text chunks AND graph relationships]
```

### Community Graph
```
Same as Entity Graph, plus:

Communities:
  - Community 1: Skull bones (Temporal, Parietal, Frontal, etc.)
  - Community 2: Facial bones (Mandible, Maxilla, Zygomatic, etc.)
  - Community 3: Vertebrae (Cervical, Thoracic, Lumbar, etc.)
```

## Design Principles

### ✅ Maintains Existing Architecture
- No changes to search strategies (lexical/semantic/hybrid)
- No changes to LLM providers (OpenAI/Claude)
- No changes to embedding models (TF-IDF/OpenAI)
- Knowledge graph is purely **additive**

### ✅ Lightweight
- spaCy small model (~15MB)
- NetworkX (pure Python, no C dependencies)
- No PyTorch or heavy ML frameworks
- Pattern-based fallback if spaCy unavailable

### ✅ Toggle-able
- Default: "none" (standard RAG)
- Users opt-in to graph features
- Zero overhead when disabled
- Lazy loading pattern

### ✅ Persistent
- Graph built once, cached to disk
- Pickle-based (matches existing indexes)
- Fast loading (<1 second)

### ✅ Vertical Slice
- Self-contained feature
- Doesn't break existing functionality
- Easy to disable/remove if needed

## Limitations

### Current Scope
- **Entity extraction is domain-specific**: Focused on anatomical terms
- **No semantic entity resolution**: "temporal bone" ≠ "Temporal Bone" (case-sensitive)
- **Co-occurrence only**: Relationships are based on co-occurrence, not semantic understanding
- **No entity disambiguation**: "head" (skull) vs "head" (muscle origin)

### Future Enhancements (Not in Slice 4)
- Semantic entity embeddings
- Entity linking and disambiguation
- Relationship type classification (part-of, connects-to, etc.)
- Graph visualization in UI
- Subgraph extraction for specific anatomical systems

## Troubleshooting

### spaCy model not found
```bash
python -m spacy download en_core_web_sm
```

### NetworkX import error
```bash
pip install networkx>=3.1
```

### Graph building is slow
- Expected: 2-5 minutes for ~10,000 chunks
- Only happens once (cached afterward)
- Progress printed to console

### Graph not providing useful context
- Try different graph modes (entity vs community)
- Graph works best for entity-rich queries
- May not help for abstract/conceptual questions

### Memory issues
- Reduce chunk size in config.py
- Use "none" mode to disable graph
- Graph adds ~50-100MB to memory usage

## Technical Details

### Entity Extraction Algorithm

```python
def _extract_entities(text):
    entities = set()

    # Method 1: spaCy NER
    if spacy_available:
        doc = nlp(text)
        for ent in doc.ents:
            entities.add(normalize(ent.text))

    # Method 2: Pattern-based (always runs)
    # - Capitalized phrases (2-4 words)
    # - Anatomical suffixes (-al, -ous, -ic, etc.)
    # - Medical prefixes (inter-, sub-, supra-, etc.)
    entities.update(extract_by_patterns(text))

    return entities
```

### Graph Construction Algorithm

```python
def build_graph(chunks):
    for idx, chunk in enumerate(chunks):
        entities = extract_entities(chunk)

        # Add nodes
        for entity in entities:
            graph.add_node(entity, count=1, chunks=[idx])

        # Add edges (co-occurrence)
        for entity1 in entities:
            for entity2 in entities:
                if entity1 != entity2:
                    graph.add_edge(entity1, entity2, weight=1)
```

### Context Retrieval Algorithm

```python
def get_context(question, search_results):
    # Extract entities from question
    question_entities = extract_entities(question)

    # Extract entities from search results
    result_entities = extract_entities(search_results)

    # Find in graph
    graph_entities = [e for e in all_entities if graph.has_node(e)]

    # Traverse to neighbors
    neighbors = []
    for entity in graph_entities:
        for neighbor in graph.neighbors(entity):
            weight = graph[entity][neighbor]['weight']
            neighbors.append({
                'entity': neighbor,
                'related_to': entity,
                'strength': weight
            })

    # Format as text for LLM
    return format_graph_context(graph_entities, neighbors)
```

## Comparison: Standard RAG vs Graph RAG

| Feature | Standard RAG | Graph RAG (Entity) | Graph RAG (Community) |
|---------|-------------|-------------------|----------------------|
| **Search** | BM25/Semantic/Hybrid | Same | Same |
| **Context** | Text chunks only | Text + entity graph | Text + entity + communities |
| **Relationships** | Implicit in text | Explicit co-occurrence | Explicit + hierarchical |
| **Build time** | ~30s (BM25) | ~2-5 min | ~2-5 min |
| **Query time** | ~1-3s | ~1-3s | ~1-3s |
| **Memory** | ~100MB | ~150MB | ~150MB |
| **Best for** | All queries | Entity-rich queries | System-level questions |

## Example Use Cases

### Entity Graph Helps
- "What connects the temporal bone to the mandible?"
- "Which muscles attach to the humerus?"
- "What vessels supply the liver?"

### Community Graph Helps
- "Explain the skeletal system of the hand"
- "What are the components of the cardiovascular system?"
- "Describe the nervous pathways in the leg"

### Standard RAG Better
- "Why does inflammation occur?" (conceptual)
- "What is the function of ligaments?" (general definition)
- "How does digestion work?" (process explanation)

## Credits

**Architecture**: Vertical slice design by graysanatomy_agent
**Knowledge Graph**: Slice 4 implementation
**NLP**: spaCy (Explosion AI)
**Graph Library**: NetworkX (NetworkX Developers)
