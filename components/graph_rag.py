"""
Knowledge Graph RAG Component for Gray's Anatomy Agent

This module implements a lightweight knowledge graph system that extracts entities
and relationships from Gray's Anatomy text chunks and provides graph-based context
retrieval to augment traditional RAG searches.

Design Principles:
- Lightweight: Uses spaCy (small model) + NetworkX (pure Python)
- No heavy ML dependencies (no PyTorch, no transformers)
- Pickle-based persistence (matches existing architecture)
- Toggle-able: Can be disabled without affecting standard RAG
"""

import os
import pickle
import re
from typing import List, Dict, Any, Set, Tuple, Optional
from collections import defaultdict, Counter
import logging

try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    logging.warning("spaCy not installed. Knowledge graph will use fallback entity extraction.")

try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False
    logging.warning("NetworkX not installed. Knowledge graph features disabled.")


class GraphRAG:
    """
    Knowledge Graph-enhanced Retrieval Augmented Generation

    Modes:
    - entity: Extract entities from search results and traverse graph for related entities
    - community: Use community detection to find hierarchically related concepts
    """

    def __init__(self, mode: str = "entity"):
        """
        Initialize GraphRAG component

        Args:
            mode: "entity" or "community" (graph RAG strategy)
        """
        self.mode = mode
        self.graph = None
        self.entity_to_chunks = defaultdict(set)  # entity -> set of chunk indices
        self.chunk_to_entities = defaultdict(set)  # chunk index -> set of entities
        self.communities = {}  # For community detection mode
        self.nlp = None

        # Initialize spaCy if available
        if SPACY_AVAILABLE:
            self._init_spacy()

        # Initialize graph
        if NETWORKX_AVAILABLE:
            self.graph = nx.Graph()

    def _init_spacy(self):
        """Initialize spaCy with small English model"""
        try:
            # Try to load the model
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            # Model not installed, provide helpful message
            logging.warning(
                "spaCy model 'en_core_web_sm' not found. "
                "Install with: python -m spacy download en_core_web_sm"
            )
            self.nlp = None

    def build_graph(self, chunks: List[str], progress_callback=None) -> None:
        """
        Build knowledge graph from text chunks

        Args:
            chunks: List of text chunks from Gray's Anatomy
            progress_callback: Optional callback function(current, total) for progress updates
        """
        if not NETWORKX_AVAILABLE:
            logging.error("NetworkX not available. Cannot build graph.")
            return

        logging.info(f"Building knowledge graph from {len(chunks)} chunks...")

        # Extract entities from each chunk
        for idx, chunk in enumerate(chunks):
            if progress_callback and idx % 100 == 0:
                progress_callback(idx, len(chunks))

            entities = self._extract_entities(chunk)

            # Store bidirectional mappings
            for entity in entities:
                self.entity_to_chunks[entity].add(idx)
                self.chunk_to_entities[idx].add(entity)

                # Add entity to graph
                if not self.graph.has_node(entity):
                    self.graph.add_node(entity, count=1, chunks=[idx])
                else:
                    self.graph.nodes[entity]['count'] += 1
                    self.graph.nodes[entity]['chunks'].append(idx)

            # Create edges between co-occurring entities in same chunk
            entities_list = list(entities)
            for i in range(len(entities_list)):
                for j in range(i + 1, len(entities_list)):
                    entity1, entity2 = entities_list[i], entities_list[j]

                    if self.graph.has_edge(entity1, entity2):
                        self.graph[entity1][entity2]['weight'] += 1
                    else:
                        self.graph.add_edge(entity1, entity2, weight=1, relation='co-occurs')

        if progress_callback:
            progress_callback(len(chunks), len(chunks))

        logging.info(
            f"Graph built: {self.graph.number_of_nodes()} entities, "
            f"{self.graph.number_of_edges()} relationships"
        )

        # Run community detection for community mode
        if self.mode == "community":
            self._detect_communities()

    def _extract_entities(self, text: str) -> Set[str]:
        """
        Extract anatomical entities from text

        Uses spaCy if available, otherwise falls back to pattern-based extraction

        Args:
            text: Text chunk to extract entities from

        Returns:
            Set of entity strings
        """
        entities = set()

        # Method 1: spaCy NER (if available)
        if self.nlp is not None:
            doc = self.nlp(text)
            for ent in doc.ents:
                # Focus on anatomical entities (body parts, organs, etc.)
                if ent.label_ in ['ORG', 'GPE', 'PERSON']:
                    # Skip non-anatomical named entities
                    continue

                # Normalize entity text
                entity_text = self._normalize_entity(ent.text)
                if entity_text:
                    entities.add(entity_text)

        # Method 2: Pattern-based extraction (always runs as supplement)
        # Extract capitalized anatomical terms
        pattern_entities = self._extract_by_patterns(text)
        entities.update(pattern_entities)

        return entities

    def _extract_by_patterns(self, text: str) -> Set[str]:
        """
        Extract anatomical entities using domain-specific patterns

        This is a fallback/supplement to spaCy NER, focusing on anatomical terminology
        """
        entities = set()

        # Pattern 1: Capitalized multi-word anatomical terms
        # e.g., "Temporal Bone", "Inferior Vena Cava"
        # Match 2+ word capitalized phrases (avoid single words)
        capitalized_phrases = re.findall(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})\b', text)
        for phrase in capitalized_phrases:
            # Filter by anatomical keywords
            if self._is_anatomical_term(phrase):
                normalized = self._normalize_entity(phrase)
                if normalized:
                    entities.add(normalized)

        # Pattern 2: Common anatomical suffixes (only if contains anatomical keywords)
        # e.g., words ending in -al, -ous, -ic, -oid (vertebral, nervous, cardiac, thyroid)
        anatomical_suffixes = re.findall(
            r'\b([A-Z][a-z]{5,}(?:al|ous|ic|oid|ar|ary))\b',
            text
        )
        for term in anatomical_suffixes:
            # Only include if looks anatomical
            if self._is_anatomical_term(term):
                normalized = self._normalize_entity(term)
                if normalized:
                    entities.add(normalized)

        # Pattern 3: Medical/anatomical prefixes (capitalized only)
        # e.g., inter-, intra-, sub-, supra-, infra-
        prefix_terms = re.findall(
            r'\b((?:Inter|Intra|Sub|Supra|Infra|Ante|Post|Pre|Retro)[a-z]{4,})\b',
            text
        )
        for term in prefix_terms:
            normalized = self._normalize_entity(term)
            if normalized:
                entities.add(normalized)

        return entities

    def _is_anatomical_term(self, phrase: str) -> bool:
        """Check if phrase is likely an anatomical term"""
        # Common anatomical keywords
        keywords = [
            'bone', 'muscle', 'nerve', 'artery', 'vein', 'ligament', 'tendon',
            'organ', 'gland', 'tissue', 'cavity', 'canal', 'foramen', 'fossa',
            'process', 'tubercle', 'tuberosity', 'spine', 'crest', 'line',
            'surface', 'border', 'angle', 'body', 'head', 'neck', 'shaft',
            'joint', 'cartilage', 'fascia', 'membrane', 'duct', 'valve',
            'node', 'plexus', 'ganglion', 'sinus', 'ventricle', 'atrium',
            'lobe', 'cortex', 'medulla', 'root', 'branch', 'trunk'
        ]

        phrase_lower = phrase.lower()
        return any(keyword in phrase_lower for keyword in keywords)

    def _normalize_entity(self, entity: str) -> Optional[str]:
        """
        Normalize entity text

        Args:
            entity: Raw entity string

        Returns:
            Normalized entity or None if invalid
        """
        # Clean whitespace
        entity = ' '.join(entity.split())

        # Remove leading articles (The, A, An)
        entity = re.sub(r'^(The|A|An)\s+', '', entity)

        # Convert to title case for consistency
        entity = entity.title()

        # Filter out very short or very long entities
        if len(entity) < 3 or len(entity) > 50:
            return None

        # Filter out common non-anatomical words
        stopwords = {
            'The', 'A', 'An', 'And', 'Or', 'But', 'In', 'On', 'At', 'To', 'For',
            'Of', 'With', 'By', 'From', 'As', 'Is', 'Was', 'Are', 'Were',
            'This', 'That', 'These', 'Those', 'It', 'Its'
        }
        if entity in stopwords:
            return None

        return entity

    def _detect_communities(self):
        """Detect communities in graph using Louvain algorithm"""
        if not NETWORKX_AVAILABLE or self.graph is None:
            return

        try:
            # Use Louvain community detection (available in networkx 3.0+)
            from networkx.algorithms import community as nx_community
            communities = nx_community.louvain_communities(self.graph, weight='weight')

            # Store community membership
            for comm_id, comm in enumerate(communities):
                for entity in comm:
                    self.communities[entity] = comm_id

            logging.info(f"Detected {len(communities)} communities in graph")
        except Exception as e:
            logging.warning(f"Community detection failed: {e}")
            self.communities = {}

    def get_context(
        self,
        question: str,
        search_results: List[Dict],
        max_entities: int = 10,
        max_neighbors: int = 5
    ) -> Dict[str, Any]:
        """
        Get knowledge graph context for a question

        Args:
            question: User's question
            search_results: List of search results from standard RAG
            max_entities: Maximum entities to extract from question + search results
            max_neighbors: Maximum neighbor entities to include per entity

        Returns:
            Dictionary with graph context:
            {
                "entities": List of main entities,
                "relationships": List of entity-entity relationships,
                "neighbor_entities": List of related entities found via graph traversal,
                "formatted_text": Formatted text for LLM prompt
            }
        """
        if not NETWORKX_AVAILABLE or self.graph is None:
            return {
                "entities": [],
                "relationships": [],
                "neighbor_entities": [],
                "formatted_text": ""
            }

        # Extract entities from question
        question_entities = self._extract_entities(question)

        # Extract entities from search results
        result_entities = set()
        for result in search_results[:3]:  # Top 3 results only
            result_entities.update(self._extract_entities(result['text']))

        # Combine and limit
        all_entities = list(question_entities.union(result_entities))[:max_entities]

        # Find entities actually in graph
        graph_entities = [e for e in all_entities if self.graph.has_node(e)]

        if not graph_entities:
            return {
                "entities": [],
                "relationships": [],
                "neighbor_entities": [],
                "formatted_text": "No entities found in knowledge graph."
            }

        # Get related entities via graph traversal
        neighbor_entities = []
        relationships = []

        for entity in graph_entities:
            # Get neighbors (connected entities)
            neighbors = list(self.graph.neighbors(entity))

            # Sort by edge weight (co-occurrence frequency)
            neighbors_with_weight = [
                (n, self.graph[entity][n]['weight'])
                for n in neighbors
            ]
            neighbors_with_weight.sort(key=lambda x: x[1], reverse=True)

            # Take top neighbors
            top_neighbors = neighbors_with_weight[:max_neighbors]

            for neighbor, weight in top_neighbors:
                neighbor_entities.append({
                    "entity": neighbor,
                    "relation_to": entity,
                    "strength": weight
                })
                relationships.append({
                    "entity1": entity,
                    "entity2": neighbor,
                    "weight": weight
                })

        # Format as text for LLM
        formatted_text = self._format_graph_context(
            graph_entities,
            neighbor_entities,
            relationships
        )

        return {
            "entities": graph_entities,
            "relationships": relationships,
            "neighbor_entities": neighbor_entities,
            "formatted_text": formatted_text
        }

    def _format_graph_context(
        self,
        entities: List[str],
        neighbors: List[Dict],
        relationships: List[Dict]
    ) -> str:
        """
        Format graph context as readable text for LLM

        Args:
            entities: List of main entities
            neighbors: List of neighbor entity dictionaries
            relationships: List of relationship dictionaries

        Returns:
            Formatted string
        """
        lines = []

        lines.append("=== Knowledge Graph Context ===\n")

        # Main entities
        lines.append("Key Anatomical Entities:")
        for entity in entities:
            count = self.graph.nodes[entity]['count']
            lines.append(f"  - {entity} (mentioned {count} times)")

        lines.append("\nRelated Entities (via co-occurrence):")

        # Group neighbors by their relation_to entity
        neighbors_by_entity = defaultdict(list)
        for neighbor in neighbors:
            neighbors_by_entity[neighbor['relation_to']].append(neighbor)

        for entity, entity_neighbors in neighbors_by_entity.items():
            lines.append(f"  {entity} is related to:")
            for neighbor in entity_neighbors:
                lines.append(
                    f"    - {neighbor['entity']} "
                    f"(co-occurs {neighbor['strength']} times)"
                )

        return "\n".join(lines)

    def save(self, filepath: str) -> None:
        """
        Save graph to disk using pickle

        Args:
            filepath: Path to save file (e.g., indexes/knowledge_graph.pkl)
        """
        if not NETWORKX_AVAILABLE:
            logging.error("Cannot save graph: NetworkX not available")
            return

        # Create directory if needed
        dirname = os.path.dirname(filepath)
        if dirname:  # Only create directory if filepath includes a directory path
            os.makedirs(dirname, exist_ok=True)

        data = {
            'graph': self.graph,
            'entity_to_chunks': dict(self.entity_to_chunks),
            'chunk_to_entities': dict(self.chunk_to_entities),
            'communities': self.communities,
            'mode': self.mode
        }

        with open(filepath, 'wb') as f:
            pickle.dump(data, f)

        logging.info(f"Knowledge graph saved to {filepath}")

    def load(self, filepath: str) -> bool:
        """
        Load graph from disk

        Args:
            filepath: Path to load file

        Returns:
            True if successful, False otherwise
        """
        if not os.path.exists(filepath):
            logging.warning(f"Graph file not found: {filepath}")
            return False

        if not NETWORKX_AVAILABLE:
            logging.error("Cannot load graph: NetworkX not available")
            return False

        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)

            self.graph = data['graph']
            self.entity_to_chunks = defaultdict(set, data['entity_to_chunks'])
            self.chunk_to_entities = defaultdict(set, data['chunk_to_entities'])
            self.communities = data.get('communities', {})
            self.mode = data.get('mode', 'entity')

            logging.info(
                f"Knowledge graph loaded: {self.graph.number_of_nodes()} entities, "
                f"{self.graph.number_of_edges()} relationships"
            )
            return True
        except Exception as e:
            logging.error(f"Failed to load graph: {e}")
            return False

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the knowledge graph"""
        if not NETWORKX_AVAILABLE or self.graph is None:
            return {
                "num_entities": 0,
                "num_relationships": 0,
                "num_communities": 0
            }

        return {
            "num_entities": self.graph.number_of_nodes(),
            "num_relationships": self.graph.number_of_edges(),
            "num_communities": len(set(self.communities.values())) if self.communities else 0,
            "avg_degree": sum(dict(self.graph.degree()).values()) / max(self.graph.number_of_nodes(), 1),
            "mode": self.mode
        }
