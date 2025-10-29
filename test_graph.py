"""
Simple test script for knowledge graph functionality
Run this before using the full Streamlit app to verify graph works
"""
import os
import sys

def test_imports():
    """Test that all required libraries are installed"""
    print("=" * 60)
    print("Testing imports...")
    print("=" * 60)

    spacy_available = False
    networkx_available = False

    try:
        import spacy
        print("✓ spaCy installed")
        spacy_available = True
    except ImportError:
        print("⚠ spaCy not installed (optional - will use pattern-based extraction)")
        print("  Install with: pip install spacy")

    if spacy_available:
        try:
            nlp = spacy.load("en_core_web_sm")
            print("✓ spaCy model 'en_core_web_sm' loaded")
        except OSError:
            print("⚠ spaCy model not found (optional)")
            print("  Install with: python -m spacy download en_core_web_sm")

    try:
        import networkx as nx
        print("✓ NetworkX installed")
        networkx_available = True
    except ImportError:
        print("✗ NetworkX not installed (REQUIRED)")
        print("  Install with: pip install networkx")
        return False

    if not networkx_available:
        print("\n⚠ NetworkX is required for knowledge graph functionality")
        return False

    return True


def test_entity_extraction():
    """Test entity extraction from sample text"""
    print("\n" + "=" * 60)
    print("Testing entity extraction...")
    print("=" * 60)

    from components.graph_rag import GraphRAG

    graph = GraphRAG(mode="entity")

    # Sample anatomical text
    sample_text = """
    The Temporal Bone articulates with the mandible through the temporomandibular joint.
    The Temporal Muscle originates from the Temporal Bone and inserts into the Coronoid
    Process of the mandible. The Internal Carotid Artery enters the skull through the
    Carotid Canal in the Temporal Bone.
    """

    entities = graph._extract_entities(sample_text)

    print(f"\nExtracted {len(entities)} entities:")
    for entity in sorted(entities):
        print(f"  - {entity}")

    # Expected multi-word anatomical terms (our pattern focuses on these)
    expected_any = [
        "Temporal Bone", "Temporal Muscle", "Coronoid Process",
        "Internal Carotid", "Carotid Artery", "Carotid Canal"
    ]

    # Check if we found at least 2 multi-word anatomical entities
    found = sum(1 for exp in expected_any if exp in entities)

    if found >= 2:
        print(f"\n✓ Found {found} multi-word anatomical entities")
        print(f"  Examples: {[e for e in expected_any if e in entities][:3]}")
        return True
    else:
        print(f"\n✗ Only found {found} multi-word anatomical entities")
        print(f"  Expected at least 2 from: {expected_any}")
        return False


def test_graph_building():
    """Test graph building with small dataset"""
    print("\n" + "=" * 60)
    print("Testing graph building...")
    print("=" * 60)

    from components.graph_rag import GraphRAG

    graph = GraphRAG(mode="entity")

    # Small sample chunks (using capitalized anatomical terms as they appear in Gray's Anatomy)
    chunks = [
        "The Temporal Bone articulates with the mandible.",
        "The Temporal Muscle attaches to the Temporal Bone.",
        "The mandible is connected to the Lower Jaw Bone.",
        "The Frontal Bone forms the forehead.",
        "The Parietal Bone is part of the skull."
    ]

    print(f"\nBuilding graph from {len(chunks)} sample chunks...")

    def progress(current, total):
        print(f"  Processing chunk {current}/{total}")

    graph.build_graph(chunks, progress_callback=progress)

    stats = graph.get_stats()
    print(f"\nGraph statistics:")
    print(f"  Entities: {stats['num_entities']}")
    print(f"  Relationships: {stats['num_relationships']}")
    print(f"  Avg connections per entity: {stats['avg_degree']:.2f}")

    if stats['num_entities'] > 0 and stats['num_relationships'] > 0:
        print("\n✓ Graph built successfully")
        return True
    else:
        print("\n✗ Graph building failed")
        return False


def test_context_retrieval():
    """Test context retrieval from graph"""
    print("\n" + "=" * 60)
    print("Testing context retrieval...")
    print("=" * 60)

    from components.graph_rag import GraphRAG

    graph = GraphRAG(mode="entity")

    # Build graph (using capitalized anatomical terms)
    chunks = [
        "The Temporal Bone articulates with the mandible.",
        "The Temporal Muscle attaches to the Temporal Bone.",
        "The mandible connects to the Lower Jaw Bone.",
        "The Zygomatic Arch connects to the Temporal Bone.",
    ]

    graph.build_graph(chunks)

    # Simulated search results
    search_results = [
        {"text": "The Temporal Bone articulates with the mandible.", "score": 0.9}
    ]

    question = "What connects to the Temporal Bone?"

    context = graph.get_context(question, search_results)

    print(f"\nQuestion: {question}")
    print(f"\nGraph context:")
    print(context['formatted_text'])

    # Check if we got any entities (relationships might be 0 if entities don't have neighbors)
    if context['entities'] or (context.get('formatted_text') and len(context['formatted_text']) > 50):
        print(f"\n✓ Context retrieval successful")
        print(f"  Found {len(context['entities'])} entities")
        print(f"  Found {len(context['relationships'])} relationships")
        return True
    else:
        print("\n✗ Context retrieval failed - no entities found")
        return False


def test_persistence():
    """Test saving and loading graph"""
    print("\n" + "=" * 60)
    print("Testing graph persistence...")
    print("=" * 60)

    from components.graph_rag import GraphRAG

    # Build graph (using capitalized anatomical terms)
    graph1 = GraphRAG(mode="entity")
    chunks = [
        "The Temporal Bone articulates with the mandible.",
        "The Frontal Bone forms the forehead."
    ]
    graph1.build_graph(chunks)

    # Save
    test_file = "test_graph.pkl"
    print(f"\nSaving graph to {test_file}...")
    graph1.save(test_file)

    # Load
    graph2 = GraphRAG(mode="entity")
    print(f"Loading graph from {test_file}...")
    success = graph2.load(test_file)

    # Clean up
    if os.path.exists(test_file):
        os.remove(test_file)
        print(f"Cleaned up {test_file}")

    if success and graph2.get_stats()['num_entities'] > 0:
        print("\n✓ Save/load successful")
        return True
    else:
        print("\n✗ Save/load failed")
        return False


def main():
    """Run all tests"""
    print("\n" + "=" * 60)
    print("KNOWLEDGE GRAPH TEST SUITE")
    print("=" * 60)

    results = {}

    # Test 1: Imports
    results['imports'] = test_imports()
    if not results['imports']:
        print("\n" + "=" * 60)
        print("FATAL: Missing dependencies. Cannot continue.")
        print("=" * 60)
        sys.exit(1)

    # Test 2: Entity extraction
    results['entity_extraction'] = test_entity_extraction()

    # Test 3: Graph building
    results['graph_building'] = test_graph_building()

    # Test 4: Context retrieval
    results['context_retrieval'] = test_context_retrieval()

    # Test 5: Persistence
    results['persistence'] = test_persistence()

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {test_name.replace('_', ' ').title()}")

    all_passed = all(results.values())

    print("\n" + "=" * 60)
    if all_passed:
        print("✓ ALL TESTS PASSED")
        print("=" * 60)
        print("\nKnowledge graph is ready!")
        print("\nOptional: For better entity extraction, install spaCy:")
        print("  pip install spacy")
        print("  python -m spacy download en_core_web_sm")
        print("\nRun the Streamlit app:")
        print("  streamlit run app.py")
    else:
        print("✗ SOME TESTS FAILED")
        print("=" * 60)
        print("\nPlease fix the issues above before using the knowledge graph.")
        print("\nNote: spaCy is optional - pattern-based extraction will work without it.")

    print("\n")
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
