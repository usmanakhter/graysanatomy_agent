"""
Test script to diagnose search quality issues
"""
import sys

def test_search():
    """Test if search retrieves relevant chunks"""
    print("=" * 60)
    print("TESTING SEARCH QUALITY")
    print("=" * 60)

    from components.search import SearchEngine

    # Test question
    question = "What are the bones of the skull?"
    print(f"\nQuestion: {question}")

    # Test different strategies
    strategies = ["lexical"]

    for strategy in strategies:
        print(f"\n{'=' * 60}")
        print(f"Testing {strategy.upper()} search")
        print(f"{'=' * 60}")

        try:
            search_engine = SearchEngine(
                strategy=strategy,
                embedding_model="tfidf" if strategy in ["semantic", "hybrid"] else None,
                top_k=5
            )

            results = search_engine.search(question)

            print(f"\nRetrieved {len(results)} results:")
            print()

            for i, result in enumerate(results, 1):
                print(f"Result {i} ({result['method']} - Score: {result['score']:.4f}):")
                print(f"  Text preview: {result['text'][:200]}...")
                print()

                # Check if the result is relevant
                question_terms = set(question.lower().split())
                result_terms = set(result['text'].lower().split())
                overlap = question_terms.intersection(result_terms)
                print(f"  Term overlap: {overlap}")
                print()

            # Check if any results mention key terms
            key_terms = ["skull", "bone", "cranium", "temporal", "parietal", "frontal"]
            for i, result in enumerate(results, 1):
                found_terms = [term for term in key_terms if term in result['text'].lower()]
                if found_terms:
                    print(f"Result {i} contains: {found_terms}")

            if not any(any(term in result['text'].lower() for term in key_terms) for result in results):
                print("\n⚠ WARNING: No results contain expected skull-related terms!")
                print("This suggests the search index may be corrupted or empty.")

                # Check index state
                print(f"\nIndex diagnostics:")
                print(f"  Total chunks in index: {len(search_engine.chunks)}")
                if len(search_engine.chunks) > 0:
                    print(f"  First chunk preview: {search_engine.chunks[0][:100]}...")
                    print(f"  Last chunk preview: {search_engine.chunks[-1][:100]}...")

                    # Sample search through chunks manually
                    skull_chunks = [c for c in search_engine.chunks if 'skull' in c.lower()]
                    print(f"\n  Chunks containing 'skull': {len(skull_chunks)}")
                    if skull_chunks:
                        print(f"  Example: {skull_chunks[0][:200]}...")

        except Exception as e:
            print(f"\n✗ Error with {strategy} search: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    test_search()
