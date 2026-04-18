# -*- coding: utf-8 -*-
"""
Evaluation module — 5 example queries and expected output descriptions.
Run manually or as part of integration tests.
"""

EXAMPLE_QUERIES = [
    {
        "id": 1,
        "query": "What is the main objective of this document?",
        "expected": "A synthesized statement of the document's primary purpose, citing the introduction section.",
        "citation_pattern": "(Source: Page 1)",
    },
    {
        "id": 2,
        "query": "Summarize the key findings or conclusions.",
        "expected": "A bullet or paragraph summary of conclusions, citing conclusion/summary pages.",
        "citation_pattern": "(Source: Page N)",
    },
    {
        "id": 3,
        "query": "Are there any tables in the document? What do they contain?",
        "expected": "Description of table contents extracted from [TABLE] chunks, with page citations.",
        "citation_pattern": "(Source: Page N)",
    },
    {
        "id": 4,
        "query": "What methodology or approach is described?",
        "expected": "Description of the method/approach from the relevant section, synthesized, not copied.",
        "citation_pattern": "(Source: Page N)",
    },
    {
        "id": 5,
        "query": "List any recommendations or future work mentioned.",
        "expected": "Synthesized list of recommendations from the document's final section.",
        "citation_pattern": "(Source: Page N)",
    },
]


def run_evaluation(vector_store, llm_client, verbose: bool = True) -> list:
    """
    Run all 5 example queries and return results.
    Requires an indexed VectorStore and a configured LLMClient.
    """
    from qa.qa_engine import answer_query

    results = []
    for item in EXAMPLE_QUERIES:
        try:
            retrieved = vector_store.search(item["query"], top_k=5)
            answer, citations = answer_query(item["query"], retrieved, llm_client)
            ok = bool(answer) and bool(citations)
            results.append(
                {
                    "id": item["id"],
                    "query": item["query"],
                    "answer": answer,
                    "citations": citations,
                    "passed": ok,
                }
            )
            if verbose:
                status = "✓ PASS" if ok else "✗ FAIL"
                print(f"\n[Q{item['id']}] {status}")
                print(f"Query   : {item['query']}")
                print(f"Answer  : {answer[:200]}...")
                print(f"Citations: {citations}")
        except Exception as e:
            results.append(
                {"id": item["id"], "query": item["query"], "error": str(e), "passed": False}
            )
            if verbose:
                print(f"\n[Q{item['id']}] ✗ ERROR: {e}")

    passed = sum(1 for r in results if r.get("passed"))
    if verbose:
        print(f"\n=== Evaluation: {passed}/{len(EXAMPLE_QUERIES)} passed ===")

    return results


if __name__ == "__main__":
    print("Load a document first, then call run_evaluation(vector_store, llm_client).")
    print("\nExample queries:")
    for q in EXAMPLE_QUERIES:
        print(f"  {q['id']}. {q['query']}")
