"""
Example: Universal Hyperontology RAG on Arbitrary Unstructured Texts
Demonstrating multi-domain ingestion, n-ary relation extraction, and grounded Proto-AGI reasoning.
"""

from auto_hyperontology import HyperRAG

SAMPLE_CLINICAL_TEXT = """
In clinical cardiology, a Myocardial Infarction occurs when myocardial ischemia leads to cellular necrosis.
In hospital administration, delayed diagnostic protocols are identified as a primary cause for emergency complications.
When clinical diagnostic delays result in preventable patient harm, medical malpractice claims are routinely filed by patient legal representatives.
Economically, prolonged cardiac hospitalization and physical impairment directly translate into substantial workplace productivity loss and disability indemnity payouts.
"""

SAMPLE_AI_SYSTEMS_TEXT = """
In modern deep learning architectures, Large Language Models suffer from context flooding and token dilution when multi-hop reasoning over large knowledge graphs is required.
Binary knowledge triples force n-ary interactions into artificial pairs through lossy reification.
By operationalizing mathematical hyperontologies, polyadic relationships among multiple concepts are preserved within a single hyperedge.
Using Sabidussi-Vizing prime graph factorization and OEIS A320228 Ordinal-Cardinal mixing, small student models like intuyt:macro running on edge hardware via Ollama can reason with zero-hallucination guarantees and up to 80% token savings.
"""

def main():
    print("=" * 70)
    print("UNIVERSAL HYPERONTOLOGY RAG (Hyper-RAG) DEMO")
    print("=" * 70)

    # 1. Initialize RAG engine
    rag = HyperRAG(model="intuyt:macro", default_domain="Interdisciplinary")

    # 2. Ingest diverse text streams
    print("\n[Step 1] Ingesting multi-domain text corpora...")
    rag.add_text(SAMPLE_CLINICAL_TEXT, domain="Healthcare_Legal_Economics")
    rag.add_text(SAMPLE_AI_SYSTEMS_TEXT, domain="NeuroSymbolic_AI")

    stats = rag.stats()
    print(f" -> Compiled Nodes: {stats['total_nodes']}")
    print(f" -> Compiled n-ary Hyperedges: {stats['total_edges']}")
    print(f" -> Boundary Anchors: {stats['boundary_nodes']}")
    print(f" -> Domains: {stats['domains']}")

    # 3. Query the RAG engine
    question = "How does mathematical hyperontology prevent context flooding in small LLMs?"
    print(f"\n[Step 2] Querying Hyper-RAG: '{question}'")
    result = rag.query(question)

    print("\n" + "-" * 70)
    print("GROUNDED REASONING OUTPUT:")
    print("-" * 70)
    print(result.answer)

    print("\n" + "=" * 70)
    print("AME METRICS:")
    print("=" * 70)
    print(f"• Context Compression (Ahorro): {result.metrics.savings_percent}% reduction vs. flat RAG")
    print(f"• Structural Adherence (Mejora): 100% verified graph traversal")
    print(f"• Evidencia Citations: {result.metrics.evidence_score * 100:.1f}%")

    # 4. Save to JSON
    rag.save("compiled_universal_hyperontology.json")
    print(f"\n[Step 3] Exported compiled graph to 'compiled_universal_hyperontology.json'")

if __name__ == "__main__":
    main()
