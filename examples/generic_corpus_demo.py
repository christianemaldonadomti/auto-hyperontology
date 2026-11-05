"""
Example: Universal Hyperontology RAG (Hyper-RAG) & OEIS A320228 Ordinal-Cardinal Mixing
Demonstrates multi-domain text ingestion, exact F_OC retrieval scoring, RAPTOR Cardinality Gating,
Prime Stride Sparse Attention, SA-MLP Distillation Curriculum, WebGL Reification, and GreenAGI Metrics.
"""

from auto_hyperontology import HyperRAG, build_khan_academy_hyperontology

SAMPLE_CLINICAL_LEGAL_TEXT = """
In Clinical Cardiology, a Myocardial Infarction occurs when Coronary Ischemia triggers Cellular Necrosis.
In Hospital Administration, Delayed Diagnostic Protocols are documented as a primary driver of preventable clinical complications.
When Diagnostic Delays cause severe patient injury, Medical Malpractice Claims are filed under Healthcare Jurisprudence.
In Health Economics, long-term post-cardiac disability induces Workplace Productivity Loss and sovereign healthcare expenditure spikes.
"""

SAMPLE_NEUROSYMBOLIC_AI_TEXT = """
In Large Language Models, Unconstrained Context Windows suffer from Context Flooding and Token Dilution during multi-hop reasoning.
Traditional Knowledge Graphs rely on Binary Triples that enforce lossy reification and expose systems to LogicPoison attacks.
Mathematical Hyperontologies model polyadic interactions natively via n-ary Hyperedges and Basic Formal Ontology categories.
By applying Sabidussi-Vizing Prime Factorization and OEIS A320228 Ordinal-Cardinal Mixing, compact models like intuyt:macro achieve zero-hallucination reasoning on edge hardware.
"""


def main():
    print("=" * 80)
    print("AUTO-HYPERONTOLOGY: FULL ORDINAL-CARDINAL MIXING & PROTO-AGI VERIFICATION")
    print("=" * 80)

    # 1. Initialize Universal HyperRAG with Reference + Raw Corpus Ingestion
    base_hyper = build_khan_academy_hyperontology()
    rag = HyperRAG(
        model="intuyt:macro",
        hyperontology=base_hyper,
        default_domain="Interdisciplinary",
        theta_max=3000,
        alpha=0.45,
        beta=0.55
    )

    print("\n[1] Ingesting Arbitrary Multi-Domain Unstructured Texts (DyHSL + RAPTOR Layers)...")
    rag.add_text(SAMPLE_CLINICAL_LEGAL_TEXT, domain="Medicine_Law_Economics")
    rag.add_text(SAMPLE_NEUROSYMBOLIC_AI_TEXT, domain="NeuroSymbolic_AI")

    stats = rag.stats()
    print(f" -> Total Vertices |X|: {stats['total_nodes']}")
    print(f" -> Total n-ary Hyperedges |E|: {stats['total_edges']}")
    print(f" -> Active Boundary Nodes (Nodos Frontera): {stats['boundary_nodes']}")
    print(f" -> Coprime Adjacency Verified (gcd(L(u), L(v)) = 1): {stats['coprime_labeling_verified']}")
    print(f" -> OEIS A320228 4-Regularity Check: {stats['a320228_regularity']}")
    print(f" -> 3-Phase Distillation Curriculum: {stats['curriculum_counts']}")

    # 2. Execute High-Cardinality Cross-Domain Query
    question = "How does the concept of derivative bridge calculus, kinematics in physics, and gradient descent in neural networks?"
    print(f"\n[2] Executing 5-Stage Proto-AGI Query:\n    '{question}'")
    res = rag.query(question, max_edges=4)

    print("\n" + "-" * 80)
    print("ADAPTIVE-RAG & ORDINAL-CARDINAL MIXING DIAGNOSTICS:")
    print("-" * 80)
    print(f"• RAPTOR Cardinality Gate: {res.raptor_gate_info}")
    print(f"• Activated Seed Nodes: {res.seed_nodes}")
    print(f"• Boundary Nodes Activated: {res.boundary_nodes_activated}")
    print(f"• F_OC Retrieval Scores R_score(H_e): {res.f_oc_scores}")

    print("\n" + res.minimal_response_context)

    print("\n" + "-" * 80)
    print("GROUNDED PROTO-AGI SYNTHESIS:")
    print("-" * 80)
    print(res.answer)

    print("\n" + "=" * 80)
    print("AME & GREEN-AGI ECOLOGICAL FOOTPRINT METRICS:")
    print("=" * 80)
    print(f"• Ahorro (A) Context Compression: {res.metrics.savings_percent}% reduction ({res.metrics.context_tokens_used} tok vs {res.metrics.context_tokens_baseline} flat baseline)")
    print(f"• Prime Stride Sparse Attention (strides 2, 3, 5): {res.metrics.prime_stride_compressed_tokens} tokens attended")
    print(f"• GreenAGI Energy (3B MoE vs 400B Dense): {res.metrics.energy_wh_3b} Wh vs {res.metrics.energy_wh_400b_baseline} Wh (~{res.metrics.compute_reduction_factor}x less energy)")
    print(f"• GreenAGI Carbon Emissions: {res.metrics.co2_kg_3b} kg CO2 vs {res.metrics.co2_kg_400b_baseline} kg CO2")
    print(f"• Evidencia (E) Traceability: {res.metrics.evidence_score * 100:.1f}% ({len(res.metrics.grounded_hyperedges)} hyperedges cited)")
    print("=" * 80)


if __name__ == "__main__":
    main()
