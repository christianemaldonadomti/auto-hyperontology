"""
Auto-Hyperontology Package
Universal n-ary mathematical hyperontologies, OEIS A320228 Ordinal-Cardinal mixing,
Sabidussi-Vizing prime factorization, and Proto-AGI v4.0 Forensic Reasoning.
"""

from auto_hyperontology.core import HyperNode, HyperEdge, BoundaryNode, Hyperontology, normalize_node_weights
from auto_hyperontology.topology import (
    TopologicalIndexer,
    oeis_a320228_weight,
    f_oc_retrieval_score,
    prime_weight,
    is_prime,
    PRIMES,
    OEIS_A320228_SEQ,
)
from auto_hyperontology.extractor import (
    AutoHyperontologyExtractor,
    OllamaClient,
    greedy_token_bounded_chunk,
    estimate_bpe_tokens,
    sanitize_json_payload,
)
from auto_hyperontology.reasoner import ProtoAGIReasoner, AMEMetrics, ReasoningResult
from auto_hyperontology.rag import HyperRAG
from auto_hyperontology.benchmark import (
    RealFlatTextRAG,
    evaluate_response_grounding,
    run_live_rag_benchmark,
)
from auto_hyperontology.inspector import (
    inspect_hyperontology,
    print_hyperontology_audit,
    generate_interactive_html,
)

__all__ = [
    "HyperRAG",
    "HyperNode",
    "HyperEdge",
    "BoundaryNode",
    "Hyperontology",
    "normalize_node_weights",
    "TopologicalIndexer",
    "oeis_a320228_weight",
    "f_oc_retrieval_score",
    "prime_weight",
    "is_prime",
    "PRIMES",
    "OEIS_A320228_SEQ",
    "AutoHyperontologyExtractor",
    "OllamaClient",
    "greedy_token_bounded_chunk",
    "estimate_bpe_tokens",
    "sanitize_json_payload",
    "ProtoAGIReasoner",
    "AMEMetrics",
    "ReasoningResult",
    "RealFlatTextRAG",
    "evaluate_response_grounding",
    "run_live_rag_benchmark",
    "inspect_hyperontology",
    "print_hyperontology_audit",
    "generate_interactive_html",
]
