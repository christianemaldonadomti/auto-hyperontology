"""
Auto-Hyperontology Package
Operationalizing n-ary mathematical hyperontologies, OEIS A320228 Ordinal-Cardinal mixing,
and Sabidussi-Vizing prime factorization for local LLM reasoning (Proto-AGI).
"""

from auto_hyperontology.core import HyperNode, HyperEdge, BoundaryNode, Hyperontology
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
from auto_hyperontology.khan_academy import build_khan_academy_hyperontology

__all__ = [
    "HyperRAG",
    "HyperNode",
    "HyperEdge",
    "BoundaryNode",
    "Hyperontology",
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
    "build_khan_academy_hyperontology",
]
