"""
Auto-Hyperontology Package
Operationalizing n-ary mathematical hyperontologies for local LLM reasoning (Proto-AGI).
"""

from auto_hyperontology.core import HyperNode, HyperEdge, BoundaryNode, Hyperontology
from auto_hyperontology.topology import TopologicalIndexer, oeis_a320228_weight
from auto_hyperontology.extractor import AutoHyperontologyExtractor, OllamaClient, greedy_token_bounded_chunk
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
    "AutoHyperontologyExtractor",
    "OllamaClient",
    "greedy_token_bounded_chunk",
    "ProtoAGIReasoner",
    "AMEMetrics",
    "ReasoningResult",
    "build_khan_academy_hyperontology"
]
