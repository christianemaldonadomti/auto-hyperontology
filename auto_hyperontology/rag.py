"""
Auto-Hyperontology RAG (Hyper-RAG) Facade
Provides a drop-in, 3-line Python API to ingest any document corpus, compile an
n-ary hyperontology, and execute structure-grounded retrieval and reasoning with local LLMs.
"""

import os
import json
from typing import List, Dict, Any, Optional, Union
from auto_hyperontology.core import Hyperontology, HyperNode, HyperEdge, BoundaryNode
from auto_hyperontology.extractor import AutoHyperontologyExtractor, OllamaClient
from auto_hyperontology.reasoner import ProtoAGIReasoner, ReasoningResult, AMEMetrics
from auto_hyperontology.topology import TopologicalIndexer


class HyperRAG:
    """
    Universal Hyperontology RAG Engine.
    
    Example:
        >>> from auto_hyperontology import HyperRAG
        >>> rag = HyperRAG(model="intuyt:macro")
        >>> rag.add_text("Quantum entanglement enables instantaneous state correlation...")
        >>> result = rag.query("How does entanglement affect communications?")
        >>> print(result.answer)
        >>> print(f"Context Tokens Saved: {result.metrics.savings_percent}%")
    """
    def __init__(
        self,
        model: str = "intuyt:macro",
        base_url: str = "http://localhost:11434",
        hyperontology: Optional[Hyperontology] = None,
        default_domain: str = "General",
        temperature: float = 0.1
    ):
        self.model = model
        self.base_url = base_url
        self.default_domain = default_domain
        self.client = OllamaClient(base_url=base_url, model=model, temperature=temperature)
        self.hyper = hyperontology or Hyperontology(name="UserHyperontology")
        self.extractor = AutoHyperontologyExtractor(ollama_client=self.client)
        self.reasoner = ProtoAGIReasoner(hyperontology=self.hyper, ollama_client=self.client)

    def add_text(self, text: str, domain: Optional[str] = None) -> "HyperRAG":
        """Ingests arbitrary text, parses into bounded chunks, and compiles into the hyperontology."""
        target_domain = domain or self.default_domain
        extracted = self.extractor.extract_from_text(text, default_domain=target_domain)
        self._merge_hyperontology(extracted)
        # Refresh reasoner and indexer
        self.reasoner = ProtoAGIReasoner(hyperontology=self.hyper, ollama_client=self.client)
        return self

    def add_documents(self, documents: List[str], domain: Optional[str] = None) -> "HyperRAG":
        """Ingests a list of document strings."""
        for doc in documents:
            if doc.strip():
                self.add_text(doc, domain=domain)
        return self

    def add_file(self, filepath: str, domain: Optional[str] = None) -> "HyperRAG":
        """Ingests a text file from disk."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
        return self.add_text(content, domain=domain)

    def add_directory(self, dirpath: str, extensions: Tuple[str, ...] = (".txt", ".md", ".json")) -> "HyperRAG":
        """Ingests all matching text files from a directory."""
        if not os.path.isdir(dirpath):
            raise NotADirectoryError(f"Directory not found: {dirpath}")
        for root, _, files in os.walk(dirpath):
            for file in files:
                if any(file.endswith(ext) for ext in extensions):
                    full_path = os.path.join(root, file)
                    domain_name = os.path.basename(root) if root != dirpath else self.default_domain
                    self.add_file(full_path, domain=domain_name)
        return self

    def query(self, question: str, max_edges: int = 4) -> ReasoningResult:
        """
        Executes the 5-stage Proto-AGI reasoning pipeline over the compiled hyperontology.
        Returns a ReasoningResult containing the grounded answer, minimal proof subgraph, and AME metrics.
        """
        return self.reasoner.answer_question(question, max_edges=max_edges)

    def save(self, filepath: str):
        """Exports the compiled hyperontology to a JSON file."""
        self.hyper.save_json(filepath)

    @classmethod
    def load(
        cls, 
        filepath: str, 
        model: str = "intuyt:macro", 
        base_url: str = "http://localhost:11434"
    ) -> "HyperRAG":
        """Loads a previously compiled hyperontology from JSON."""
        hyper = Hyperontology.load_json(filepath)
        return cls(model=model, base_url=base_url, hyperontology=hyper)

    def stats(self) -> Dict[str, Any]:
        """Returns structural statistics of the active hyperontology."""
        return {
            "total_nodes": len(self.hyper.nodes),
            "total_edges": len(self.hyper.edges),
            "boundary_nodes": len(self.hyper.boundary_nodes),
            "domains": list(self.hyper.domain_to_nodes.keys()),
            "categories": list(self.hyper.category_to_nodes.keys())
        }

    def _merge_hyperontology(self, other: Hyperontology):
        for node in other.nodes.values():
            existing = self.hyper.get_node(node.id)
            if existing:
                for d in node.domains:
                    if d not in existing.domains:
                        existing.domains.append(d)
                for c in node.categories:
                    if c not in existing.categories:
                        existing.categories.append(c)
                if existing.is_boundary:
                    self.hyper.boundary_nodes.add(existing.id)
            else:
                self.hyper.add_node(node)

        for edge in other.edges.values():
            if not self.hyper.get_edge(edge.id):
                # Ensure all edge nodes exist
                valid_nodes = [nid for nid in edge.nodes if self.hyper.get_node(nid)]
                if len(valid_nodes) >= 2:
                    edge.nodes = valid_nodes
                    self.hyper.add_edge(edge)
