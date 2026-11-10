"""
Auto-Hyperontology RAG (Hyper-RAG) Universal Facade
Provides a clean, corpus-agnostic Python API for building, indexing, distilling,
visualizing, and querying n-ary mathematical hyperontologies.
"""

import os
import json
from typing import List, Dict, Any, Optional, Tuple
from auto_hyperontology.core import Hyperontology, HyperNode, HyperEdge, BoundaryNode
from auto_hyperontology.extractor import AutoHyperontologyExtractor, OllamaClient
from auto_hyperontology.reasoner import ProtoAGIReasoner, ReasoningResult, AMEMetrics
from auto_hyperontology.topology import TopologicalIndexer


class HyperRAG:
    def __init__(
        self,
        model: str = "intuyt:macro",
        base_url: str = "https://comic-hugely-eft.ngrok-free.app",
        hyperontology: Optional[Hyperontology] = None,
        default_domain: str = "General",
        temperature: float = 0.1,
        theta_max: int = 3000,
        alpha: float = 0.45,
        beta: float = 0.55
    ):
        self.model = model
        self.base_url = base_url
        self.default_domain = default_domain
        self.alpha = alpha
        self.beta = beta
        self.client = OllamaClient(base_url=base_url, model=model, temperature=temperature)
        self.hyper = hyperontology or Hyperontology(name="UserHyperontology")
        self.extractor = AutoHyperontologyExtractor(ollama_client=self.client, theta_max=theta_max)
        self.reasoner = ProtoAGIReasoner(
            hyperontology=self.hyper,
            ollama_client=self.client,
            alpha=self.alpha,
            beta=self.beta
        )

    def _refresh_engine(self):
        self.reasoner = ProtoAGIReasoner(
            hyperontology=self.hyper,
            ollama_client=self.client,
            alpha=self.alpha,
            beta=self.beta
        )

    def add_text(self, text: str, domain: Optional[str] = None) -> "HyperRAG":
        target_domain = domain or self.default_domain
        extracted = self.extractor.extract_from_text(text, default_domain=target_domain)
        self._merge_hyperontology(extracted)
        self._refresh_engine()
        return self

    def add_documents(self, documents: List[str], domain: Optional[str] = None) -> "HyperRAG":
        for doc in documents:
            if doc.strip():
                self.add_text(doc, domain=domain)
        return self

    def add_file(self, filepath: str, domain: Optional[str] = None) -> "HyperRAG":
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
        return self.add_text(content, domain=domain)

    def add_directory(self, dirpath: str, extensions: Tuple[str, ...] = (".txt", ".md", ".json")) -> "HyperRAG":
        if not os.path.isdir(dirpath):
            raise NotADirectoryError(f"Directory not found: {dirpath}")
        for root, _, files in os.walk(dirpath):
            for file in sorted(files):
                if any(file.endswith(ext) for ext in extensions):
                    full_path = os.path.join(root, file)
                    domain_name = os.path.basename(root) if root != dirpath else self.default_domain
                    self.add_file(full_path, domain=domain_name)
        return self

    def query(
        self,
        question: str,
        max_edges: int = 5,
        memory_node_ids: Optional[List[str]] = None
    ) -> ReasoningResult:
        return self.reasoner.answer_question(
            question,
            max_edges=max_edges,
            memory_node_ids=memory_node_ids
        )

    def get_distillation_curriculum(self) -> Dict[str, List[HyperEdge]]:
        return self.reasoner.indexer.build_distillation_curriculum()

    def export_sa_mlp_features(self) -> List[Dict[str, Any]]:
        return self.reasoner.indexer.export_sa_mlp_features()

    def export_webgl_reification(
        self,
        filepath: str,
        hub_node_id: Optional[str] = None,
        n_limit: int = 300
    ) -> Dict[str, Any]:
        target_hub = hub_node_id or (
            max(self.hyper.nodes.keys(), key=lambda k: self.hyper.nodes[k].connectivity_degree)
            if self.hyper.nodes else ""
        )
        payload = self.hyper.extract_bounded_subgraph(target_hub, n_limit=n_limit)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        return payload

    def save(self, filepath: str):
        self.hyper.save_json(filepath)

    @classmethod
    def load(
        cls,
        filepath: str,
        model: str = "intuyt:macro",
        base_url: str = "https://comic-hugely-eft.ngrok-free.app"
    ) -> "HyperRAG":
        hyper = Hyperontology.load_json(filepath)
        return cls(model=model, base_url=base_url, hyperontology=hyper)

    def stats(self) -> Dict[str, Any]:
        indexer = self.reasoner.indexer
        reg_info = indexer.enforce_4_regular_expander_check()
        curriculum = indexer.build_distillation_curriculum()
        return {
            "total_nodes": len(self.hyper.nodes),
            "total_edges": len(self.hyper.edges),
            "boundary_nodes": len(self.hyper.boundary_nodes),
            "domains": list(self.hyper.domain_to_nodes.keys()),
            "categories": list(self.hyper.category_to_nodes.keys()),
            "coprime_labeling_verified": indexer.verify_coprime_adjacency(),
            "a320228_regularity": reg_info,
            "curriculum_counts": {k: len(v) for k, v in curriculum.items()}
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
                for lbl in node.labels:
                    if lbl not in existing.labels:
                        existing.labels.append(lbl)
                existing.attributes.update(node.attributes)
                existing.raptor_layer = max(existing.raptor_layer, node.raptor_layer)
                if existing.is_boundary:
                    self.hyper.boundary_nodes.add(existing.id)
            else:
                self.hyper.add_node(node)

        for edge in other.edges.values():
            if not self.hyper.get_edge(edge.id):
                for nid in edge.nodes:
                    if not self.hyper.get_node(nid):
                        self.hyper.add_node(HyperNode(id=nid, name=nid.replace("_", " ").title()))
                if len(edge.nodes) >= 1:
                    self.hyper.add_edge(edge)
