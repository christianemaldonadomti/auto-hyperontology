from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional, Set
import json
import uuid

@dataclass
class HyperNode:
    id: str
    name: str
    domains: List[str] = field(default_factory=list)
    categories: List[str] = field(default_factory=list)
    attributes: Dict[str, Any] = field(default_factory=dict)
    aliases: List[str] = field(default_factory=list)
    provenance: List[Dict[str, Any]] = field(default_factory=list)
    prime_index: Optional[int] = None

    @property
    def is_boundary(self) -> bool:
        return len(set(self.domains)) > 1

@dataclass
class HyperEdge:
    id: str
    nodes: List[str]
    relation_type: str
    roles: Dict[str, str] = field(default_factory=dict)
    confidence: float = 1.0
    domain_scope: List[str] = field(default_factory=list)
    attributes: Dict[str, Any] = field(default_factory=dict)
    provenance: Dict[str, Any] = field(default_factory=dict)

    @property
    def cardinality(self) -> int:
        return len(self.nodes)

@dataclass
class BoundaryNode(HyperNode):
    domain_mappings: Dict[str, str] = field(default_factory=dict)
    bridge_degree: float = 1.0

class Hyperontology:
    def __init__(self, name: str = "RootHyperontology"):
        self.name = name
        self.nodes: Dict[str, HyperNode] = {}
        self.edges: Dict[str, HyperEdge] = {}
        self.node_to_edges: Dict[str, Set[str]] = {}
        self.domain_to_nodes: Dict[str, Set[str]] = {}
        self.category_to_nodes: Dict[str, Set[str]] = {}
        self.boundary_nodes: Set[str] = set()

    def add_node(self, node: HyperNode) -> HyperNode:
        self.nodes[node.id] = node
        if node.id not in self.node_to_edges:
            self.node_to_edges[node.id] = set()
        for d in node.domains:
            self.domain_to_nodes.setdefault(d, set()).add(node.id)
        for c in node.categories:
            self.category_to_nodes.setdefault(c, set()).add(node.id)
        if node.is_boundary:
            self.boundary_nodes.add(node.id)
        return node

    def add_edge(self, edge: HyperEdge) -> HyperEdge:
        self.edges[edge.id] = edge
        for nid in edge.nodes:
            self.node_to_edges.setdefault(nid, set()).add(edge.id)
        return edge

    def get_node(self, node_id: str) -> Optional[HyperNode]:
        return self.nodes.get(node_id)

    def get_edge(self, edge_id: str) -> Optional[HyperEdge]:
        return self.edges.get(edge_id)

    def find_nodes_by_name(self, name: str, domain: Optional[str] = None) -> List[HyperNode]:
        name_lower = name.lower().strip()
        matches = []
        for n in self.nodes.values():
            if domain and domain not in n.domains:
                continue
            if n.name.lower() == name_lower or any(alias.lower() == name_lower for alias in n.aliases):
                matches.append(n)
        return matches

    def get_incident_edges(self, node_id: str) -> List[HyperEdge]:
        edge_ids = self.node_to_edges.get(node_id, set())
        return [self.edges[eid] for eid in edge_ids if eid in self.edges]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "nodes": [asdict(n) for n in self.nodes.values()],
            "edges": [asdict(e) for e in self.edges.values()],
            "boundary_nodes": list(self.boundary_nodes)
        }

    def save_json(self, filepath: str):
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
