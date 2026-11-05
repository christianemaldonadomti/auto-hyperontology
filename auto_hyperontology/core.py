"""
Auto-Hyperontology: Core Data Structures & Topological Reification
Grounded in Set Theory and Hypergraph Theory H = (X, E) where E <= P(X) \\ {empty},
Basic Formal Ontology (BFO) tiers, Distributed Ontology Language (DOL) morphisms,
and WebGL Topological Reification (N_limit <= 300).

References:
- Maldonado-Sifuentes et al. (2025), "Systemic Knowledge Compilation: Operationalizing n-ary
  Hypergraphs for Local LLM Reasoning", POLIBITS, vol. 67(1), pp. 27-32.
- Maldonado-Sifuentes (2025), "Ordinal-Cardinal Mixing in Hyperontology Construction via OEIS A320228".
"""

from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional, Set, Tuple
import json
import math
import uuid


@dataclass
class HyperNode:
    """
    Multi-category conceptual vertex x in X within one or more domain ontologies.
    Categories adhere to Basic Formal Ontology (BFO) and Proto-AGI tiers:
    Concept, Property, Process, Evidence, Level.
    """
    id: str
    name: str
    domains: List[str] = field(default_factory=list)
    categories: List[str] = field(default_factory=list)
    attributes: Dict[str, Any] = field(default_factory=dict)
    aliases: List[str] = field(default_factory=list)
    provenance: List[Dict[str, Any]] = field(default_factory=list)
    prime_label: Optional[int] = None          # Coprime vertex label for collision-free CRT indexing
    connectivity_degree: int = 0               # Connectivity degree D_v
    raptor_layer: int = 0                      # 0 = granular leaf chunk, 1 = cluster summary, 2 = root abstraction
    matryoshka_embedding: List[float] = field(default_factory=list) # Nested multi-scale representation

    @property
    def is_boundary(self) -> bool:
        """A Boundary Node (Nodo Frontera) exists simultaneously in >= 2 domain ontologies."""
        return len(set(self.domains)) > 1


@dataclass
class BoundaryNode(HyperNode):
    """
    Explicit logical anchor connecting heterogeneous domain ontologies (DOL morphisms).
    Supports multivalued/fuzzy equivalence across domains.
    """
    domain_mappings: Dict[str, str] = field(default_factory=dict)
    bridge_degree: float = 1.0


@dataclass
class HyperEdge:
    """
    An n-ary hyperedge e in E <= P(X) \\ {empty} connecting k >= 2 nodes simultaneously.
    Eliminates binary reification loss and prevents LogicPoison topological exploits.
    """
    id: str
    nodes: List[str]                           # Participating vertex IDs (cardinality |H_e| >= 2)
    relation_type: str                         # CausalLink, PartOf, Subsumption, CrossDomainBridge, PrerequisiteLattice
    roles: Dict[str, str] = field(default_factory=dict)
    node_weights: Dict[str, float] = field(default_factory=dict) # DyHSL intra-hyperedge attention weights
    confidence: float = 1.0
    domain_scope: List[str] = field(default_factory=list)
    attributes: Dict[str, Any] = field(default_factory=dict)
    provenance: Dict[str, Any] = field(default_factory=dict)
    factorization_pos: int = 1                 # Pos(H_e) in the Sabidussi-Vizing factorization sequence
    raptor_layer: int = 0                      # Abstraction layer in the fractal RAPTOR hierarchy

    @property
    def cardinality(self) -> int:
        """The semantic width |H_e| of the n-ary relation."""
        return len(self.nodes)


class Hyperontology:
    """
    Mathematical Hyperontology H = (X, E) container, incidence indexer, and
    WebGL Topological Reification compiler.
    """
    def __init__(self, name: str = "RootHyperontology"):
        self.name = name
        self.nodes: Dict[str, HyperNode] = {}
        self.edges: Dict[str, HyperEdge] = {}
        self.node_to_edges: Dict[str, Set[str]] = {}
        self.domain_to_nodes: Dict[str, Set[str]] = {}
        self.category_to_nodes: Dict[str, Set[str]] = {}
        self.layer_to_edges: Dict[int, Set[str]] = {0: set(), 1: set(), 2: set()}
        self.boundary_nodes: Set[str] = set()

    def add_node(self, node: HyperNode) -> HyperNode:
        if not node.categories:
            node.categories = ["Concept"]
        self.nodes[node.id] = node
        self.node_to_edges.setdefault(node.id, set())
        for d in node.domains:
            self.domain_to_nodes.setdefault(d, set()).add(node.id)
        for c in node.categories:
            self.category_to_nodes.setdefault(c, set()).add(node.id)
        if node.is_boundary:
            self.boundary_nodes.add(node.id)
        self._update_degree(node.id)
        return node

    def add_edge(self, edge: HyperEdge) -> HyperEdge:
        # Compute default DyHSL intra-edge self-attention weights if not provided
        if not edge.node_weights and edge.nodes:
            uniform_w = round(1.0 / len(edge.nodes), 4)
            edge.node_weights = {nid: uniform_w for nid in edge.nodes}
        self.edges[edge.id] = edge
        self.layer_to_edges.setdefault(edge.raptor_layer, set()).add(edge.id)
        for nid in edge.nodes:
            self.node_to_edges.setdefault(nid, set()).add(edge.id)
            self._update_degree(nid)
        return edge

    def _update_degree(self, node_id: str):
        if node_id in self.nodes:
            self.nodes[node_id].connectivity_degree = len(self.node_to_edges.get(node_id, set()))

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

    def get_neighbors(self, node_id: str) -> Set[str]:
        neighbors = set()
        for edge in self.get_incident_edges(node_id):
            for target_id in edge.nodes:
                if target_id != node_id:
                    neighbors.add(target_id)
        return neighbors

    def extract_bounded_subgraph(
        self,
        hub_node_id: str,
        n_limit: int = 300
    ) -> Dict[str, Any]:
        """
        Implements POLIBITS Section III.A & III.B:
        Local Subgraph Extraction and Topological Reification for WebGL 3D Force-Directed
        visualization (e.g., reader-math.php).
        - Bounds neighborhood exploration strictly to 10 <= N_limit <= 300 to preserve 60 FPS.
        - Injects intermediate relational polygons R_k for each n-ary hyperedge e in E
          producing directed links (v_s, R_k) and (R_k, v_t).
        """
        n_limit = max(10, min(300, n_limit))
        if hub_node_id not in self.nodes:
            if not self.nodes:
                return {"hub": None, "nodes": [], "links": [], "total_rendered": 0}
            # Default to highest connectivity degree hub D_v
            hub_node_id = max(self.nodes.keys(), key=lambda k: self.nodes[k].connectivity_degree)

        visited_nodes: Set[str] = {hub_node_id}
        reified_nodes: List[Dict[str, Any]] = []
        links: List[Dict[str, Any]] = []

        hub = self.nodes[hub_node_id]
        reified_nodes.append({
            "id": hub.id,
            "label": hub.name,
            "type": "hub",
            "geometry": "sphere_red",
            "degree": hub.connectivity_degree,
            "domains": hub.domains,
            "prime_label": hub.prime_label
        })

        # BFS traversal prioritized by connectivity degree D_v and hyperedge cardinality
        queue = [hub_node_id]
        processed_edges: Set[str] = set()

        while queue and len(reified_nodes) < n_limit:
            curr_id = queue.pop(0)
            incident = sorted(
                self.get_incident_edges(curr_id),
                key=lambda e: (e.cardinality, e.confidence),
                reverse=True
            )
            for edge in incident:
                if edge.id in processed_edges:
                    continue
                if len(reified_nodes) >= n_limit:
                    break
                processed_edges.add(edge.id)

                # Inject intermediate relational polygon node R_k (purple tetrahedron)
                rk_id = f"R_{edge.id}"
                reified_nodes.append({
                    "id": rk_id,
                    "label": edge.relation_type,
                    "type": "reified_hyperedge",
                    "geometry": "tetrahedron_purple",
                    "cardinality": edge.cardinality,
                    "confidence": edge.confidence,
                    "domain_scope": edge.domain_scope
                })

                for participant_id in edge.nodes:
                    if participant_id not in visited_nodes and len(reified_nodes) < n_limit:
                        visited_nodes.add(participant_id)
                        p_node = self.nodes[participant_id]
                        reified_nodes.append({
                            "id": p_node.id,
                            "label": p_node.name,
                            "type": "boundary_concept" if p_node.is_boundary else "concept",
                            "geometry": "sphere_gold" if p_node.is_boundary else "sphere_blue",
                            "degree": p_node.connectivity_degree,
                            "domains": p_node.domains,
                            "prime_label": p_node.prime_label
                        })
                        queue.append(participant_id)

                    if participant_id in visited_nodes:
                        role = edge.roles.get(participant_id, "participant")
                        # (v_s, R_k) or (R_k, v_t) directional particle stream
                        if participant_id == curr_id:
                            links.append({"source": participant_id, "target": rk_id, "role": role})
                        else:
                            links.append({"source": rk_id, "target": participant_id, "role": role})

        return {
            "hub": hub_node_id,
            "n_limit": n_limit,
            "total_rendered": len(reified_nodes),
            "nodes": reified_nodes,
            "links": links
        }

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

    def save_webgl_json(self, filepath: str, hub_node_id: Optional[str] = None, n_limit: int = 300):
        """Exports the topologically reified JSON ready for WebGL 3D Force-Directed visualizers."""
        target_hub = hub_node_id or (next(iter(self.nodes.keys())) if self.nodes else "")
        payload = self.extract_bounded_subgraph(target_hub, n_limit=n_limit)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

    @classmethod
    def load_json(cls, filepath: str) -> "Hyperontology":
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        h = cls(name=data.get("name", "LoadedHyperontology"))
        for nd in data.get("nodes", []):
            if "domain_mappings" in nd or "bridge_degree" in nd:
                h.add_node(BoundaryNode(**nd))
            else:
                h.add_node(HyperNode(**nd))
        for ed in data.get("edges", []):
            h.add_edge(HyperEdge(**ed))
        return h
