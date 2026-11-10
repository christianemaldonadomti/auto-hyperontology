"""
Auto-Hyperontology: Core Data Structures & Topological Reification
Grounded in Set Theory and Hypergraph Theory H = (X, E) where E <= P(X) \\ {empty},
Basic Formal Ontology (BFO) tiers, Distributed Ontology Language (DOL) morphisms,
and the Proto-AGI v4.0 Forensic JSON Schema (biol1.json & index.php).
"""

from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional, Set, Tuple
import json
import math
import uuid


def normalize_list_str(val: Any, default: Optional[List[str]] = None) -> List[str]:
    if isinstance(val, list):
        out = []
        for x in val:
            if isinstance(x, dict):
                n_val = x.get("node") or x.get("id") or ""
                if n_val:
                    out.append(str(n_val).strip())
            elif str(x).strip():
                out.append(str(x).strip())
        return out
    if isinstance(val, str) and val.strip():
        return [val.strip()]
    return list(default) if default else []


def normalize_roles_dict(roles_val: Any, nodes: List[str]) -> Dict[str, str]:
    if isinstance(roles_val, dict):
        return {str(k): str(v) for k, v in roles_val.items()}
    if isinstance(roles_val, list):
        out = {}
        for idx, item in enumerate(roles_val):
            nid = nodes[idx] if idx < len(nodes) else f"node_{idx+1}"
            if isinstance(item, dict):
                if "node" in item and "note" in item:
                    out[str(item["node"])] = str(item["note"])
                else:
                    for k, v in item.items():
                        out[str(k)] = str(v)
            else:
                out[str(item)] = nid
        return out
    return {f"role_{i+1}": nid for i, nid in enumerate(nodes)}


def normalize_node_weights(weights_val: Any, nodes: List[str]) -> Dict[str, float]:
    if not nodes:
        return {}
    uniform_w = round(1.0 / len(nodes), 4)
    if isinstance(weights_val, dict):
        out = {}
        for nid in nodes:
            raw_v = weights_val.get(nid, uniform_w)
            try:
                out[nid] = float(raw_v)
            except (ValueError, TypeError):
                out[nid] = uniform_w
        return out
    if isinstance(weights_val, list):
        out = {}
        for idx, nid in enumerate(nodes):
            if idx < len(weights_val):
                item = weights_val[idx]
                if isinstance(item, (int, float)):
                    out[nid] = float(item)
                elif isinstance(item, dict) and nid in item:
                    try:
                        out[nid] = float(item[nid])
                    except (ValueError, TypeError):
                        out[nid] = uniform_w
                else:
                    try:
                        out[nid] = float(item)
                    except (ValueError, TypeError):
                        out[nid] = uniform_w
            else:
                out[nid] = uniform_w
        return out
    return {nid: uniform_w for nid in nodes}


@dataclass
class HyperNode:
    """
    Multi-category conceptual vertex x in X within one or more domain ontologies.
    Includes full `labels` and rich `attributes` (examples, mechanisms, sources, definitional_note)
    as used in Proto-AGI v4.0 (`biol1.json`).
    """
    id: str
    name: str
    domains: List[str] = field(default_factory=list)
    categories: List[str] = field(default_factory=list)
    labels: List[str] = field(default_factory=list)
    attributes: Dict[str, Any] = field(default_factory=dict)
    aliases: List[str] = field(default_factory=list)
    provenance: List[Dict[str, Any]] = field(default_factory=list)
    prime_label: Optional[int] = None
    connectivity_degree: int = 0
    raptor_layer: int = 0
    matryoshka_embedding: List[float] = field(default_factory=list)

    def __post_init__(self):
        self.id = str(self.id).strip()
        self.name = str(self.name).strip() if self.name else self.id
        self.domains = normalize_list_str(self.domains, default=["General"])
        self.categories = normalize_list_str(self.categories, default=["Concept"])
        self.labels = normalize_list_str(self.labels)
        self.aliases = normalize_list_str(self.aliases)
        if not isinstance(self.attributes, dict):
            self.attributes = {"definitional_note": str(self.attributes)}

    @property
    def is_boundary(self) -> bool:
        return len(set(self.domains)) > 1 or len(set(self.categories)) > 1

    def to_slice_dict(self) -> Dict[str, Any]:
        """Formats node in the exact biol1.json schema expected by Proto-AGI v4.0."""
        return {
            "id": self.id,
            "name": self.name,
            "categories": self.categories,
            "labels": self.labels if self.labels else self.aliases,
            "attributes": self.attributes
        }


@dataclass
class BoundaryNode(HyperNode):
    domain_mappings: Dict[str, str] = field(default_factory=dict)
    bridge_degree: float = 1.0

    def __post_init__(self):
        super().__post_init__()
        if not isinstance(self.domain_mappings, dict):
            self.domain_mappings = {}


@dataclass
class HyperEdge:
    """
    An n-ary hyperedge e in E <= P(X) \\ {empty} connecting k >= 2 nodes simultaneously.
    Supports both `nodes` and `connects`, `relation_type` (`type`), `relation`, `constraints`, and `notes`.
    """
    id: str
    nodes: List[str]
    relation_type: str = "StudyRelation"
    relation: str = "relates"
    roles: Dict[str, str] = field(default_factory=dict)
    node_weights: Dict[str, float] = field(default_factory=dict)
    constraints: Dict[str, Any] = field(default_factory=dict)
    notes: str = ""
    confidence: float = 1.0
    domain_scope: List[str] = field(default_factory=list)
    attributes: Dict[str, Any] = field(default_factory=dict)
    provenance: Dict[str, Any] = field(default_factory=dict)
    factorization_pos: int = 1
    raptor_layer: int = 0

    def __post_init__(self):
        self.id = str(self.id).strip()
        self.nodes = normalize_list_str(self.nodes)
        self.relation_type = str(self.relation_type).strip() if self.relation_type else "AssociatedWith"
        self.relation = str(self.relation).strip() if self.relation else self.relation_type.lower()
        self.roles = normalize_roles_dict(self.roles, self.nodes)
        self.node_weights = normalize_node_weights(self.node_weights, self.nodes)
        self.domain_scope = normalize_list_str(self.domain_scope)
        if not isinstance(self.constraints, dict):
            self.constraints = {"rule": str(self.constraints)}
        if not isinstance(self.attributes, dict):
            self.attributes = {"description": str(self.attributes)}
        if not self.notes and isinstance(self.attributes, dict):
            self.notes = str(
                self.attributes.get("application")
                or self.attributes.get("evidence_sentence")
                or self.attributes.get("description")
                or ""
            )
        if not isinstance(self.provenance, dict):
            self.provenance = {"raw": str(self.provenance)}
        try:
            self.confidence = max(0.0, min(1.0, float(self.confidence)))
        except (ValueError, TypeError):
            self.confidence = 0.9

    @property
    def cardinality(self) -> int:
        return len(self.nodes)

    def to_slice_dict(self) -> Dict[str, Any]:
        """Formats hyperedge in the exact biol1.json schema expected by Proto-AGI v4.0."""
        d: Dict[str, Any] = {
            "id": self.id,
            "type": self.relation_type,
            "relation": self.relation,
            "connects": self.nodes,
        }
        if self.constraints:
            d["constraints"] = self.constraints
        if self.notes:
            d["notes"] = self.notes
        return d


class Hyperontology:
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
        edge.node_weights = normalize_node_weights(edge.node_weights, edge.nodes)
        edge.roles = normalize_roles_dict(edge.roles, edge.nodes)
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
            if (
                n.name.lower() == name_lower
                or any(alias.lower() == name_lower for alias in n.aliases)
                or any(lbl.lower() == name_lower for lbl in n.labels)
            ):
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
        n_limit = max(10, min(300, n_limit))
        if hub_node_id not in self.nodes:
            if not self.nodes:
                return {"hub": None, "nodes": [], "links": [], "total_rendered": 0}
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
                        role = edge.roles.get(participant_id, "participant") if isinstance(edge.roles, dict) else "participant"
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
            "ontology": {
                "id": self.name.lower(),
                "name": self.name,
                "version": "4.0"
            },
            "name": self.name,
            "nodes": [asdict(n) for n in self.nodes.values()],
            "edges": [asdict(e) for e in self.edges.values()],
            "hyperedges": [e.to_slice_dict() for e in self.edges.values()],
            "boundary_nodes": list(self.boundary_nodes)
        }

    def save_json(self, filepath: str):
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    def save_webgl_json(self, filepath: str, hub_node_id: Optional[str] = None, n_limit: int = 300):
        target_hub = hub_node_id or (next(iter(self.nodes.keys())) if self.nodes else "")
        payload = self.extract_bounded_subgraph(target_hub, n_limit=n_limit)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

    @classmethod
    def load_json(cls, filepath: str) -> "Hyperontology":
        """
        Loads both our compiled JSON format AND native Proto-AGI v4.0 JSON files like `biol1.json`!
        """
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        ont_meta = data.get("ontology", {})
        h_name = data.get("name") or ont_meta.get("name") or "LoadedHyperontology"
        h = cls(name=h_name)

        for nd in data.get("nodes", []):
            nid = nd.get("id", "")
            if not nid:
                continue
            node = HyperNode(
                id=nid,
                name=nd.get("name", nid),
                domains=nd.get("domains", ["Biology"]),
                categories=nd.get("categories", ["Concept"]),
                labels=nd.get("labels", []),
                attributes=nd.get("attributes", {}),
                aliases=nd.get("aliases", nd.get("labels", [])),
                provenance=nd.get("provenance", []),
                prime_label=nd.get("prime_label"),
                connectivity_degree=nd.get("connectivity_degree", 0),
                raptor_layer=nd.get("raptor_layer", 0)
            )
            h.add_node(node)

        # Support both 'edges' and 'hyperedges' (biol1.json uses 'hyperedges' with 'connects')
        raw_edges = data.get("edges") or data.get("hyperedges") or []
        for ed in raw_edges:
            eid = ed.get("id") or f"h_{uuid.uuid4().hex[:6]}"
            raw_connects = ed.get("nodes") or ed.get("connects") or []
            connects_ids = normalize_list_str(raw_connects)
            # Extract inline notes from connects if structured like h_truth_constraints_examples
            inline_notes = []
            if isinstance(raw_connects, list):
                for item in raw_connects:
                    if isinstance(item, dict) and "note" in item:
                        inline_notes.append(f"{item.get('node')}: {item.get('note')}")
            notes_str = ed.get("notes", "")
            if inline_notes:
                notes_str = (notes_str + " | " + "; ".join(inline_notes)).strip(" |")

            edge = HyperEdge(
                id=eid,
                nodes=connects_ids,
                relation_type=ed.get("relation_type") or ed.get("type") or "StudyRelation",
                relation=ed.get("relation") or "relates",
                roles=ed.get("roles", {}),
                node_weights=ed.get("node_weights", {}),
                constraints=ed.get("constraints", {}),
                notes=notes_str,
                confidence=ed.get("confidence", 1.0),
                domain_scope=ed.get("domain_scope", ["Biology"]),
                attributes=ed.get("attributes", {"notes": notes_str} if notes_str else {})
            )
            if len(edge.nodes) >= 1:
                h.add_edge(edge)

        return h
