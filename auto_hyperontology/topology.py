import math
from typing import List, Dict, Tuple, Set, Optional
from auto_hyperontology.core import HyperEdge, HyperNode, Hyperontology

PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71]

def oeis_a320228_weight(n: int, cardinality: int) -> float:
    if cardinality <= 0:
        return 0.0
    return (float(cardinality) ** 1.35) / (1.0 + math.log2(float(n) + 1.0))

class TopologicalIndexer:
    def __init__(self, hyperontology: Hyperontology):
        self.hyper = hyperontology
        self.prime_map: Dict[str, int] = {}
        self._assign_prime_coordinates()

    def _assign_prime_coordinates(self):
        domain_roots = {}
        for domain, node_ids in self.hyper.domain_to_nodes.items():
            sorted_nodes = sorted(
                node_ids, 
                key=lambda nid: len(self.hyper.get_incident_edges(nid)), 
                reverse=True
            )
            domain_roots[domain] = sorted_nodes[:3]

        assigned_idx = 0
        for domain, roots in domain_roots.items():
            for r_id in roots:
                if r_id not in self.prime_map and assigned_idx < len(PRIMES):
                    prime_val = PRIMES[assigned_idx]
                    self.prime_map[r_id] = prime_val
                    node = self.hyper.get_node(r_id)
                    if node:
                        node.prime_index = prime_val
                    assigned_idx += 1

    def compute_edge_topological_signature(self, edge: HyperEdge) -> int:
        sig = 1
        for nid in edge.nodes:
            prime = self.prime_map.get(nid, 1)
            sig *= prime
        return sig

    def schedule_ordinal_cardinal_retrieval(
        self, 
        candidate_edges: List[HyperEdge], 
        seed_nodes: List[str]
    ) -> List[Tuple[HyperEdge, float]]:
        seed_set = set(seed_nodes)
        ranked = []
        for ordinal_rank, edge in enumerate(candidate_edges, start=1):
            overlap = len(seed_set.intersection(edge.nodes))
            overlap_score = (overlap / len(edge.nodes)) if edge.nodes else 0.0
            cardinal_weight = oeis_a320228_weight(ordinal_rank, edge.cardinality)
            boundary_bonus = 1.4 if len(edge.domain_scope) > 1 else 1.0
            score = (edge.confidence * 0.4 + overlap_score * 0.3) * cardinal_weight * boundary_bonus
            ranked.append((edge, score))
        ranked.sort(key=lambda x: x[1], reverse=True)
        return ranked
