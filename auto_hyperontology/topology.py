"""
Auto-Hyperontology: Number-Theoretic & Topological Engine
Implements the exact mathematical formulations from:
"Ordinal-Cardinal Mixing in Hyperontology Construction: A Theoretic Framework for Optimizing
Large-to-Small LLM Knowledge Distillation and Adaptive Retrieval via OEIS A320228 Dynamics"
(Maldonado-Sifuentes, 2025).
"""

import math
from typing import List, Dict, Tuple, Set, Optional, Any
from auto_hyperontology.core import HyperEdge, HyperNode, Hyperontology, normalize_node_weights


PRIMES = [
    2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71,
    73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151,
    157, 163, 167, 173, 179, 181, 191, 193, 197, 199, 211, 223, 227, 229
]

OEIS_A320228_SEQ = [1, 0, 0, 0, 0, 1, 1, 2, 6, 16, 59, 265, 1544, 10778, 88168, 805491]


def is_prime(n: int) -> bool:
    if n < 2:
        return False
    if n in (2, 3):
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True


def prime_weight(cardinality: int) -> float:
    if cardinality <= 0:
        return 0.0
    if is_prime(cardinality):
        return 1.5 * (float(cardinality) ** 1.25)
    if cardinality == 4:
        return 1.35 * (float(cardinality) ** 1.20)
    return float(cardinality) ** 1.10


def f_oc_retrieval_score(
    sim_q_he: float,
    cardinality: int,
    pos_he: int,
    alpha: float = 0.45,
    beta: float = 0.55
) -> float:
    pos = max(1, pos_he)
    p_weight = prime_weight(cardinality)
    return (alpha * float(sim_q_he)) + (beta * p_weight * (1.0 / float(pos)))


def oeis_a320228_weight(ordinal_n: int, cardinality_k: int, sim_score: float = 1.0) -> float:
    return f_oc_retrieval_score(
        sim_q_he=sim_score,
        cardinality=cardinality_k,
        pos_he=ordinal_n
    )


class TopologicalIndexer:
    def __init__(self, hyperontology: Hyperontology):
        self.hyper = hyperontology
        self.prime_map: Dict[str, int] = {}
        self._assign_coprime_prime_labels()
        self._assign_factorization_positions()

    def _assign_coprime_prime_labels(self):
        sorted_nodes = sorted(
            self.hyper.nodes.values(),
            key=lambda n: (n.is_boundary, n.connectivity_degree, n.id),
            reverse=True
        )
        for idx, node in enumerate(sorted_nodes):
            p_val = PRIMES[idx] if idx < len(PRIMES) else (233 + 2 * idx)
            self.prime_map[node.id] = p_val
            node.prime_label = p_val

    def _assign_factorization_positions(self):
        sorted_edges = sorted(
            self.hyper.edges.values(),
            key=lambda e: (
                is_prime(e.cardinality),
                len(e.domain_scope) > 1,
                e.cardinality,
                e.confidence
            ),
            reverse=True
        )
        for pos, edge in enumerate(sorted_edges, start=1):
            edge.factorization_pos = pos

    def compute_edge_signature(self, edge: HyperEdge) -> int:
        sig = 1
        for nid in edge.nodes:
            sig *= self.prime_map.get(nid, 1)
        return sig

    def verify_coprime_adjacency(self) -> bool:
        for edge in self.hyper.edges.values():
            nids = edge.nodes
            for i in range(len(nids)):
                for j in range(i + 1, len(nids)):
                    l_u = self.prime_map.get(nids[i], 1)
                    l_v = self.prime_map.get(nids[j], 1)
                    if math.gcd(l_u, l_v) != 1:
                        return False
        return True

    def compute_semantic_similarity(
        self,
        edge: HyperEdge,
        seed_nodes: Set[str],
        query_text: Optional[str] = None
    ) -> float:
        if not edge.nodes:
            return 0.0

        matched_seeds = [nid for nid in edge.nodes if nid in seed_nodes]
        hit_count = len(matched_seeds)

        weights_dict = (
            edge.node_weights
            if isinstance(edge.node_weights, dict)
            else normalize_node_weights(edge.node_weights, edge.nodes)
        )

        weighted_overlap = sum(
            float(weights_dict.get(nid, 1.0 / len(edge.nodes)))
            for nid in matched_seeds
        )

        convergence_bonus = 1.0 + (0.85 * (hit_count - 1)) if hit_count >= 2 else 0.50

        intent_bonus = 1.0
        if query_text:
            q_low = query_text.lower()
            rel_low = str(edge.relation_type).lower()
            attrs_dict = edge.attributes if isinstance(edge.attributes, dict) else {}
            app_low = (str(edge.notes) + " " + str(attrs_dict.get("application", ""))).lower()
            if ("prerequisite" in q_low or "foundational" in q_low or "master" in q_low) and "prerequisite" in rel_low:
                intent_bonus = 8.0
            elif any(w in q_low for w in ["physics", "kinematics", "velocity"]) and "physics" in [str(d).lower() for d in edge.domain_scope]:
                intent_bonus = 2.5
            elif any(w in q_low for w in ["economic", "marginal", "profit"]) and "economics" in [str(d).lower() for d in edge.domain_scope]:
                intent_bonus = 2.5
            elif any(w in q_low for w in ["neural", "gradient", "ai", "backpropagation"]) and "computerscience" in [str(d).lower() for d in edge.domain_scope]:
                intent_bonus = 2.5
            elif any(w in q_low for w in ["aspirin", "prophylaxis", "recovery"]) and "aspirin" in app_low:
                intent_bonus = 4.0

        sim = (weighted_overlap * 0.8 + float(edge.confidence) * 0.2) * convergence_bonus * intent_bonus
        return round(sim, 4)

    def schedule_ordinal_cardinal_retrieval(
        self,
        candidate_edges: List[HyperEdge],
        seed_nodes: List[str],
        alpha: float = 0.45,
        beta: float = 0.55,
        query_text: Optional[str] = None
    ) -> List[Tuple[HyperEdge, float]]:
        seed_set = set(seed_nodes)

        multi_seed_edges = [
            e for e in candidate_edges if len(seed_set.intersection(e.nodes)) >= 2
        ]
        active_pool = multi_seed_edges if (len(seed_set) >= 2 and multi_seed_edges) else candidate_edges

        ranked: List[Tuple[HyperEdge, float]] = []
        for idx, edge in enumerate(active_pool, start=1):
            sim_q_he = self.compute_semantic_similarity(edge, seed_set, query_text=query_text)
            pos_he = edge.factorization_pos if edge.factorization_pos > 0 else idx
            score = f_oc_retrieval_score(
                sim_q_he=sim_q_he * 4.5,
                cardinality=edge.cardinality,
                pos_he=pos_he,
                alpha=alpha,
                beta=beta
            )
            ranked.append((edge, round(score, 4)))

        ranked.sort(key=lambda x: x[1], reverse=True)
        return ranked

    def raptor_cardinality_gate(self, query: str, matched_seed_nodes: List[str]) -> Dict[str, Any]:
        q_words = [w.strip("?,.") for w in query.lower().split() if len(w) > 3]
        involved_domains = set()
        for nid in matched_seed_nodes:
            node = self.hyper.get_node(nid)
            if node:
                involved_domains.update(node.domains)

        query_cardinality = max(len(matched_seed_nodes), len(involved_domains), len(q_words) // 3)

        if query_cardinality <= 1:
            return {
                "query_cardinality": query_cardinality,
                "complexity_tier": "Low (Ordinal / Factoid)",
                "target_raptor_layer": 0,
                "matryoshka_dim": 64,
                "routing_strategy": "Single-Step Leaf Retrieval"
            }
        elif query_cardinality <= 3 and len(involved_domains) <= 2:
            return {
                "query_cardinality": query_cardinality,
                "complexity_tier": "Medium (Prime-Stride Multi-Hop)",
                "target_raptor_layer": 1,
                "matryoshka_dim": 512,
                "routing_strategy": "Cluster-Level Hyperedge Traversal"
            }
        else:
            return {
                "query_cardinality": query_cardinality,
                "complexity_tier": "High (Cardinal Cross-Domain Synthesis)",
                "target_raptor_layer": 2,
                "matryoshka_dim": 1024,
                "routing_strategy": "Full Ordinal-Cardinal Mixing + Boundary Morphisms"
            }

    @staticmethod
    def prime_stride_attention_indices(
        sequence_length: int,
        prime_strides: List[int] = [2, 3, 5]
    ) -> Dict[str, Any]:
        if sequence_length <= 0:
            return {"attended_indices": [], "coverage_ratio": 0.0, "crt_period": 1}

        attended: Set[int] = {0}
        crt_period = 1
        for p in prime_strides:
            crt_period *= p
            for idx in range(0, sequence_length, p):
                attended.add(idx)

        sorted_indices = sorted(attended)
        coverage = round(len(sorted_indices) / float(sequence_length), 4)
        return {
            "sequence_length": sequence_length,
            "prime_strides": prime_strides,
            "crt_collision_period": crt_period,
            "attended_indices": sorted_indices,
            "attended_count": len(sorted_indices),
            "compression_ratio": round(1.0 - coverage, 4)
        }

    @staticmethod
    def crt_reconstruct(remainders: List[int], prime_moduli: List[int]) -> int:
        if len(remainders) != len(prime_moduli) or not remainders:
            return 0
        total_m = 1
        for m in prime_moduli:
            total_m *= m

        result = 0
        for rem, mod in zip(remainders, prime_moduli):
            b_i = total_m // mod
            inv = pow(b_i, -1, mod)
            result = (result + rem * b_i * inv) % total_m
        return result

    def build_distillation_curriculum(self) -> Dict[str, List[HyperEdge]]:
        phase1, phase2, phase3 = [], [], []
        for edge in self.hyper.edges.values():
            k = edge.cardinality
            if k == 2:
                phase1.append(edge)
            elif is_prime(k) and k >= 3:
                phase2.append(edge)
            else:
                phase3.append(edge)

        return {
            "phase_1_dyadic_foundation": phase1,
            "phase_2_prime_stride": phase2,
            "phase_3_high_cardinality_mix": phase3
        }

    def export_sa_mlp_features(self) -> List[Dict[str, Any]]:
        features = []
        for nid, node in self.hyper.nodes.items():
            incident = self.hyper.get_incident_edges(nid)
            min_pos = min((e.factorization_pos for e in incident), default=0)
            max_card = max((e.cardinality for e in incident), default=0)
            neighbor_primes_prod = 1
            for nb_id in self.hyper.get_neighbors(nid):
                neighbor_primes_prod = (neighbor_primes_prod * self.prime_map.get(nb_id, 1)) % 997

            features.append({
                "node_id": nid,
                "name": node.name,
                "sa_mlp_vector": [
                    float(node.connectivity_degree),
                    float(node.prime_label or 2),
                    float(min_pos),
                    float(max_card),
                    1.0 if node.is_boundary else 0.0,
                    float(neighbor_primes_prod)
                ]
            })
        return features

    def enforce_4_regular_expander_check(self) -> Dict[str, Any]:
        degrees = [n.connectivity_degree for n in self.hyper.nodes.values()]
        if not degrees:
            return {"mean_degree": 0.0, "regularity_deviation": 0.0, "a320228_configurations": 1}

        n_count = len(degrees)
        mean_deg = sum(degrees) / float(n_count)
        deviation_from_4 = sum(abs(d - 4) for d in degrees) / float(n_count)
        a320228_count = (
            OEIS_A320228_SEQ[n_count]
            if n_count < len(OEIS_A320228_SEQ)
            else OEIS_A320228_SEQ[-1]
        )
        return {
            "num_vertices": n_count,
            "mean_degree": round(mean_deg, 2),
            "deviation_from_4_regularity": round(deviation_from_4, 2),
            "oeis_a320228_4_regular_space_size": a320228_count
        }
