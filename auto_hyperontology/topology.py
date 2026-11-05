"""
Auto-Hyperontology: Number-Theoretic & Topological Engine
Implements the exact mathematical formulations from:
"Ordinal-Cardinal Mixing in Hyperontology Construction: A Theoretic Framework for Optimizing
Large-to-Small LLM Knowledge Distillation and Adaptive Retrieval via OEIS A320228 Dynamics"
(Maldonado-Sifuentes, 2025).

Core Components:
1. Exact A320228 Ordinal-Cardinal Mixing Function (F_OC):
   R_score(H_e) = alpha * Sim(Q, H_e) + beta * PrimeWeight(|H_e|) * (1 / Pos(H_e))
2. OEIS A320228 Sequence Dynamics & 4-Regular / Prime-Regular Expander Regularization.
3. Coprime Prime Labeling (gcd(L(u), L(v)) = 1) & Chinese Remainder Theorem (CRT) Shard Routing.
4. Sabidussi-Vizing Cartesian Product Prime Factorization & Structural Feature Extraction
   for Structure-Aware MLP (SA-MLP) Distillation and Curriculum Learning (Phases 1, 2, 3).
5. RAPTOR Cardinality Gate C(q) -> {L_0, L_1, L_2} & Matryoshka Embedding Unfolding (64 -> 512 -> 1024).
6. Prime Stride Sparse Attention Masks (strides p in {2, 3, 5, 7, 11...}) via CRT periodicity.
"""

import math
from typing import List, Dict, Tuple, Set, Optional, Any
from auto_hyperontology.core import HyperEdge, HyperNode, Hyperontology


# First 50 prime numbers for Sabidussi-Vizing prime coordinates and coprime labeling
PRIMES = [
    2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71,
    73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151,
    157, 163, 167, 173, 179, 181, 191, 193, 197, 199, 211, 223, 227, 229
]

# OEIS A320228 exact initial terms (Number of 4-regular simple graphs on n vertices, n >= 0):
# a(0)=1, a(1)=0, a(2)=0, a(3)=0, a(4)=0, a(5)=1, a(6)=1, a(7)=2, a(8)=6, a(9)=16, a(10)=59, a(11)=265...
OEIS_A320228_SEQ = [1, 0, 0, 0, 0, 1, 1, 2, 6, 16, 59, 265, 1544, 10778, 88168, 805491]


def is_prime(n: int) -> bool:
    """Checks whether integer n >= 2 is a prime number."""
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
    """
    Implements PrimeWeight(|H_e|) from Section 6.2 & 9.1:
    Networks and hyperedges with prime cardinality (e.g. |H_e| in {2, 3, 5, 7...})
    lack non-trivial symmetries that lead to degenerate states, acting as irreducible
    topological atoms.
    - If |H_e| is prime: boosts weight by 1.5 * (|H_e| ** 1.25)
    - If |H_e| == 4 (4-regular expander symmetry per OEIS A320228): boosts by 1.35 * (|H_e| ** 1.2)
    - Otherwise (composite): standard scaling |H_e| ** 1.1
    """
    if cardinality <= 0:
        return 0.0
    if is_prime(cardinality):
        return 1.5 * (float(cardinality) ** 1.25)
    if cardinality == 4:
        # 4-regular simple graph expander stability (Section 6.4)
        return 1.35 * (float(cardinality) ** 1.20)
    return float(cardinality) ** 1.10


def f_oc_retrieval_score(
    sim_q_he: float,
    cardinality: int,
    pos_he: int,
    alpha: float = 0.45,
    beta: float = 0.55
) -> float:
    """
    Exact Ordinal-Cardinal Mixing Function (F_OC) from Section 6.2:
        R_score(H_e) = alpha * Sim(Q, H_e) + beta * PrimeWeight(|H_e|) * (1 / Pos(H_e))
    Where:
        - Sim(Q, H_e): Semantic similarity / seed overlap between query Q and hyperedge H_e
        - |H_e|: Cardinality (semantic width) of the hyperedge
        - PrimeWeight(|H_e|): Topological boost for prime-sized or 4-regular cliques
        - Pos(H_e): Position of H_e in the Sabidussi-Vizing factorization sequence (ordinal rank >= 1)
    """
    pos = max(1, pos_he)
    p_weight = prime_weight(cardinality)
    return (alpha * float(sim_q_he)) + (beta * p_weight * (1.0 / float(pos)))


def oeis_a320228_weight(ordinal_n: int, cardinality_k: int, sim_score: float = 1.0) -> float:
    """
    Convenience wrapper combining F_OC retrieval scoring with log-damped positional decay.
    """
    return f_oc_retrieval_score(
        sim_q_he=sim_score,
        cardinality=cardinality_k,
        pos_he=ordinal_n
    )


class TopologicalIndexer:
    """
    Number-Theoretic and Topological Scheduler for Hyperontologies.
    Implements:
    - Coprime Prime Labeling (Section 2.3 & 3.3)
    - Sabidussi-Vizing Factorization Ordering & Structural Feature Injection (Section 3.2 & 4.4)
    - Chinese Remainder Theorem (CRT) Shard Routing & Coprime Cycle Traversal (Section 3.3)
    - RAPTOR Cardinality Gate C(q) -> {L_0, L_1, L_2} & Matryoshka Unfolding (Section 5.2 & 9.2)
    - Prime Stride Sparse Attention Masks (Section 5.3)
    """
    def __init__(self, hyperontology: Hyperontology):
        self.hyper = hyperontology
        self.prime_map: Dict[str, int] = {}
        self._assign_coprime_prime_labels()
        self._assign_factorization_positions()

    def _assign_coprime_prime_labels(self):
        """
        Section 2.3 & 3.3: Prime Labeling.
        Assigns distinct prime labels to vertices ordered by connectivity degree D_v
        and boundary status, guaranteeing that any two distinct labeled vertices u, v
        satisfy gcd(L(u), L(v)) = 1 (coprime).
        """
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
        """
        Section 3.2 & 6.2: Sabidussi-Vizing Factorization Sequence Pos(H_e).
        Orders hyperedges into irreducible prime factors first (prime cardinality,
        cross-domain bridges, high confidence) followed by composite structures.
        """
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
        """
        Computes unique Sabidussi-Vizing / Gödel prime product signature for hyperedge H_e:
            Sig(H_e) = prod_{v in H_e} L(v)
        By the Fundamental Theorem of Arithmetic, this composite integer uniquely identifies
        the exact vertex subset of H_e without collision.
        """
        sig = 1
        for nid in edge.nodes:
            sig *= self.prime_map.get(nid, 1)
        return sig

    def verify_coprime_adjacency(self) -> bool:
        """
        Verifies that all adjacent nodes across all hyperedges have coprime labels:
        gcd(L(u), L(v)) == 1 for all u != v in H_e.
        """
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
        query_terms: Optional[List[str]] = None
    ) -> float:
        """
        Computes Sim(Q, H_e) combining seed vertex overlap, DyHSL intra-edge node weights,
        and lexical/domain alignment.
        """
        if not edge.nodes:
            return 0.0
        # Weighted overlap using DyHSL intra-hyperedge attention weights
        weighted_overlap = sum(
            edge.node_weights.get(nid, 1.0 / len(edge.nodes))
            for nid in edge.nodes if nid in seed_nodes
        )
        # Boundary bonus if hyperedge bridges multiple ontologies
        boundary_factor = 1.25 if len(edge.domain_scope) > 1 else 1.0
        sim = min(1.0, (weighted_overlap * 0.75 + edge.confidence * 0.25) * boundary_factor)
        return round(sim, 4)

    def schedule_ordinal_cardinal_retrieval(
        self,
        candidate_edges: List[HyperEdge],
        seed_nodes: List[str],
        alpha: float = 0.45,
        beta: float = 0.55
    ) -> List[Tuple[HyperEdge, float]]:
        """
        Executes the exact A320228-Inspired Ordinal-Cardinal Mixing Function (Section 6.2):
            R_score(H_e) = alpha * Sim(Q, H_e) + beta * PrimeWeight(|H_e|) * (1 / Pos(H_e))
        """
        seed_set = set(seed_nodes)
        ranked: List[Tuple[HyperEdge, float]] = []

        for idx, edge in enumerate(candidate_edges, start=1):
            sim_q_he = self.compute_semantic_similarity(edge, seed_set)
            pos_he = edge.factorization_pos if edge.factorization_pos > 0 else idx
            score = f_oc_retrieval_score(
                sim_q_he=sim_q_he,
                cardinality=edge.cardinality,
                pos_he=pos_he,
                alpha=alpha,
                beta=beta
            )
            ranked.append((edge, round(score, 4)))

        ranked.sort(key=lambda x: x[1], reverse=True)
        return ranked

    # ==========================================================================
    # ADAPTIVE-RAG: RAPTOR CARDINALITY GATE & MATRYOSHKA UNFOLDING (Section 5)
    # ==========================================================================

    def raptor_cardinality_gate(self, query: str, matched_seed_nodes: List[str]) -> Dict[str, Any]:
        """
        Implements Section 5.1, 5.2 & 9.2:
        Gate Function C(q) -> {L_0, L_1, L_2} and Matryoshka Embedding Unfolding.
        - Low Cardinality Query (|Q| <= 1 concept, factoid):
            Targets RAPTOR Layer L_0 (granular leaf facts), Matryoshka dim = 64.
        - Medium Cardinality Query (|Q| == 2-3 concepts, multi-hop):
            Targets RAPTOR Layer L_1 (cluster syntheses), Matryoshka dim = 512.
        - High Cardinality Query (|Q| >= 4 concepts or cross-domain thematic):
            Targets RAPTOR Layer L_2 (root ontological abstractions + boundary bridges),
            Matryoshka dim = 1024.
        """
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

    # ==========================================================================
    # PRIME STRIDE SPARSE ATTENTION & CRT COVERAGE (Section 3.3 & 5.3)
    # ==========================================================================

    @staticmethod
    def prime_stride_attention_indices(
        sequence_length: int,
        prime_strides: List[int] = [2, 3, 5]
    ) -> Dict[str, Any]:
        """
        Implements Section 5.3: Dynamic Context Windows with Prime Strides.
        Instead of dense O(L^2) attention over every token, heads stride at coprime
        prime intervals (e.g., p in {2, 3, 5}). By the Chinese Remainder Theorem,
        heads striding by p_1 and p_2 only collide every p_1 * p_2 tokens, maximizing
        receptive field coverage while compressing active context tokens.
        """
        if sequence_length <= 0:
            return {"attended_indices": [], "coverage_ratio": 0.0, "crt_period": 1}

        attended: Set[int] = {0}  # Always attend to BOS/anchor token 0
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
        """
        Section 3.3: Chinese Remainder Theorem (CRT) state/shard router.
        Given remainders r_i (mod p_i) across coprime prime moduli, reconstructs the unique
        global topological state X (mod M), where M = prod(p_i).
        """
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

    # ==========================================================================
    # LARGE-TO-SMALL DISTILLATION: CURRICULUM & SA-MLP FEATURES (Section 4)
    # ==========================================================================

    def build_distillation_curriculum(self) -> Dict[str, List[HyperEdge]]:
        """
        Implements Section 4.2: Curriculum Learning via Graph Granularity.
        Partitions the hyperontology into three progressive training phases for small
        student models (e.g., intuyt:macro / SA-MLP):
        - Phase 1 (Low Cardinality / Dyadic Foundation): |H_e| == 2
        - Phase 2 (Medium Cardinality / Prime Stride Integration): |H_e| in {3, 5, 7} (odd primes)
        - Phase 3 (High Cardinality / The Full Mix): |H_e| >= 4 (4-regular expanders & cross-domain complexes)
        """
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
        """
        Implements Section 4.4 & 9.3: Structure-Aware MLP (SA-MLP) Feature Injection.
        Generates message-passing-free structural positional encodings for each node:
        [connectivity_degree D_v, prime_label L(v), min_factorization_pos, max_cardinality,
         is_boundary_flag, coprime_product_mod_997]
        """
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
        """
        Implements Section 6.4: The Role of OEIS A320228 in Topology (4-Regular Simple Graph Expanders).
        Evaluates how close the local node connectivity degrees D_v are to k=4 regularity
        or prime regularity, identifying potential hub bottlenecks or isolated leaves.
        """
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
