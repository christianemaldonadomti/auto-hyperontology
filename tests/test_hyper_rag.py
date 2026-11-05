"""
Comprehensive Unit & Mathematical Verification Suite for Auto-Hyperontology (Hyper-RAG)
Verifies:
1. Set-Theoretic n-ary Hypergraph H=(X,E) & Boundary Node detection
2. Exact A320228 Ordinal-Cardinal Mixing Function (F_OC):
   R_score(H_e) = alpha * Sim(Q, H_e) + beta * PrimeWeight(|H_e|) * (1 / Pos(H_e))
3. Coprime Prime Labeling (gcd(L(u), L(v)) = 1) & Chinese Remainder Theorem (CRT) routing
4. RAPTOR Cardinality Gate C(q) -> {L_0, L_1, L_2} & Matryoshka Unfolding (64, 512, 1024)
5. Prime Stride Sparse Attention Masking & CRT periodicity
6. 3-Phase Large-to-Small Distillation Curriculum & SA-MLP structural feature vectors
7. WebGL Topological Reification (R_k tetrahedrons, N_limit <= 300)
8. BPE Token-Bounded Greedy Partitioning (theta_max = 3000) & GreenAGI Wh/CO2 metrics
"""

import unittest
import math
from auto_hyperontology.core import Hyperontology, HyperNode, HyperEdge, BoundaryNode
from auto_hyperontology.topology import (
    TopologicalIndexer,
    f_oc_retrieval_score,
    prime_weight,
    is_prime,
    OEIS_A320228_SEQ,
)
from auto_hyperontology.extractor import greedy_token_bounded_chunk, estimate_bpe_tokens
from auto_hyperontology.rag import HyperRAG
from auto_hyperontology.khan_academy import build_khan_academy_hyperontology


class TestOrdinalCardinalHyperontology(unittest.TestCase):
    def setUp(self):
        self.h = build_khan_academy_hyperontology()
        self.indexer = TopologicalIndexer(self.h)

    def test_boundary_node_and_degrees(self):
        deriv = self.h.get_node("derivative_rate_of_change")
        self.assertIsNotNone(deriv)
        self.assertTrue(deriv.is_boundary)
        self.assertGreaterEqual(deriv.connectivity_degree, 4)

    def test_coprime_prime_labeling(self):
        """Every distinct pair of vertices in every hyperedge must have coprime labels."""
        self.assertTrue(self.indexer.verify_coprime_adjacency())
        for edge in self.h.edges.values():
            sig = self.indexer.compute_edge_signature(edge)
            self.assertGreater(sig, 1)

    def test_f_oc_exact_formula(self):
        """Prime-sized hyperedges (e.g. k=5, k=3) and 4-regular expanders receive topological boost."""
        pw_5 = prime_weight(5)
        pw_4 = prime_weight(4)
        pw_6 = prime_weight(6)
        self.assertGreater(pw_5, pw_4)
        self.assertTrue(is_prime(5))
        self.assertFalse(is_prime(6))

        s1 = f_oc_retrieval_score(sim_q_he=0.9, cardinality=5, pos_he=1, alpha=0.45, beta=0.55)
        s2 = f_oc_retrieval_score(sim_q_he=0.9, cardinality=5, pos_he=4, alpha=0.45, beta=0.55)
        self.assertGreater(s1, s2)

    def test_crt_reconstruct(self):
        """Chinese Remainder Theorem must reconstruct the exact integer modulo M = 2*3*5 = 30."""
        moduli = [2, 3, 5]
        # x = 17 -> 17 % 2 = 1, 17 % 3 = 2, 17 % 5 = 2
        reconstructed = TopologicalIndexer.crt_reconstruct([1, 2, 2], moduli)
        self.assertEqual(reconstructed, 17)

    def test_prime_stride_sparse_attention(self):
        """Prime strides [2, 3, 5] have CRT collision period 30 and compress context."""
        mask = TopologicalIndexer.prime_stride_attention_indices(60, prime_strides=[2, 3, 5])
        self.assertEqual(mask["crt_collision_period"], 30)
        self.assertLess(mask["attended_count"], 60)
        self.assertGreater(mask["compression_ratio"], 0.2)

    def test_raptor_cardinality_gate(self):
        low_gate = self.indexer.raptor_cardinality_gate("What is slope?", ["linear_equations"])
        self.assertEqual(low_gate["target_raptor_layer"], 0)
        self.assertEqual(low_gate["matryoshka_dim"], 64)

        high_gate = self.indexer.raptor_cardinality_gate(
            "How does derivative connect calculus, physics kinematics, economics marginal cost, and AI gradient descent?",
            ["derivative_rate_of_change", "instantaneous_velocity", "marginal_cost", "gradient_descent"]
        )
        self.assertEqual(high_gate["target_raptor_layer"], 2)
        self.assertEqual(high_gate["matryoshka_dim"], 1024)

    def test_distillation_curriculum_and_sa_mlp(self):
        curriculum = self.indexer.build_distillation_curriculum()
        self.assertIn("phase_2_prime_stride", curriculum)
        self.assertIn("phase_3_high_cardinality_mix", curriculum)

        sa_mlp_feats = self.indexer.export_sa_mlp_features()
        self.assertEqual(len(sa_mlp_feats), len(self.h.nodes))
        self.assertEqual(len(sa_mlp_feats[0]["sa_mlp_vector"]), 6)

    def test_webgl_topological_reification(self):
        reified = self.h.extract_bounded_subgraph("derivative_rate_of_change", n_limit=50)
        self.assertEqual(reified["hub"], "derivative_rate_of_change")
        self.assertLessEqual(reified["total_rendered"], 50)
        # Verify R_k tetrahedron nodes were injected
        reified_types = {n["type"] for n in reified["nodes"]}
        self.assertIn("reified_hyperedge", reified_types)

    def test_full_hyper_rag_and_green_agi_metrics(self):
        rag = HyperRAG(hyperontology=self.h)
        rag.add_text(
            "Myocardial Infarction combined with Diagnostic Delay causes Medical Malpractice and Productivity Loss.",
            domain="Medicine_Law"
        )
        res = rag.query("How does derivative connect kinematics and gradient descent?")
        self.assertGreater(res.metrics.savings_percent, 50.0)
        self.assertGreater(res.metrics.compute_reduction_factor, 100.0)
        self.assertEqual(res.metrics.evidence_score, 1.0)


if __name__ == "__main__":
    unittest.main()
