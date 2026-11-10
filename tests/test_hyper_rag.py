"""
Comprehensive Unit & Mathematical Verification Suite for Auto-Hyperontology (Hyper-RAG)
100% Corpus-Agnostic (zero Khan Academy hardcoding).
"""

import unittest
from auto_hyperontology.core import Hyperontology, HyperNode, HyperEdge, BoundaryNode
from auto_hyperontology.topology import (
    TopologicalIndexer,
    f_oc_retrieval_score,
    prime_weight,
    is_prime,
)
from auto_hyperontology.rag import HyperRAG


class TestOrdinalCardinalHyperontology(unittest.TestCase):
    def setUp(self):
        self.rag = HyperRAG()
        self.rag.add_text(
            "In Clinical Cardiology, Myocardial Infarction occurs when Coronary Ischemia triggers Cellular Necrosis. "
            "When Delayed Diagnostic Protocols compound Myocardial Infarction, Medical Malpractice Claims are filed "
            "and Workplace Productivity Loss occurs.",
            domain="Medicine_Law_Economics"
        )
        self.rag.add_text(
            "In Deep Learning, Myocardial Infarction risk prediction models use Gradient Descent Optimization "
            "and Backpropagation over clinical electronic health records.",
            domain="ComputerScience_AI"
        )
        self.h = self.rag.hyper
        self.indexer = self.rag.reasoner.indexer

    def test_boundary_node_and_degrees(self):
        self.assertGreaterEqual(len(self.h.boundary_nodes), 1)
        self.assertIn("myocardial_infarction", self.h.boundary_nodes)

    def test_coprime_prime_labeling(self):
        self.assertTrue(self.indexer.verify_coprime_adjacency())
        for edge in self.h.edges.values():
            sig = self.indexer.compute_edge_signature(edge)
            self.assertGreater(sig, 1)

    def test_f_oc_exact_formula(self):
        pw_5 = prime_weight(5)
        pw_4 = prime_weight(4)
        self.assertGreater(pw_5, pw_4)
        self.assertTrue(is_prime(5))
        self.assertFalse(is_prime(6))

        s1 = f_oc_retrieval_score(sim_q_he=0.9, cardinality=5, pos_he=1, alpha=0.45, beta=0.55)
        s2 = f_oc_retrieval_score(sim_q_he=0.9, cardinality=5, pos_he=4, alpha=0.45, beta=0.55)
        self.assertGreater(s1, s2)

    def test_crt_reconstruct(self):
        moduli = [2, 3, 5]
        reconstructed = TopologicalIndexer.crt_reconstruct([1, 2, 2], moduli)
        self.assertEqual(reconstructed, 17)

    def test_prime_stride_sparse_attention(self):
        mask = TopologicalIndexer.prime_stride_attention_indices(60, prime_strides=[2, 3, 5])
        self.assertEqual(mask["crt_collision_period"], 30)
        self.assertLess(mask["attended_count"], 60)

    def test_raptor_cardinality_gate(self):
        low_gate = self.indexer.raptor_cardinality_gate("What is necrosis?", ["cellular_necrosis"])
        self.assertEqual(low_gate["target_raptor_layer"], 0)
        self.assertEqual(low_gate["matryoshka_dim"], 64)

        high_gate = self.indexer.raptor_cardinality_gate(
            "How does Myocardial Infarction connect Delayed Diagnostic Protocols, Medical Malpractice Claims, and Gradient Descent Optimization?",
            ["myocardial_infarction", "delayed_diagnostic_protocols", "medical_malpractice_claims", "gradient_descent_optimization"]
        )
        self.assertEqual(high_gate["target_raptor_layer"], 2)
        self.assertEqual(high_gate["matryoshka_dim"], 1024)

    def test_distillation_curriculum_and_sa_mlp(self):
        curriculum = self.indexer.build_distillation_curriculum()
        self.assertIn("phase_2_prime_stride", curriculum)
        sa_mlp_feats = self.indexer.export_sa_mlp_features()
        self.assertEqual(len(sa_mlp_feats), len(self.h.nodes))

    def test_webgl_topological_reification(self):
        reified = self.h.extract_bounded_subgraph("myocardial_infarction", n_limit=50)
        self.assertEqual(reified["hub"], "myocardial_infarction")
        self.assertLessEqual(reified["total_rendered"], 50)
        reified_types = {n["type"] for n in reified["nodes"]}
        self.assertIn("reified_hyperedge", reified_types)

    def test_full_hyper_rag_and_green_agi_metrics(self):
        res = self.rag.query("How does Myocardial Infarction relate to Medical Malpractice Claims?")
        self.assertGreater(res.metrics.savings_percent, 50.0)
        self.assertGreater(res.metrics.compute_reduction_factor, 100.0)


if __name__ == "__main__":
    unittest.main()
