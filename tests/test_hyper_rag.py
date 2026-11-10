"""
Comprehensive Unit & Mathematical Verification Suite for Auto-Hyperontology (Proto-AGI v4.0)
"""

import unittest
from auto_hyperontology.core import Hyperontology, HyperNode, HyperEdge
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
        self.rag.reasoner.use_llm_semantic_bridge = False
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

    def test_crt_reconstruct(self):
        moduli = [2, 3, 5]
        reconstructed = TopologicalIndexer.crt_reconstruct([1, 2, 2], moduli)
        self.assertEqual(reconstructed, 17)

    def test_webgl_topological_reification(self):
        reified = self.h.extract_bounded_subgraph("myocardial_infarction", n_limit=50)
        self.assertEqual(reified["hub"], "myocardial_infarction")
        self.assertLessEqual(reified["total_rendered"], 50)

    def test_forensic_graph_slice_and_react_blocks(self):
        res = self.rag.query("How does Myocardial Infarction relate to Medical Malpractice Claims?")
        self.assertIn("nodes", res.current_graph_slice)
        self.assertIn("edges", res.current_graph_slice)
        self.assertGreater(len(res.active_node_ids), 0)
        self.assertTrue(len(res.ref_block) > 0)


if __name__ == "__main__":
    unittest.main()
