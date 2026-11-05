"""
Unit Tests for Auto-Hyperontology (Hyper-RAG)
"""

import unittest
from auto_hyperontology.core import Hyperontology, HyperNode, HyperEdge, BoundaryNode
from auto_hyperontology.topology import TopologicalIndexer, oeis_a320228_weight
from auto_hyperontology.rag import HyperRAG


class TestHyperontology(unittest.TestCase):
    def setUp(self):
        self.h = Hyperontology(name="TestHyperontology")
        self.node1 = HyperNode(id="n1", name="Node 1", domains=["DomainA"], categories=["Concept"])
        self.node2 = HyperNode(id="n2", name="Node 2", domains=["DomainA", "DomainB"], categories=["Process"])
        self.node3 = HyperNode(id="n3", name="Node 3", domains=["DomainB"], categories=["Property"])

        self.h.add_node(self.node1)
        self.h.add_node(self.node2)
        self.h.add_node(self.node3)

        self.edge = HyperEdge(
            id="edge_1",
            nodes=["n1", "n2", "n3"],
            relation_type="CrossDomainBridge",
            domain_scope=["DomainA", "DomainB"]
        )
        self.h.add_edge(self.edge)

    def test_boundary_node_detection(self):
        self.assertFalse(self.node1.is_boundary)
        self.assertTrue(self.node2.is_boundary)
        self.assertIn("n2", self.h.boundary_nodes)

    def test_hyperedge_cardinality(self):
        self.assertEqual(self.edge.cardinality, 3)

    def test_topological_indexer(self):
        indexer = TopologicalIndexer(self.h)
        sig = indexer.compute_edge_signature(self.edge)
        self.assertGreater(sig, 0)

    def test_oeis_a320228_weight(self):
        w1 = oeis_a320228_weight(1, 4)
        w2 = oeis_a320228_weight(5, 4)
        self.assertGreater(w1, w2)

    def test_hyper_rag_pipeline(self):
        rag = HyperRAG(hyperontology=self.h)
        res = rag.query("How does Node 1 connect with Node 2?")
        self.assertIsNotNone(res.answer)
        self.assertGreater(res.metrics.savings_percent, 0.0)


if __name__ == "__main__":
    unittest.main()
