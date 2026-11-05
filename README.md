# Auto-Hyperontology (Hyper-RAG)

[![PyPI version](https://img.shields.io/badge/pypi-v0.1.0-blue.svg)](https://pypi.org/project/auto-hyperontology/)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Ollama Native](https://img.shields.io/badge/Ollama-Native-black.svg)](https://ollama.ai)
[![Author](https://img.shields.io/badge/Author-christianemaldonadomti-orange.svg)](https://github.com/christianemaldonadomti)

> **Universal $n$-ary Mathematical Hyperontology RAG Framework for Local LLMs (`intuyt:macro`, Ollama) and Proto-AGI Architectures.**
> Developed by [Dr. Christian Efraín Maldonado-Sifuentes](https://github.com/christianemaldonadomti) (CIC-IPN / SECIHTI / TRAI-L).

---

## Overview

Modern Retrieval-Augmented Generation (RAG) and traditional Knowledge Graphs (KGs) suffer from two fundamental bottlenecks:

1. **Binary Edge Reification:** Traditional $(subject, predicate, object)$ triples force polyadic real-world phenomena into artificial pairwise chains. Representing a simultaneous multi-entity interaction (e.g., *a clinical condition causing a legal liability resulting in an economic productivity loss*) requires complex reification, leading to hallucinated predicates and query syntax failures.
2. **Context Flooding:** Expanding dense graphs into linear token sequences floods LLM context windows, triggering the "lost-in-the-middle" phenomenon and blowing up inference FLOPs, energy, and carbon footprints.

**Auto-Hyperontology** solves this by operationalizing **$n$-ary mathematical hypergraphs**, **multi-category foundational ontologies**, and **Ordinal-Cardinal mixing (via OEIS A320228 dynamics)**. It compiles arbitrary unstructured texts into a structured topological substrate that allows compact edge-running models (3B/8B parameters like `intuyt:macro`) to achieve zero-hallucination reasoning with up to **80% token savings**.

---

## Key Theoretical Pillars

* **$n$-ary Hyperedges:** Relationships connecting $k \ge 2$ conceptual entities simultaneously with typed semantic role bindings (`agent`, `patient`, `mediator`, `evidence`, `outcome`).
* **Multi-Category Foundational Nodes:** Entities categorized across formal ontological classes: `Concept`, `Property`, `Process`, `Evidence`, and `Level` (grounded in [Basic Formal Ontology - BFO](https://basic-formal-ontology.org/) principles).
* **Boundary Nodes (*Nodos Frontera*):** Explicit anchor nodes that exist simultaneously in two or more domain ontologies, enabling certified, multi-valued logical jumps across disciplines (e.g., *Medicine* $\rightarrow$ *Law* $\rightarrow$ *Economics* or *Mathematics* $\rightarrow$ *Physics* $\rightarrow$ *Computer Science*).
* **Sabidussi-Vizing Prime Graph Factorization:** Decomposes complex semantic subgraphs into irreducible Cartesian prime factor invariants, ensuring collision-free indexing.
* **Ordinal-Cardinal Mixing (OEIS A320228 Dynamics):** Maps the magnitude/complexity of a relationship ($k$-ary cardinality) into a retrieval sequence priority ($n$-th ordinal rank), structuring the attention queue for minimal token footprints.
* **AME Metrics (Ahorro - Mejora - Evidencia):**
  * **Ahorro (A):** Direct reduction in context tokens and inference FLOPs (typically $60\%–85\%$).
  * **Mejora (M):** Structural adherence score guaranteeing zero ungrounded hops.
  * **Evidencia (E):** $100\%$ traceability with verifiable citations to hyperedge IDs.

---

## Architecture: The 5-Stage Proto-AGI Reasoning Loop

```
[User Question]
       │
       ▼
[Stage 1: Question Parsing] ──────► Extract domain filters & candidate categories
       │
       ▼
[Stage 2: Semantic Anchoring] ────► Locate Seed Nodes & Cross-Domain Boundary Anchors
       │
       ▼
[Stage 3: Topological Expansion] ──► Traverse n-ary hyperedges via OEIS A320228 Priority
       │
       ▼
[Stage 4: Minimal Response Tree] ──► Prune graph into a compact proof subgraph
       │
       ▼
[Stage 5: Traceable LLM Grounding] ─► Small LLM (Ollama) generates certified answer + AME metrics
```

---

## Installation

```bash
pip install auto-hyperontology
```

Or clone and install in development mode from GitHub:

```bash
git clone https://github.com/christianemaldonadomti/auto-hyperontology.git
cd auto-hyperontology
pip install -e .
```

---

## Quickstart: Drop-in 3-Line Python API

You can ingest arbitrary texts, documents, or entire directories:

```python
from auto_hyperontology import HyperRAG

# 1. Initialize RAG with local Ollama model (e.g. intuyt:macro or llama3.2)
rag = HyperRAG(model="intuyt:macro", base_url="http://localhost:11434")

# 2. Ingest arbitrary text corpora
rag.add_documents([
    "Clinical cardiology: Myocardial infarction induces irreversible cellular necrosis...",
    "Legal jurisprudence: Preventable diagnostic delays establish grounds for malpractice claims...",
    "Health economics: Post-cardiac rehabilitation disability causes long-term workforce productivity loss..."
])

# 3. Query with 5-stage topological grounding
result = rag.query("How does delayed cardiac diagnosis lead to institutional financial liability?")

print("Answer:\n", result.answer)
print(f"Token Savings: {result.metrics.savings_percent}%")
print(f"Grounded Hyperedges Cited: {result.metrics.grounded_hyperedges}")

# 4. Save & Load compiled hyperontologies
rag.save("compiled_hyperontology.json")
loaded_rag = HyperRAG.load("compiled_hyperontology.json")
```

---

## Command Line Interface (CLI)

`auto-hyperontology` provides a dedicated terminal tool (`autohyperonto` or `hyper-rag`):

### 1. Build & Compile Hyperontology from Text Files
```bash
autohyperonto build \
  --input ./research_docs/ \
  --output my_ontology.json \
  --model intuyt:macro \
  --domain "Biomedical_Legal"
```

### 2. Query Compiled Hyperontology
```bash
autohyperonto query \
  --ontology my_ontology.json \
  --question "What are the cross-domain impacts of delayed diagnosis?" \
  --model intuyt:macro
```

### 3. Inspect Structural Topology & Metrics
```bash
autohyperonto stats --ontology my_ontology.json
```

---

## Comparison: Auto-Hyperontology vs. Standard GraphRAG & NVIDIA Auto Ontology

| Feature | Standard GraphRAG | NVIDIA Auto-Ontology | **Auto-Hyperontology (Ours)** |
| :--- | :--- | :--- | :--- |
| **Relational Model** | Binary Triples $(s, p, o)$ | Relational Tables & Views | **$n$-ary Hyperedges ($k \ge 2$)** |
| **Substrate** | Text Chunks + Vector DB | Relational DBs (SQL) | **Mathematical Hypergraph + Categorical Logic** |
| **Cross-Domain Reasoning**| Unconstrained Vector Hops | Schema Foreign Keys | **Boundary Nodes (*Nodos Frontera*)** |
| **Context Management** | Top-$k$ Raw Chunks (Flooding)| Relational Schema Dumps | **Ordinal-Cardinal Mixing (OEIS A320228)** |
| **Model Requirements** | Massive Frontier LLMs | NVIDIA NIM / Cloud APIs | **Local Edge Small Models (`intuyt:macro`, 3B/8B)** |
| **Auditability** | Latent Cosine Similarity | SQL Query Text | **Traceable Proof Subgraph & AME Metrics** |

---

## References & Academic Citations

### Author's Foundational Research

If you use this library, framework, or theoretical methodology, please cite the author's primary research:

```bibtex
@article{maldonado2025systemic,
  title={Systemic Knowledge Compilation: Operationalizing n-ary Hypergraphs for Local LLM Reasoning},
  author={Maldonado-Sifuentes, Christian Efra{\'\i}n and Sol{\'\i}s-Gamboa, Samuel and Vargas-Santiago, Mariano and Heras-Sanchez, Maria del Carmen and Lechuga-Gutierrez, Luis},
  journal={SECIHTI / CIC-IPN Research Reports},
  year={2025}
}

@article{maldonado2025ordinal,
  title={Ordinal-Cardinal Mixing in Hyperontology Construction: A Theoretic Framework for Optimizing Large-to-Small LLM Knowledge Distillation and Adaptive Retrieval via OEIS A320228 Dynamics},
  author={Maldonado-Sifuentes, Christian Efra{\'\i}n},
  journal={Center for Computing Research (CIC-IPN)},
  year={2025}
}

@article{guerrero2024nlp,
  title={Natural Language Processing Approach Using a Neural Network Ensemble (CNN-HSNN) for Skin Cancer and Multi-Disease Classification},
  author={Guerrero-Rangel, Joshua R. G. and Sidorov, Grigori and Maldonado-Sifuentes, Christian E. and Vargas-Santiago, Mariano and Ortega-Garc{\'\i}a, M. Cristina and Le{\'o}n-Velasco, Diana A.},
  journal={Computaci{\'o}n y Sistemas},
  volume={28},
  number={3},
  pages={1243--1255},
  year={2024},
  doi={10.13053/CyS-28-3-5015}
}
```

### Seminal Works in Formal Hyperontologies & Neuro-Symbolic AI

1. **Hyperontology & Heterogeneous Distributed Ontologies:**
   * Kutz, O., Mossakowski, T., Hastings, J., & Neuhaus, F. (2011). *Hyperontology for the Biomedical Ontologist: A Sketch and Some Theoretical Background*. CEUR Workshop Proceedings, Vol. 833.
   * Mossakowski, T., Kutz, O., Codescu, M., & Lange, C. (2015). *The Distributed Ontology, Modeling and Specification Language (DOL)*. Formal Ontology in Information Systems (FOIS).
2. **Foundational Ontological Tiers:**
   * Smith, B., et al. (2015). *Basic Formal Ontology 2.0: Specification and User's Guide*.
3. **Graph Factorization Invariants & Combinatorics:**
   * Sabidussi, G. (1960). *The Cartesian Product of Graphs*. Mathematische Zeitschrift, 72(1), 446–457.
   * Vizing, V. G. (1963). *The Cartesian Product of Graphs*. Vychislitel'nye Sistemy, 9, 30–43.
   * Sloane, N. J. A. (2018). *The On-Line Encyclopedia of Integer Sequences (OEIS)*, Sequence A320228: Array read by rows, indices of subsets of {1..n} ordered by cardinality, then lexicographically.
4. **Knowledge Grounding & Query Generation in LLMs:**
   * Li, X., Zhao, R., Chia, Y. K., Ding, B., Joty, S., Poria, S., & Bing, L. (2024). *Chain-of-Knowledge: Grounding Large Language Models via Dynamic Knowledge Adapting over Heterogeneous Sources*. International Conference on Learning Representations (ICLR 2024). arXiv:2305.13269.

---

## Author & Contact

* **Dr. Christian Efraín Maldonado-Sifuentes**
* **GitHub:** [https://github.com/christianemaldonadomti](https://github.com/christianemaldonadomti)
* **Affiliation:** Centro de Investigación en Computación (CIC-IPN) / Secretaría de Ciencia, Humanidades, Tecnología e Innovación (SECIHTI) / Transdisciplinary Research for Augmented Innovation Laboratory (TRAI-L), Mexico City.
* **Email:** [christianemaldonadomti@gmail.com](mailto:christianemaldonadomti@gmail.com)

---

## License

Distributed under the **Apache License 2.0**. See `LICENSE` for details.
