# Auto-Hyperontology (Hyper-RAG)

[![PyPI version](https://img.shields.io/badge/pypi-v0.2.0-blue.svg)](https://pypi.org/project/auto-hyperontology/)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Ollama Native](https://img.shields.io/badge/Ollama-Native-black.svg)](https://ollama.ai)
[![Author](https://img.shields.io/badge/Author-christianemaldonadomti-orange.svg)](https://github.com/christianemaldonadomti)

> **Universal $n$-ary Mathematical Hyperontology RAG, OEIS A320228 Ordinal-Cardinal Mixing, and Large-to-Small LLM Distillation Framework for Local LLMs (`intuyt:macro`, Ollama) and Proto-AGI Architectures.**
> Developed by [Dr. Christian Efraín Maldonado-Sifuentes](https://github.com/christianemaldonadomti) (CIC-IPN / SECIHTI / TRAI-L).

---

## 1. Executive Summary & Theoretical Motivation

While Large Language Models (LLMs) have demonstrated fluency, their reliance on linear token sequences (**ordinal**) and binary knowledge graphs ($E \subseteq V \times V$) introduces two severe bottlenecks:

1. **The Binary Reification Bottleneck & `LogicPoison` Vulnerability:** Standard Knowledge Graphs force polyadic real-world phenomena into artificial pairwise triples $(s, p, o)$. Reifying a simultaneous 4-entity medical-legal-economic interaction into binary edges explodes graph size and makes RAG systems vulnerable to topological exploits (`LogicPoison`).
2. **Context Flooding & Thermodynamic Entropy:** Linearizing dense subgraphs into a 1D prompt stream floods small LLM context windows, causing the *"lost-in-the-middle"* phenomenon and consuming massive compute/energy.

**Auto-Hyperontology** replaces binary graphs with Set-Theoretic **$n$-ary Mathematical Hyperontologies** $H = (X, E)$ where $E \subseteq \mathcal{P}(X) \setminus \{\emptyset\}$, governed by **General Systems Theory (GST)**, **Sabidussi–Vizing Prime Factorization**, and **OEIS A320228 Ordinal-Cardinal Mixing**.

---

## 2. Core Mathematical Architecture

### 2.1 The Exact A320228 Ordinal-Cardinal Mixing Function ($F_{OC}$)
Rather than ranking retrieved relations solely by flat vector similarity, `TopologicalIndexer` maps the **cardinality** $|H_e|$ of an $n$-ary hyperedge (its semantic width/magnitude) to its **ordinal** position in the linear LLM attention stream via the exact $F_{OC}$ scoring equation:

$$R_{score}(H_e) = \alpha \cdot \text{Sim}(Q, H_e) + \beta \cdot \text{PrimeWeight}(|H_e|) \cdot \frac{1}{\text{Pos}(H_e)}$$

Where:
* $\text{Sim}(Q, H_e)$: Semantic similarity weighted by **Dynamic Hypergraph Structure Learning (DyHSL)** intra-edge self-attention weights and cross-domain boundary coverage.
* $|H_e|$: Cardinality (number of simultaneously connected concepts in $H_e$).
* $\text{PrimeWeight}(|H_e|)$: Topological boost prioritizing irreducible prime-sized hyperedges ($|H_e| \in \{2, 3, 5, 7\dots\}$, which lack degenerate symmetries) and **4-regular simple graph expanders** (enumerated by [OEIS A320228](https://oeis.org/A320228): $1, 0, 0, 0, 0, 1, 1, 2, 6, 16, 59, 265, 1544\dots$).
* $\text{Pos}(H_e)$: Ordinal index in the **Sabidussi–Vizing Cartesian product prime factorization** sequence.

### 2.2 Coprime Prime Labeling & Chinese Remainder Theorem (CRT) Routing
Every vertex $v \in X$ is assigned a prime label $L(v) \in \{2, 3, 5, 7, 11\dots\}$ ordered by connectivity degree $D_v$ and boundary status, ensuring adjacent vertices are strictly coprime ($\gcd(L(u), L(v)) = 1$):
* **Unique Gödel/Sabidussi Signatures:** Each hyperedge $H_e$ has a collision-free composite invariant $\text{Sig}(H_e) = \prod_{v \in H_e} L(v)$.
* **CRT Distributed Sharding:** Reconstructs unique global topological states from modular remainders $r_i \equiv X \pmod{p_i}$.

### 2.3 Adaptive-RAG: RAPTOR Cardinality Gate & Matryoshka Unfolding
The reasoner evaluates query complexity via the **Cardinality Gate** $C(q) \rightarrow \{L_0, L_1, L_2\}$:
* **Low Cardinality ($|Q| \le 1$, Factoid):** Routes to **RAPTOR Layer $L_0$** (granular leaf chunks) and unfolds **Matryoshka Embeddings** to `64` dimensions.
* **Medium Cardinality ($|Q| \in \{2, 3\}$, Multi-Hop):** Routes to **RAPTOR Layer $L_1$** (cluster syntheses) and unfolds Matryoshka Embeddings to `512` dimensions.
* **High Cardinality ($|Q| \ge 4$ or Cross-Domain):** Routes to **RAPTOR Layer $L_2$** (root ontological abstractions + *Nodos Frontera*) and unfolds Matryoshka Embeddings to `1024` dimensions.

### 2.4 Prime Stride Sparse Attention
To prevent context saturation in compact student models (`intuyt:macro` 3B active MoE), attention heads stride at coprime prime intervals ($p \in \{2, 3, 5\}$). By the Chinese Remainder Theorem, heads striding by $2, 3,$ and $5$ only collide every $2 \times 3 \times 5 = 30$ tokens, maximizing long-range receptive field coverage at logarithmic token cost.

### 2.5 Large-to-Small LLM Distillation & Structure-Aware MLP (SA-MLP)
Supports distilling topological knowledge from a Teacher LLM/GNN into a message-passing-free Student **SA-MLP** (~10–15ms inference vs ~2000ms for recursive GNNs):
* **3-Phase Curriculum Learning:**
  * *Phase 1 (Dyadic Foundation):* $|H_e| = 2$
  * *Phase 2 (Prime Stride Integration):* $|H_e| \in \{3, 5, 7\}$
  * *Phase 3 (High-Cardinality Mix):* $|H_e| \ge 4$ (4-regular expanders & cross-domain complexes)
* **SA-MLP Structural Positional Encodings:** Exports `[D_v, L(v), Pos(H_e), max(|H_e|), is_boundary, coprime_mod_997]` per vertex.

### 2.6 WebGL Topological Reification (`N_limit = 300`)
To render 20,000+ node hyperontologies at 60 FPS in 3D Force-Directed WebGL visualizers ([trai-l.com/hyperonto/reader-math.php](https://trai-l.com/hyperonto/reader-math.php)), `extract_bounded_subgraph()` bounds local neighborhood extraction to $10 \le N_{limit} \le 300$ around the highest-degree hub $D_v$ and injects intermediate relational tetrahedrons $R_k$ with directed links $(v_s, R_k)$ and $(R_k, v_t)$.

---

## 3. Installation

```bash
pip install -e .
```

---

## 4. Quickstart: Python API

```python
from auto_hyperontology import HyperRAG

# 1. Initialize Universal Hyper-RAG (BPE chunk bound theta_max=3000, F_OC weights alpha=0.45, beta=0.55)
rag = HyperRAG(model="intuyt:macro", base_url="http://localhost:11434", theta_max=3000)

# 2. Ingest arbitrary unstructured texts, files, or directories
rag.add_documents([
    "In Clinical Cardiology, Myocardial Infarction occurs when Coronary Ischemia triggers Cellular Necrosis.",
    "When Diagnostic Delays cause preventable harm, Medical Malpractice Claims are filed under Healthcare Law.",
    "In Health Economics, post-cardiac disability induces Workplace Productivity Loss."
], domain="Medicine_Law_Economics")

# 3. Execute 5-Stage Proto-AGI Query (with RAPTOR Gate + F_OC Mixing + GreenAGI Metrics)
res = rag.query("How do diagnostic delays lead to malpractice claims and productivity loss?")

print(res.minimal_response_context)
print("\nAnswer:\n", res.answer)
print(f"\nToken Savings (Ahorro): {res.metrics.savings_percent}%")
print(f"GreenAGI Compute Reduction (3B MoE vs 400B Dense): {res.metrics.compute_reduction_factor}x")

# 4. Export 3-Phase Distillation Curriculum, SA-MLP Vectors, & WebGL Reification
curriculum = rag.get_distillation_curriculum()
sa_mlp_vectors = rag.export_sa_mlp_features()
rag.export_webgl_reification("hiperontologia_final.json", n_limit=300)
```

---

## 5. Command Line Interface (CLI)

```bash
# 1. Build & compile hyperontology + WebGL reified graph from any file or directory
autohyperonto build --input ./docs/ --output ontology.json --webgl hiperontologia_final.json --theta-max 3000

# 2. Query with tunable F_OC parameters (alpha, beta)
autohyperonto query --ontology ontology.json --question "Explain the cross-domain causal chain" --alpha 0.45 --beta 0.55

# 3. Inspect Coprime Labeling, OEIS A320228 Regularity, and Curriculum Stats
autohyperonto stats --ontology ontology.json

# 4. Export bounded WebGL 3D subgraph (10 <= N_limit <= 300)
autohyperonto export-webgl --ontology ontology.json --output hiperontologia_final.json --n-limit 300
```

---

## 6. References & Academic Citations

### Author's Foundational Research

```bibtex
@article{maldonado2025systemic,
  title={Systemic Knowledge Compilation: Operationalizing n-ary Hypergraphs for Local LLM Reasoning},
  author={Maldonado-Sifuentes, Christian Efra{\'\i}n and Sol{\'\i}s-Gamboa, Samuel and Vargas-Santiago, Mariano and Heras-Sanchez, Maria del Carmen and Lechuga-Gutierrez, Luis},
  journal={POLIBITS},
  volume={67},
  number={1},
  pages={27--32},
  year={2025},
  doi={10.17562/PB-67(1)-3}
}

@article{maldonado2025ordinal,
  title={Ordinal-Cardinal Mixing in Hyperontology Construction: A Theoretic Framework for Optimizing Large-to-Small LLM Knowledge Distillation and Adaptive Retrieval via OEIS A320228 Dynamics},
  author={Maldonado-Sifuentes, Christian Efra{\'\i}n},
  journal={Center for Computing Research (CIC-IPN)},
  year={2025}
}

@article{maldonado2024towards,
  title={Towards a Proto Artificial General Intelligence: The Role of Large Language Model Ontologies in its Development},
  author={Maldonado-Sifuentes, Christian E. and Vargas-Santiago, Mariano and Solis-Gamboa, Samuel and Sidorov, Grigori and Lechuga-Gutierrez, Luis and Gonz{\'a}lez-Andrade, F. and Heras-S{\'a}nchez, Maria},
  journal={Computaci{\'o}n y Sistemas},
  volume={28},
  number={4},
  pages={1401--1415},
  year={2024}
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

@article{maldonado2020compression,
  title={Knowledge and Context Compression via Question Generation},
  author={Maldonado-Sifuentes, Christian E. and Sidorov, Grigori and Kolesnikova, Olga and Caballero, C.},
  journal={Research in Computing Science},
  year={2020}
}
```

### Seminal Literature

* **General Systems Theory & Negentropy:** von Bertalanffy, L. (1968). *General System Theory: Foundations, Development, Applications*. George Braziller.
* **Hyperontologies & Distributed Ontology Language (DOL):** Kutz, O., Mossakowski, T., Hastings, J., & Neuhaus, F. (2011). *Hyperontology for the Biomedical Ontologist*. CEUR-WS Vol. 833; Mossakowski, T., et al. (2015). *The Distributed Ontology, Modeling and Specification Language (DOL)*.
* **Basic Formal Ontology (BFO):** Smith, B., et al. (2015). *Basic Formal Ontology 2.0: Specification and User's Guide*.
* **Cartesian Graph Factorization & Primes:** Sabidussi, G. (1960). *The Cartesian Product of Graphs*. Math. Z., 72(1), 446–457; Vizing, V. G. (1963). *The Cartesian Product of Graphs*. Vychisl. Sistemy, 9, 30–43; Sloane, N. J. A. *OEIS Sequence A320228*.
* **RAPTOR, Matryoshka & SA-MLP Distillation:** Sarthi, P., et al. (2024). *RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval* (arXiv:2401.18059); Kusupati, A., et al. (2022). *Matryoshka Representation Learning* (NeurIPS); Chen, J., et al. (2022). *SA-MLP: Distilling Graph Knowledge from GNNs into Structure-Aware MLP* (arXiv:2210.09609).

---

## Author & Contact

* **Dr. Christian Efraín Maldonado-Sifuentes**
* **GitHub:** [https://github.com/christianemaldonadomti](https://github.com/christianemaldonadomti)
* **Visual Environment:** [https://trai-l.com/hyperonto/reader-math.php](https://trai-l.com/hyperonto/reader-math.php)
* **License:** Apache License 2.0
