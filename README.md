# Auto-Hyperontology (`Hyper-RAG` v0.4.0)

[[https://img.shields.io/badge/pypi-v0.4.0-blue.svg]](https://pypi.org/project/auto-hyperontology/)
[[https://img.shields.io/badge/License-Apache%202.0-green.svg]](https://opensource.org/licenses/Apache-2.0)
[[https://img.shields.io/badge/python-3.9+-blue.svg]](https://www.python.org/downloads/)
[[https://img.shields.io/badge/Author-christianemaldonadomti-orange.svg]](https://github.com/christianemaldonadomti)
[[https://colab.research.google.com/assets/colab-badge.svg]](https://colab.research.google.com/github/christianemaldonadomti/auto-hyperontology/blob/main/notebooks/Auto_Hyperontology_Colab_Benchmark.ipynb)

> **Universal $n$-ary Mathematical Hyperontology RAG, Proto-AGI v4.0 Forensic Reasoning (`[THINK]`/`[REF]`/`[NAT]`), and OEIS A320228 Ordinal-Cardinal Mixing for Local Language Models (Ollama).**  
> Developed by [Dr. Christian Efraín Maldonado-Sifuentes](https://github.com/christianemaldonadomti) (CIC-IPN / SECIHTI / TRAI-L).

---

## 🚀 Live Evaluation Benchmark (PubMedQA: `Hyper-RAG` vs. `Okapi BM25 Flat RAG`)

Recent live evaluation over real biomedical articles from **Hugging Face `qiaojin/PubMedQA`** (`pqa_labeled`) demonstrates the decisive quality and grounding advantages of compiling knowledge into an $n$-ary hypergraph ontology $H = (X, E)$ compared to standard chunk-based retrieval:


```

## ===================================================================================================
MÉTRICA                                      | Flat Text RAG (BM25) | Hyper-RAG v0.4.0 (Ours) | Ganancia

# Faithfulness (Soporte Factual de Claims) ↑  |        70.3%         |         81.1%           |  +10.8%
Tokens no Fundamentados (Alucinación) ↓     |        29.7%         |         18.9%           |  -10.8% (36% menos alucinación)
Recall de Conceptos Clave (Ground-Truth) ↑   |        39.9%         |         51.9%           |  +12.0%

```

> **⚡ Active Engineering Roadmap: Context & Token Compression:**  
> In the current Proto-AGI v4.0 forensic release, the initial expansion stage preserves the complete neighborhood subgraph (`CURRENT GRAPH SLICE`) containing rich node attributes, edge constraints, and provenance notes into the prompt (~5,000 prompt tokens on multi-document slices vs. ~456 tokens for isolated BM25 text chunks).  
> **We are currently actively developing a topological compression phase to drastically reduce prompt token overhead** by combining:
> 1. Dynamic cardinality thresholding to prune peripheral node attributes before prompt generation.
> 2. Compressing multi-hop slices through Chinese Remainder Theorem (CRT) coprime remainder indexing.
> 3. Prime Stride Sparse Attention projections to achieve sub-linear prompt token footprints without sacrificing the +10.8% faithfulness gain.

---

## Core Architecture (Proto-AGI v4.0 Forensic + OEIS A320228)

1. **Rich Node & $n$-ary Hyperedge Schema (`biol1.json` compatible):**
   * **Nodes:** `id`, `name`, `categories`, `labels`, and rich `attributes` preserving all definitions, quantitative findings, mechanisms, and examples.
   * **Hyperedges:** `id`, `type`, `relation`, `connects`, `constraints`, and detailed `notes`.
2. **Semantic Bridge:** Extracts 3 core English domain nouns via the LLM to bridge natural-language and multilingual queries directly into the ontology namespace.
3. **1-Hop Neighborhood Subgraph Expansion + $F_{OC}$ Ordinal-Cardinal Mixing:**
   $$R_{score}(H_e) = \alpha \cdot \text{Sim}(Q, H_e) + \beta \cdot \text{PrimeWeight}(\vert{}H_e\vert{}) \cdot \frac{1}{\text{Pos}(H_e)}$$
   Loads all connected neighbor nodes and serializes the complete `CURRENT GRAPH SLICE` JSON.
4. **Structured ReAct Forensic Output:** Generates `[THINK]` (internal topological trace), `[REF]` (technical explanation citing `[node_id]` and `[edge_id]`), and `[NAT]` (fluid natural language synthesis).

---

## Installation & Configuration

```bash
pip install auto-hyperontology

```

Or install in development mode from GitHub:

```bash
git clone [https://github.com/christianemaldonadomti/auto-hyperontology.git](https://github.com/christianemaldonadomti/auto-hyperontology.git)
cd auto-hyperontology
pip install -e .

```

To configure private local or remote Ollama endpoints without exposing credentials in code, copy `.env.example` to `.env`:

```bash
cp .env.example .env

```

And set your local variables:

```bash
OLLAMA_ENDPOINT=http://localhost:11434
OLLAMA_EXTRACTION_MODEL=llama3.2:3b
OLLAMA_REASONING_MODEL=llama3.2:3b

```

---

## Quickstart: Python API

```python
from auto_hyperontology import HyperRAG

# 1. Initialize RAG
rag = HyperRAG(model="llama3.2:3b", base_url="http://localhost:11434")

# 2. Ingest arbitrary text corpora or load existing hyperontology (biol1.json)
rag.add_documents([
    "In Clinical Cardiology, Myocardial Infarction occurs when Coronary Ischemia triggers Cellular Necrosis.",
    "When Diagnostic Delays cause preventable harm, Medical Malpractice Claims are filed under Healthcare Law.",
    "In Health Economics, post-cardiac disability induces Workplace Productivity Loss."
], domain="Medicine_Law_Economics")

# 3. Query with 5-stage topological grounding & ReAct forensic trace
res = rag.query("How do diagnostic delays lead to malpractice claims and productivity loss?")

print("THINK BLOCK:\n", res.think_block)
print("\nTECHNICAL REF CITATIONS:\n", res.ref_block)
print("\nNATURAL ANSWER:\n", res.nat_block)

```

---

## References & Academic Citations

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

```

---

## License

Distributed under the **Apache License 2.0**. See `LICENSE` for details.

```
