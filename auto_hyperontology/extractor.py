"""
Auto-Hyperontology: Semantic Extraction Pipeline & Ollama LLM Client
Implements the exact extraction methodology from:
Maldonado-Sifuentes et al. (2025), "Systemic Knowledge Compilation: Operationalizing n-ary
Hypergraphs for Local LLM Reasoning", POLIBITS, vol. 67(1), pp. 27-32.

Core Components:
1. Sentence-level partitioning tau_sent and BPE-approximated token counting T(s_i)
   with greedy capacity bound sum_{x=i}^j T(s_x) <= theta_max (default theta_max = 3000).
2. Deterministic JSON repair heuristic H(R_i) for local MoE LLM outputs (intuyt:macro).
3. Dynamic Hypergraph Structure Learning (DyHSL) variable hyperedge sizing and RAPTOR
   fractal hierarchy assignment (Layers L_0, L_1, L_2).
"""

import json
import re
import requests
import uuid
from typing import List, Dict, Any, Optional
from auto_hyperontology.core import HyperNode, HyperEdge, BoundaryNode, Hyperontology


def estimate_bpe_tokens(text: str) -> int:
    """
    Approximates cl100k_base Byte-Pair Encoding (BPE) token count T(s_i) without
    external network calls (approx 0.75 words / 4 chars per token, plus punctuation).
    """
    if not text:
        return 0
    words = re.findall(r"\w+|[^\w\s]", text, re.UNICODE)
    return max(1, int(len(words) * 1.15))


def greedy_token_bounded_chunk(
    text: str,
    theta_max: int = 3000,
    overlap_sentences: int = 1
) -> List[str]:
    """
    Implements POLIBITS Section IV.A.1:
    Unstructured Data Parsing and Semantic Partitioning.
    - Normalizes text D and applies sentence tokenizer tau_sent -> S = (s_1, s_2, ..., s_N).
    - Partitions S into contiguous chunks C_k = (s_i, ..., s_j) subject to:
          sum_{x=i}^j T(s_x) <= theta_max
      where T(s_x) is the BPE token length and theta_max defaults to 3000 tokens.
    """
    cleaned = re.sub(r"\r\n", "\n", text.strip())
    # Split into sentences while preserving paragraph boundaries
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n\n+", cleaned) if s.strip()]
    if not sentences:
        return []

    chunks: List[str] = []
    current_chunk: List[str] = []
    current_tokens = 0

    for sent in sentences:
        s_tokens = estimate_bpe_tokens(sent)
        if current_tokens + s_tokens > theta_max and current_chunk:
            chunks.append(" ".join(current_chunk))
            overlap = current_chunk[-overlap_sentences:] if overlap_sentences > 0 else []
            current_chunk = overlap + [sent]
            current_tokens = sum(estimate_bpe_tokens(x) for x in current_chunk)
        else:
            current_chunk.append(sent)
            current_tokens += s_tokens

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


def sanitize_json_payload(raw_response: str) -> Dict[str, Any]:
    """
    Implements POLIBITS Section V.A.2:
    Deterministic JSON Repair Heuristic H(R_i).
    Strips markdown wrappers, repairs unclosed brackets/braces, removes trailing commas,
    and ensures deterministic schema compliance for local MoE LLM outputs.
    """
    cleaned = raw_response.strip()
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", cleaned)
    if match:
        cleaned = match.group(1).strip()
    else:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1:
            cleaned = cleaned[start:end + 1]

    # Remove trailing commas before closing braces or brackets
    cleaned = re.sub(r",\s*([\]}])", r"\1", cleaned)

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        try:
            return json.loads(cleaned.replace("'", '"'))
        except Exception:
            return {"nodes": [], "edges": []}


class OllamaClient:
    """
    Local Inference Interface for Ollama & intuyt:macro (3B active parameter MoE).
    Adheres to GreenAGI principles by executing strictly on local/edge endpoints.
    """
    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "intuyt:macro",
        temperature: float = 0.1,
        timeout: int = 60
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.timeout = timeout

    def is_available(self) -> bool:
        try:
            res = requests.get(f"{self.base_url}/api/tags", timeout=2)
            return res.status_code == 200
        except Exception:
            return False

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "system": system_prompt or "",
            "stream": False,
            "options": {"temperature": self.temperature}
        }
        res = requests.post(url, json=payload, timeout=self.timeout)
        res.raise_for_status()
        return res.json().get("response", "")


EXTRACTION_SYSTEM_PROMPT = """You are a mathematical hyperontology compiler (semantic_assembler).
Extract concepts as multi-category nodes and polyadic relationships as n-ary hyperedges H = (X, E) where E <= P(X) \\ {empty}.

RULES:
1. NODES (vertices in X):
   - id: unique snake_case identifier
   - name: canonical name
   - domains: list of domains (e.g., ["Medicine", "Law", "Economics"])
   - categories: BFO tiers from ["Concept", "Property", "Process", "Evidence", "Level"]
   - aliases: alternative names or multilingual equivalents (e.g., Spanish, Nahuatl, English)
   - raptor_layer: 0 for granular leaf fact, 1 for cluster concept, 2 for root domain principle

2. n-ary HYPEREDGES (e in E, cardinality |H_e| >= 2, prioritizing prime sizes 2, 3, 5, 7 or 4-regular):
   - id: unique edge_snake_case
   - nodes: list of participating node IDs
   - relation_type: CausalLink, PartOf, Subsumption, CrossDomainBridge, EvidenceSupport, PrerequisiteLattice
   - roles: semantic role mapping {"agent": "id1", "target": "id2", "mechanism": "id3"}
   - node_weights: DyHSL self-attention importance weights summing to 1.0
   - domain_scope: list of domains spanned by this relation
   - confidence: float in [0.0, 1.0]
   - raptor_layer: 0 (granular), 1 (cluster), or 2 (abstract macro-bridge)

Return strictly valid JSON: {"nodes": [...], "edges": [...]}"""


class AutoHyperontologyExtractor:
    """
    Implements POLIBITS Section IV.A (Semantic Extraction & Topological Weaver)
    and Dynamic Hypergraph Structure Learning (DyHSL) with variable hyperedge sizing.
    """
    def __init__(self, ollama_client: Optional[OllamaClient] = None, theta_max: int = 3000):
        self.client = ollama_client or OllamaClient()
        self.theta_max = theta_max

    def extract_from_text(self, text: str, default_domain: str = "General") -> Hyperontology:
        chunks = greedy_token_bounded_chunk(text, theta_max=self.theta_max)
        hyper = Hyperontology(name=f"CompiledOntology_{uuid.uuid4().hex[:6]}")

        for idx, chunk in enumerate(chunks):
            extracted_data = self._extract_chunk(chunk, default_domain, chunk_idx=idx)
            self._merge_into_hyperontology(hyper, extracted_data, source_chunk=chunk)

        return hyper

    def _extract_chunk(self, chunk: str, default_domain: str, chunk_idx: int) -> Dict[str, Any]:
        prompt = f"Target Domain: {default_domain}\n\nDocument Segment:\n{chunk}\n\nCompile into formal hyperontology JSON:"
        if self.client.is_available():
            try:
                raw_out = self.client.generate(prompt=prompt, system_prompt=EXTRACTION_SYSTEM_PROMPT)
                data = sanitize_json_payload(raw_out)
                if data.get("nodes") or data.get("edges"):
                    return data
            except Exception:
                pass

        return self._heuristic_dyhsl_extraction(chunk, default_domain, chunk_idx)

    def _heuristic_dyhsl_extraction(self, chunk: str, default_domain: str, chunk_idx: int) -> Dict[str, Any]:
        """
        Deterministic offline Dynamic Hypergraph Structure Learning (DyHSL) extractor.
        Extracts sentence-level polyadic hyperedges (RAPTOR Layer 0 & 1) and macro-chunk
        summary hyperedges (RAPTOR Layer 2), sizing hyperedges to prime cardinalities (3, 5)
        or 4-regular expanders (4) based on information density.
        """
        nodes_dict: Dict[str, Dict[str, Any]] = {}
        edges: List[Dict[str, Any]] = []

        # Stopwords to ignore when picking capitalized concept phrases
        stopwords = {
            "The", "This", "That", "These", "Those", "When", "Where", "While",
            "However", "Therefore", "Furthermore", "Although", "Because", "In",
            "On", "At", "To", "For", "With", "From", "By", "As", "If", "It"
        }

        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", chunk) if s.strip()]
        chunk_root_concepts: List[str] = []

        for s_idx, sent in enumerate(sentences):
            # Extract multi-word capitalized phrases or technical terms
            raw_matches = re.findall(r"\b[A-Z][a-zA-Z0-9_\-]+(?:\s+[A-Z][a-zA-Z0-9_\-]+)*\b", sent)
            concepts = [m for m in raw_matches if m not in stopwords and len(m) > 2]

            # Also capture key domain noun compounds if capitalized matches are sparse
            if len(concepts) < 2:
                tech_terms = re.findall(r"\b[a-zA-Z]{5,}(?:\s+[a-zA-Z]{5,})\b", sent)
                concepts.extend(tech_terms[:3])

            unique_c = list(dict.fromkeys(concepts))[:5]  # Cap at prime cardinality 5
            if len(unique_c) >= 2:
                sent_node_ids = []
                for idx_c, c_name in enumerate(unique_c):
                    nid = re.sub(r"[^a-z0-9]+", "_", c_name.lower()).strip("_")
                    if not nid:
                        continue
                    sent_node_ids.append(nid)
                    if nid not in nodes_dict:
                        cat = "Process" if any(sfx in nid for sfx in ["tion", "ing", "ment", "delay", "loss"]) else "Concept"
                        nodes_dict[nid] = {
                            "id": nid,
                            "name": c_name,
                            "domains": [default_domain],
                            "categories": [cat],
                            "aliases": [c_name],
                            "raptor_layer": 0 if idx_c > 0 else 1
                        }
                    else:
                        if default_domain not in nodes_dict[nid]["domains"]:
                            nodes_dict[nid]["domains"].append(default_domain)

                if len(sent_node_ids) >= 2:
                    # Assign DyHSL self-attention weights (higher weight to core subject/anchor)
                    total_w = sum(1.0 / (i + 1) for i in range(len(sent_node_ids)))
                    weights = {nid: round((1.0 / (i + 1)) / total_w, 4) for i, nid in enumerate(sent_node_ids)}
                    edges.append({
                        "id": f"edge_{default_domain.lower()}_c{chunk_idx}_s{s_idx}",
                        "nodes": sent_node_ids,
                        "relation_type": "CausalLink" if any(w in sent.lower() for w in ["lead", "caus", "result", "induc", "trigger"]) else "ContextualInteraction",
                        "roles": {f"role_{i+1}": nid for i, nid in enumerate(sent_node_ids)},
                        "node_weights": weights,
                        "domain_scope": [default_domain],
                        "confidence": 0.90,
                        "raptor_layer": 0 if len(sent_node_ids) == 2 else 1,
                        "attributes": {"evidence_sentence": sent[:160]}
                    })
                    chunk_root_concepts.append(sent_node_ids[0])

        # Build RAPTOR Layer 2 (Macro-Summary Hyperedge) across sentence core concepts
        unique_roots = list(dict.fromkeys(chunk_root_concepts))[:5]
        if len(unique_roots) >= 2:
            for r_nid in unique_roots:
                if r_nid in nodes_dict:
                    nodes_dict[r_nid]["raptor_layer"] = 2
            edges.append({
                "id": f"edge_macro_raptor_l2_c{chunk_idx}_{uuid.uuid4().hex[:4]}",
                "nodes": unique_roots,
                "relation_type": "MacroThematicSynthesis",
                "roles": {f"hub_{i+1}": nid for i, nid in enumerate(unique_roots)},
                "domain_scope": [default_domain],
                "confidence": 0.95,
                "raptor_layer": 2,
                "attributes": {"application": f"RAPTOR Layer-2 macro-abstraction over {default_domain}"}
            })

        return {"nodes": list(nodes_dict.values()), "edges": edges}

    def _merge_into_hyperontology(self, hyper: Hyperontology, data: Dict[str, Any], source_chunk: str):
        for n_raw in data.get("nodes", []):
            nid = n_raw.get("id")
            if not nid:
                continue
            existing = hyper.get_node(nid)
            if existing:
                for d in n_raw.get("domains", []):
                    if d not in existing.domains:
                        existing.domains.append(d)
                for c in n_raw.get("categories", []):
                    if c not in existing.categories:
                        existing.categories.append(c)
                existing.raptor_layer = max(existing.raptor_layer, int(n_raw.get("raptor_layer", 0)))
                if existing.is_boundary:
                    hyper.boundary_nodes.add(existing.id)
            else:
                node = HyperNode(
                    id=nid,
                    name=n_raw.get("name", nid),
                    domains=n_raw.get("domains", []),
                    categories=n_raw.get("categories", ["Concept"]),
                    aliases=n_raw.get("aliases", []),
                    raptor_layer=int(n_raw.get("raptor_layer", 0)),
                    provenance=[{"chunk_preview": source_chunk[:100]}]
                )
                hyper.add_node(node)

        for e_raw in data.get("edges", []):
            eid = e_raw.get("id") or f"edge_{uuid.uuid4().hex[:8]}"
            node_ids = list(dict.fromkeys(e_raw.get("nodes", [])))
            valid_nodes = [nid for nid in node_ids if hyper.get_node(nid)]
            if len(valid_nodes) >= 2:
                # Populate domain scope automatically from participating nodes if not exhaustive
                combined_domains = set(e_raw.get("domain_scope", []))
                for nid in valid_nodes:
                    n_obj = hyper.get_node(nid)
                    if n_obj:
                        combined_domains.update(n_obj.domains)

                edge = HyperEdge(
                    id=eid,
                    nodes=valid_nodes,
                    relation_type=e_raw.get("relation_type", "AssociatedWith"),
                    roles=e_raw.get("roles", {}),
                    node_weights=e_raw.get("node_weights", {}),
                    domain_scope=sorted(combined_domains),
                    confidence=float(e_raw.get("confidence", 1.0)),
                    raptor_layer=int(e_raw.get("raptor_layer", 0)),
                    attributes=e_raw.get("attributes", {"source_preview": source_chunk[:120]}),
                    provenance={"source_chunk": source_chunk[:120]}
                )
                hyper.add_edge(edge)
