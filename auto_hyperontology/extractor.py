"""
Auto-Hyperontology: Semantic Extraction Pipeline & Ollama LLM Client
Upgraded to extract the rich Proto-AGI v4.0 (`biol1.json`) hyperontology schema:
- Nodes with `id`, `name`, `categories`, `labels`, and rich `attributes` (definitional_note,
  mechanisms, examples, conditions, sources, quantitative findings).
- Hyperedges with `id`, `type`, `relation`, `connects`, `constraints`, and `notes`.
"""

import json
import re
import requests
import uuid
from typing import List, Dict, Any, Optional
from auto_hyperontology.core import HyperNode, HyperEdge, BoundaryNode, Hyperontology, normalize_list_str


def estimate_bpe_tokens(text: str) -> int:
    if not text:
        return 0
    words = re.findall(r"\w+|[^\w\s]", text, re.UNICODE)
    return max(1, int(len(words) * 1.15))


def greedy_token_bounded_chunk(
    text: str,
    theta_max: int = 3000,
    overlap_sentences: int = 1
) -> List[str]:
    cleaned = re.sub(r"\r\n", "\n", text.strip())
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
    cleaned = raw_response.strip()
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", cleaned)
    if match:
        cleaned = match.group(1).strip()
    else:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1:
            cleaned = cleaned[start:end + 1]

    cleaned = re.sub(r",\s*([\]}])", r"\1", cleaned)

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        try:
            return json.loads(cleaned.replace("'", '"'))
        except Exception:
            return {"nodes": [], "hyperedges": []}


class OllamaClient:
    """
    Local & Remote Tunnel Inference Interface for Ollama (`intuyt:macro` & `qwen3.8-flash-next:125b-a6b-q4_K_M`).
    Includes `ngrok-skip-browser-warning: true` header for seamless Google Colab -> ngrok -> Ollama execution.
    """
    def __init__(
        self,
        base_url: str = "https://comic-hugely-eft.ngrok-free.app",
        model: str = "intuyt:macro",
        temperature: float = 0.1,
        timeout: int = 180
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.timeout = timeout
        self.headers = {
            "Content-Type": "application/json",
            "ngrok-skip-browser-warning": "true",
            "User-Agent": "AutoHyperontology-ProtoAGI/4.0"
        }

    def is_available(self) -> bool:
        try:
            res = requests.get(f"{self.base_url}/api/tags", headers=self.headers, timeout=5)
            return res.status_code == 200
        except Exception:
            return False

    def list_models(self) -> List[str]:
        try:
            res = requests.get(f"{self.base_url}/api/tags", headers=self.headers, timeout=5)
            if res.status_code == 200:
                data = res.json()
                return [m.get("name", "") for m in data.get("models", [])]
        except Exception:
            pass
        return []

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        url = f"{self.base_url}/api/generate"
        payload: Dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": self.temperature}
        }
        if system_prompt:
            payload["system"] = system_prompt
        res = requests.post(url, headers=self.headers, json=payload, timeout=self.timeout)
        res.raise_for_status()
        return res.json().get("response", "")


EXTRACTION_SYSTEM_PROMPT = """You are an expert formal hyperontology compiler (Proto-AGI v4.0 schema).
Derive a rich n-ary hypergraph ontology from the provided text.

SCHEMA CONVENTIONS:
1. "nodes": list of objects, each with:
   - "id": snake_case string (e.g., "n_concept", "p_process", "mol_protein")
   - "name": canonical name
   - "categories": 1 or more from ["Concept", "Property", "Process", "Discipline", "Level", "Structure", "Molecule", "Function", "Environment", "EthicalCase", "LegalConcept", "Evidence", "PopulationUnit"]
   - "labels": list of descriptive keywords/tags in English (and Spanish if applicable)
   - "attributes": rich dictionary preserving ALL concrete definitions, mechanisms, examples, quantitative findings, conditions, and outcomes from the text (e.g., {"description": "...", "mechanisms": [...], "findings": "...", "conditions": [...]})

2. "hyperedges": list of n-ary relations connecting 2 or more node IDs simultaneously, each with:
   - "id": snake_case string (e.g., "h_causal_pathway")
   - "type": e.g., "CausalLink", "StudyRelation", "PropertyCoverage", "Supports", "Composition", "LevelHierarchy", "TruthConstraints"
   - "relation": specific predicate verb (e.g., "induces", "regulates", "exhibits_subset_of", "enables")
   - "connects": list of participating node IDs
   - "constraints": dictionary of conditions, exceptions, or missing properties (optional)
   - "notes": detailed summary sentence explaining the exact mechanism, empirical finding, or conclusion linking these nodes.

Return strictly valid JSON: {"nodes": [...], "hyperedges": [...]}"""


class AutoHyperontologyExtractor:
    def __init__(self, ollama_client: Optional[OllamaClient] = None, theta_max: int = 3000):
        self.client = ollama_client or OllamaClient()
        self.theta_max = theta_max

    def extract_from_text(self, text: str, default_domain: str = "General") -> Hyperontology:
        chunks = greedy_token_bounded_chunk(text, theta_max=self.theta_max)
        hyper = Hyperontology(name=f"CompiledOntology_{uuid.uuid4().hex[:6]}")

        for idx, chunk in enumerate(chunks):
            extracted_data = self._extract_chunk(chunk, default_domain, chunk_idx=idx)
            self._merge_into_hyperontology(hyper, extracted_data, source_chunk=chunk, default_domain=default_domain)

        return hyper

    def _extract_chunk(self, chunk: str, default_domain: str, chunk_idx: int) -> Dict[str, Any]:
        prompt = (
            f"Domain: {default_domain}\n\n"
            f"Source Text:\n{chunk}\n\n"
            f"Compile into Proto-AGI v4.0 Hyperontology JSON (preserve all mechanisms, findings, and definitions inside node 'attributes' and hyperedge 'notes'):"
        )
        if self.client.is_available():
            try:
                raw_out = self.client.generate(prompt=prompt, system_prompt=EXTRACTION_SYSTEM_PROMPT)
                data = sanitize_json_payload(raw_out)
                if data.get("nodes") or data.get("hyperedges") or data.get("edges"):
                    return data
            except Exception:
                pass

        return self._heuristic_dyhsl_extraction(chunk, default_domain, chunk_idx)

    def _heuristic_dyhsl_extraction(self, chunk: str, default_domain: str, chunk_idx: int) -> Dict[str, Any]:
        nodes_dict: Dict[str, Dict[str, Any]] = {}
        edges: List[Dict[str, Any]] = []

        stopwords = {
            "The", "This", "That", "These", "Those", "When", "Where", "While",
            "However", "Therefore", "Furthermore", "Although", "Because", "In",
            "On", "At", "To", "For", "With", "From", "By", "As", "If", "It"
        }

        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", chunk) if s.strip()]
        chunk_root_concepts: List[str] = []

        for s_idx, sent in enumerate(sentences):
            raw_matches = re.findall(r"\b[A-Z][a-zA-Z0-9_\-]+(?:\s+[A-Z][a-zA-Z0-9_\-]+)*\b", sent)
            concepts = [m for m in raw_matches if m not in stopwords and len(m) > 2]

            if len(concepts) < 2:
                tech_terms = re.findall(r"\b[a-zA-Z]{5,}(?:\s+[a-zA-Z]{5,})\b", sent)
                concepts.extend(tech_terms[:4])

            unique_c = list(dict.fromkeys(concepts))[:5]
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
                            "labels": [w.lower() for w in c_name.split() if len(w) > 2],
                            "attributes": {"definitional_note": sent},
                            "raptor_layer": 0 if idx_c > 0 else 1
                        }
                    else:
                        if default_domain not in nodes_dict[nid]["domains"]:
                            nodes_dict[nid]["domains"].append(default_domain)
                        existing_note = nodes_dict[nid]["attributes"].get("definitional_note", "")
                        if sent not in existing_note and len(existing_note) < 350:
                            nodes_dict[nid]["attributes"]["definitional_note"] = (existing_note + " " + sent).strip()

                if len(sent_node_ids) >= 2:
                    total_w = sum(1.0 / (i + 1) for i in range(len(sent_node_ids)))
                    weights = {nid: round((1.0 / (i + 1)) / total_w, 4) for i, nid in enumerate(sent_node_ids)}
                    rel_type = "CausalLink" if any(w in sent.lower() for w in ["lead", "caus", "result", "induc", "trigger", "associat"]) else "StudyRelation"
                    edges.append({
                        "id": f"h_{default_domain.lower()}_c{chunk_idx}_s{s_idx}",
                        "type": rel_type,
                        "relation": "relates_in_context",
                        "connects": sent_node_ids,
                        "roles": {f"role_{i+1}": nid for i, nid in enumerate(sent_node_ids)},
                        "node_weights": weights,
                        "domain_scope": [default_domain],
                        "confidence": 0.92,
                        "raptor_layer": 0 if len(sent_node_ids) == 2 else 1,
                        "notes": sent,
                        "attributes": {"evidence_sentence": sent}
                    })
                    chunk_root_concepts.append(sent_node_ids[0])

        unique_roots = list(dict.fromkeys(chunk_root_concepts))[:6]
        if len(unique_roots) >= 2:
            for r_nid in unique_roots:
                if r_nid in nodes_dict:
                    nodes_dict[r_nid]["raptor_layer"] = 2
            edges.append({
                "id": f"h_macro_raptor_l2_c{chunk_idx}_{uuid.uuid4().hex[:4]}",
                "type": "StudyRelation",
                "relation": "synthesizes_findings",
                "connects": unique_roots,
                "domain_scope": [default_domain],
                "confidence": 0.95,
                "raptor_layer": 2,
                "notes": chunk[:400],
                "attributes": {"application": chunk[:400]}
            })

        return {"nodes": list(nodes_dict.values()), "hyperedges": edges}

    def _merge_into_hyperontology(
        self,
        hyper: Hyperontology,
        data: Dict[str, Any],
        source_chunk: str,
        default_domain: str = "General"
    ):
        for n_raw in data.get("nodes", []):
            if not isinstance(n_raw, dict):
                continue
            nid = str(n_raw.get("id", "")).strip()
            if not nid:
                continue
            existing = hyper.get_node(nid)
            raw_attrs = n_raw.get("attributes", {})
            if not isinstance(raw_attrs, dict):
                raw_attrs = {"definitional_note": str(raw_attrs)}
            if not raw_attrs:
                raw_attrs = {"context_excerpt": source_chunk[:240]}

            if existing:
                for d in normalize_list_str(n_raw.get("domains", [default_domain])):
                    if d not in existing.domains:
                        existing.domains.append(d)
                for c in normalize_list_str(n_raw.get("categories", ["Concept"])):
                    if c not in existing.categories:
                        existing.categories.append(c)
                for lbl in normalize_list_str(n_raw.get("labels", [])):
                    if lbl not in existing.labels:
                        existing.labels.append(lbl)
                existing.attributes.update(raw_attrs)
                existing.raptor_layer = max(existing.raptor_layer, int(n_raw.get("raptor_layer", 0)))
                if existing.is_boundary:
                    hyper.boundary_nodes.add(existing.id)
            else:
                labels_list = normalize_list_str(n_raw.get("labels", n_raw.get("aliases", [])))
                node = HyperNode(
                    id=nid,
                    name=str(n_raw.get("name", nid)),
                    domains=normalize_list_str(n_raw.get("domains", [default_domain])),
                    categories=normalize_list_str(n_raw.get("categories", ["Concept"])),
                    labels=labels_list,
                    attributes=raw_attrs,
                    aliases=normalize_list_str(n_raw.get("aliases", labels_list)),
                    raptor_layer=int(n_raw.get("raptor_layer", 0)),
                    provenance=[{"chunk_preview": source_chunk[:120]}]
                )
                hyper.add_node(node)

        raw_edges = data.get("hyperedges") or data.get("edges") or []
        for e_raw in raw_edges:
            if not isinstance(e_raw, dict):
                continue
            eid = str(e_raw.get("id") or f"h_{uuid.uuid4().hex[:8]}").strip()
            raw_connects = e_raw.get("connects") or e_raw.get("nodes") or []
            node_ids = list(dict.fromkeys(normalize_list_str(raw_connects)))

            # Auto-register any missing node referenced by a hyperedge so edges are never dropped!
            for nid in node_ids:
                if not hyper.get_node(nid):
                    hyper.add_node(HyperNode(
                        id=nid,
                        name=nid.replace("_", " ").title(),
                        domains=[default_domain],
                        categories=["Concept"],
                        attributes={"context_excerpt": source_chunk[:200]}
                    ))

            if len(node_ids) >= 1:
                combined_domains = set(normalize_list_str(e_raw.get("domain_scope", [default_domain])))
                for nid in node_ids:
                    n_obj = hyper.get_node(nid)
                    if n_obj:
                        combined_domains.update(n_obj.domains)

                notes_val = str(
                    e_raw.get("notes")
                    or (e_raw.get("attributes", {}) if isinstance(e_raw.get("attributes"), dict) else {}).get("application")
                    or (e_raw.get("attributes", {}) if isinstance(e_raw.get("attributes"), dict) else {}).get("evidence_sentence")
                    or source_chunk[:220]
                )
                constraints_val = e_raw.get("constraints", {})
                if not isinstance(constraints_val, dict):
                    constraints_val = {"rule": str(constraints_val)}

                edge = HyperEdge(
                    id=eid,
                    nodes=node_ids,
                    relation_type=str(e_raw.get("type") or e_raw.get("relation_type") or "StudyRelation"),
                    relation=str(e_raw.get("relation") or "relates"),
                    roles=e_raw.get("roles", {}),
                    node_weights=e_raw.get("node_weights", {}),
                    constraints=constraints_val,
                    notes=notes_val,
                    domain_scope=sorted(combined_domains),
                    confidence=float(e_raw.get("confidence", 1.0)),
                    raptor_layer=int(e_raw.get("raptor_layer", 1)),
                    attributes=e_raw.get("attributes", {"notes": notes_val}),
                    provenance={"source_chunk": source_chunk[:120]}
                )
                hyper.add_edge(edge)
