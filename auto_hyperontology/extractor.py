"""
Auto-Hyperontology: Extraction Engine & Ollama LLM Client
Implements greedy token-bounded partitioning, deterministic JSON sanitization,
and n-ary hyperontology extraction for local LLM pipelines (intuyt:macro / Ollama).
"""

import json
import re
import requests
import uuid
from typing import List, Dict, Any, Optional
from auto_hyperontology.core import HyperNode, HyperEdge, BoundaryNode, Hyperontology


def greedy_token_bounded_chunk(text: str, max_chars: int = 1800, overlap_chars: int = 200) -> List[str]:
    """
    Greedy text partitioner that splits large unstructured domain corpora into
    bounded semantic segments without splitting critical sentence structures.
    """
    paragraphs = text.split("\n\n")
    chunks = []
    current_chunk = []
    current_len = 0

    for para in paragraphs:
        p_len = len(para)
        if current_len + p_len > max_chars and current_chunk:
            combined = "\n\n".join(current_chunk)
            chunks.append(combined)
            if len(combined) > overlap_chars:
                overlap_text = combined[-overlap_chars:]
                current_chunk = [overlap_text, para]
                current_len = len(overlap_text) + p_len
            else:
                current_chunk = [para]
                current_len = p_len
        else:
            current_chunk.append(para)
            current_len += p_len

    if current_chunk:
        chunks.append("\n\n".join(current_chunk))

    return chunks


def sanitize_json_payload(raw_response: str) -> Dict[str, Any]:
    """
    Deterministic JSON sanitizer. Cleans LLM outputs containing markdown backticks,
    trailing commas, or loose conversational wrapper tokens before parsing.
    """
    cleaned = raw_response.strip()

    # Extract JSON object or array from markdown fences if present
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", cleaned)
    if match:
        cleaned = match.group(1).strip()
    else:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1:
            cleaned = cleaned[start:end + 1]

    # Remove trailing commas before closing braces/brackets
    cleaned = re.sub(r",\s*([\]}])", r"\1", cleaned)

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        try:
            fixed = cleaned.replace("'", '"')
            return json.loads(fixed)
        except Exception:
            return {"nodes": [], "edges": []}


class OllamaClient:
    """
    Interface to local Ollama inference daemon.
    Compatible with custom models (e.g., intuyt:macro, llama3.2, qwen2.5).
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
        """Invokes Ollama's /api/generate endpoint."""
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "system": system_prompt or "",
            "stream": False,
            "options": {
                "temperature": self.temperature
            }
        }
        res = requests.post(url, json=payload, timeout=self.timeout)
        res.raise_for_status()
        return res.json().get("response", "")


EXTRACTION_SYSTEM_PROMPT = """You are an expert formal ontology compiler specialized in mathematical hyperontologies.
Your task is to extract concepts and polyadic relationships from the provided text into a formal n-ary hyperontology.

RULES:
1. Extract NODES:
   - id: unique snake_case string
   - name: formal canonical name
   - domains: list of domains (e.g. Medicine, Law, Economy, Engineering, Physics, ComputerScience, General)
   - categories: list containing foundational classes (Concept, Property, Process, Evidence, Level)
   - aliases: list of alternative terms

2. Extract n-ary HYPEREDGES (relationships connecting 2, 3, or more nodes simultaneously):
   - id: unique edge_snake_case
   - nodes: array of node IDs involved in the relationship (cardinality >= 2)
   - relation_type: CausalLink, PartOf, Subsumption, CrossDomainBridge, EvidenceSupport, InteroperabilityMorphism, PrerequisiteLattice
   - roles: map of node IDs to semantic roles (e.g. {"agent": "id1", "patient": "id2", "condition": "id3", "outcome": "id4"})
   - domain_scope: list of participating domains
   - confidence: float between 0.0 and 1.0

Return strictly valid JSON matching this schema:
{
  "nodes": [
    {"id": "...", "name": "...", "domains": ["..."], "categories": ["..."], "aliases": []}
  ],
  "edges": [
    {"id": "...", "nodes": ["...", "..."], "relation_type": "...", "roles": {}, "domain_scope": ["..."], "confidence": 0.95}
  ]
}
Do not output any introductory or conversational text outside the JSON."""


class AutoHyperontologyExtractor:
    """
    Automated extraction pipeline for compiling arbitrary text corpora into an n-ary Hyperontology.
    """
    def __init__(self, ollama_client: Optional[OllamaClient] = None):
        self.client = ollama_client or OllamaClient()

    def extract_from_text(self, text: str, default_domain: str = "General") -> Hyperontology:
        """
        Chunks the input text and iteratively compiles extracted nodes and hyperedges
        into a unified Hyperontology instance.
        """
        chunks = greedy_token_bounded_chunk(text)
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

        # Offline / Deterministic Heuristic Extractor when LLM is offline
        return self._heuristic_domain_extraction(chunk, default_domain, chunk_idx)

    def _heuristic_domain_extraction(self, chunk: str, default_domain: str, chunk_idx: int) -> Dict[str, Any]:
        """
        Fallback heuristic entity and polyadic relationship extractor.
        Identifies capitalized phrases, technical concepts, and groups co-occurring terms into hyperedges.
        """
        nodes = []
        edges = []

        # Find significant noun phrases or capitalized concepts
        words = re.findall(r"\b[A-Z][a-zA-Z0-9_\-]{2,}\b", chunk)
        unique_concepts = list(dict.fromkeys(words))[:6]

        if len(unique_concepts) >= 2:
            node_ids = []
            for concept in unique_concepts:
                nid = concept.lower().replace("-", "_")
                node_ids.append(nid)
                nodes.append({
                    "id": nid,
                    "name": concept,
                    "domains": [default_domain],
                    "categories": ["Concept"],
                    "aliases": [concept]
                })

            # Create an n-ary hyperedge over co-occurring concepts
            edge_id = f"edge_{default_domain.lower()}_{chunk_idx}_{uuid.uuid4().hex[:4]}"
            edges.append({
                "id": edge_id,
                "nodes": node_ids,
                "relation_type": "ContextualCoOccurrence",
                "roles": {f"participant_{i+1}": nid for i, nid in enumerate(node_ids)},
                "domain_scope": [default_domain],
                "confidence": 0.85
            })

        return {"nodes": nodes, "edges": edges}

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
                if existing.is_boundary:
                    hyper.boundary_nodes.add(existing.id)
            else:
                node = HyperNode(
                    id=nid,
                    name=n_raw.get("name", nid),
                    domains=n_raw.get("domains", []),
                    categories=n_raw.get("categories", ["Concept"]),
                    aliases=n_raw.get("aliases", []),
                    attributes={"preview": source_chunk[:100]}
                )
                hyper.add_node(node)

        for e_raw in data.get("edges", []):
            eid = e_raw.get("id") or f"edge_{uuid.uuid4().hex[:8]}"
            node_ids = e_raw.get("nodes", [])
            valid_nodes = [nid for nid in node_ids if hyper.get_node(nid)]
            if len(valid_nodes) >= 2:
                edge = HyperEdge(
                    id=eid,
                    nodes=valid_nodes,
                    relation_type=e_raw.get("relation_type", "AssociatedWith"),
                    roles=e_raw.get("roles", {}),
                    domain_scope=e_raw.get("domain_scope", []),
                    confidence=float(e_raw.get("confidence", 1.0)),
                    attributes={"source_preview": source_chunk[:120]}
                )
                hyper.add_edge(edge)
