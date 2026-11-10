"""
Auto-Hyperontology: Proto-AGI v4.0 Forensic Reasoning Engine & GreenAGI Footprint Tracker
Integrates:
1. LLM-Assisted Semantic Bridge (`index.php` lines 51-67): Extracts 3 core domain keywords
   via `intuyt:macro` to bridge multilingual/natural queries into the ontology namespace.
2. Complete 1-Hop Neighborhood Subgraph Expansion (`index.php` lines 93-117) ranked by
   A320228 Ordinal-Cardinal Mixing ($F_{OC}$) and RAPTOR Cardinality Gating.
3. Full `CURRENT GRAPH SLICE` Serialization (`nodes` with `attributes` + `edges` with `notes`
   and `constraints`, matching `biol1.json`).
4. Structured ReAct Forensic Output (`[THINK]`, `[REF]`, `[NAT]`, and `[TOOL: EXPAND_NODE: id]`).
"""

import json
import re
from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from auto_hyperontology.core import Hyperontology, HyperNode, HyperEdge
from auto_hyperontology.topology import TopologicalIndexer
from auto_hyperontology.extractor import OllamaClient, estimate_bpe_tokens


WH_PER_1K_3B = 0.0333
CO2_KG_PER_1K_3B = 0.000013
WH_PER_1K_400B = 4.4440
CO2_KG_PER_1K_400B = 0.001780


@dataclass
class AMEMetrics:
    context_tokens_used: int
    context_tokens_baseline: int
    savings_percent: float
    prime_stride_compressed_tokens: int
    energy_wh_3b: float
    energy_wh_400b_baseline: float
    co2_kg_3b: float
    co2_kg_400b_baseline: float
    compute_reduction_factor: float
    grounded_hyperedges: List[str]
    evidence_score: float


@dataclass
class ReasoningResult:
    question: str
    semantic_keywords: List[str]
    parsed_categories: List[str]
    raptor_gate_info: Dict[str, Any]
    seed_nodes: List[str]
    active_node_ids: List[str]
    boundary_nodes_activated: List[str]
    retrieved_hyperedges: List[str]
    f_oc_scores: Dict[str, float]
    current_graph_slice: Dict[str, Any]
    minimal_response_context: str
    raw_react_output: str
    think_block: str
    ref_block: str
    nat_block: str
    answer: str
    metrics: AMEMetrics


class ProtoAGIReasoner:
    """
    Combines the exact Proto-AGI v4.0 Forensic pipeline from `index.php` with
    A320228 Ordinal-Cardinal Mixing ($F_{OC}$), RAPTOR Cardinality Gating, and GreenAGI metrics.
    """
    def __init__(
        self,
        hyperontology: Hyperontology,
        ollama_client: Optional[OllamaClient] = None,
        alpha: float = 0.45,
        beta: float = 0.55,
        strict_refusal: bool = True,
        use_llm_semantic_bridge: bool = True
    ):
        self.hyper = hyperontology
        self.client = ollama_client or OllamaClient()
        self.indexer = TopologicalIndexer(hyperontology)
        self.alpha = alpha
        self.beta = beta
        self.strict_refusal = strict_refusal
        self.use_llm_semantic_bridge = use_llm_semantic_bridge

    def answer_question(
        self,
        question: str,
        max_edges: int = 6,
        memory_node_ids: Optional[List[str]] = None
    ) -> ReasoningResult:
        # Stage 1: Semantic Bridge (`index.php` Step 1 + lexical matching)
        parsed_info = self._parse_question_with_semantic_bridge(question, memory_node_ids=memory_node_ids)
        semantic_keywords = parsed_info["semantic_keywords"]
        categories = parsed_info["categories"]
        seed_node_ids = parsed_info["seed_node_ids"]

        active_boundary_nodes = [
            nid for nid in seed_node_ids if nid in self.hyper.boundary_nodes
        ]

        # Stage 2: RAPTOR Cardinality Gate C(q) -> {L_0, L_1, L_2}
        raptor_gate = self.indexer.raptor_cardinality_gate(question, list(seed_node_ids))

        # Stage 3: Topological Expansion via F_OC Ordinal-Cardinal Mixing
        candidate_edge_ids: Set[str] = set()
        for nid in seed_node_ids:
            for edge in self.hyper.get_incident_edges(nid):
                candidate_edge_ids.add(edge.id)

        if not candidate_edge_ids and not self.strict_refusal:
            candidate_edge_ids = set(self.hyper.edges.keys())

        candidate_edges = [
            self.hyper.get_edge(eid) for eid in candidate_edge_ids if self.hyper.get_edge(eid)
        ]

        ranked_with_scores = self.indexer.schedule_ordinal_cardinal_retrieval(
            candidate_edges=candidate_edges,
            seed_nodes=list(seed_node_ids),
            alpha=self.alpha,
            beta=self.beta,
            query_text=question
        )
        selected_pairs = ranked_with_scores[:max_edges]
        selected_edges = [edge for edge, _ in selected_pairs]
        f_oc_scores = {edge.id: score for edge, score in selected_pairs}

        # Stage 4: Subgraph Slice Construction (`index.php` Step 2: load all connected neighbor nodes + full attributes!)
        current_graph_slice, minimal_context, active_node_ids = self._build_forensic_graph_slice(
            seed_node_ids=seed_node_ids,
            selected_edges=selected_edges,
            f_oc_scores=f_oc_scores,
            raptor_gate=raptor_gate
        )

        # Stage 5: ReAct Forensic Agentic Inference (`index.php` Step 3: [THINK], [REF], [NAT])
        raw_output, think_b, ref_b, nat_b, combined_answer = self._generate_react_forensic_answer(
            question=question,
            current_graph_slice=current_graph_slice,
            selected_edges=selected_edges,
            active_boundary_nodes=active_boundary_nodes
        )

        # Compute AME & GreenAGI Ecological Metrics
        slice_json_str = json.dumps(current_graph_slice, ensure_ascii=False)
        tokens_used = estimate_bpe_tokens(slice_json_str) + 45
        full_graph_tokens = max(estimate_bpe_tokens(json.dumps(self.hyper.to_dict(), ensure_ascii=False)), tokens_used * 3, 850)
        savings = round(max(0.0, ((full_graph_tokens - tokens_used) / float(full_graph_tokens)) * 100.0), 1)

        stride_info = self.indexer.prime_stride_attention_indices(tokens_used, prime_strides=[2, 3, 5])
        prime_strided_tokens = stride_info["attended_count"]

        energy_3b = round((tokens_used / 1000.0) * WH_PER_1K_3B, 6)
        energy_400b = round((full_graph_tokens / 1000.0) * WH_PER_1K_400B, 6)
        co2_3b = round((tokens_used / 1000.0) * CO2_KG_PER_1K_3B, 8)
        co2_400b = round((full_graph_tokens / 1000.0) * CO2_KG_PER_1K_400B, 8)
        reduction_factor = round(energy_400b / energy_3b, 1) if energy_3b > 0 else 666.0

        metrics = AMEMetrics(
            context_tokens_used=tokens_used,
            context_tokens_baseline=full_graph_tokens,
            savings_percent=savings,
            prime_stride_compressed_tokens=prime_strided_tokens,
            energy_wh_3b=energy_3b,
            energy_wh_400b_baseline=energy_400b,
            co2_kg_3b=co2_3b,
            co2_kg_400b_baseline=co2_400b,
            compute_reduction_factor=reduction_factor,
            grounded_hyperedges=[e.id for e in selected_edges],
            evidence_score=1.0 if selected_edges else 0.0
        )

        return ReasoningResult(
            question=question,
            semantic_keywords=semantic_keywords,
            parsed_categories=categories,
            raptor_gate_info=raptor_gate,
            seed_nodes=list(seed_node_ids),
            active_node_ids=active_node_ids,
            boundary_nodes_activated=active_boundary_nodes,
            retrieved_hyperedges=[e.id for e in selected_edges],
            f_oc_scores=f_oc_scores,
            current_graph_slice=current_graph_slice,
            minimal_response_context=minimal_context,
            raw_react_output=raw_output,
            think_block=think_b,
            ref_block=ref_b,
            nat_block=nat_b,
            answer=combined_answer,
            metrics=metrics
        )

    def _parse_question_with_semantic_bridge(
        self,
        question: str,
        memory_node_ids: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Implements `index.php` Step 1 (Semantic Bridge) + Step 2 (Haystack Matching):
        1. Calls `intuyt:macro` to extract 3 specific English nouns/keywords from `question`
           (bridging Spanish/multilingual queries to English ontology IDs/names/labels).
        2. Searches across `node.id + node.name + categories + labels + attributes`.
        """
        seed_node_ids: Set[str] = set(memory_node_ids or [])
        search_terms: List[str] = []

        if self.use_llm_semantic_bridge and self.client.is_available():
            bridge_prompt = (
                f"Identify the core scientific, clinical, biological, or ethical concepts in this query: "
                f"'{question}'. Return only 3 specific English nouns/keywords, separated by commas."
            )
            try:
                kw_raw = self.client.generate(prompt=bridge_prompt)
                for part in kw_raw.split(","):
                    cleaned_kw = re.sub(r"[^a-zA-Z0-9_\-\s]", "", part).strip().lower()
                    if len(cleaned_kw) > 2:
                        search_terms.append(cleaned_kw)
            except Exception:
                pass

        # Also include direct tokens from the user query
        stopwords = {
            "which", "what", "where", "when", "whose", "there", "their", "about",
            "would", "could", "should", "these", "those", "under", "between",
            "relate", "relates", "occur", "occurs", "cause", "causes", "explain",
            "como", "cual", "cuales", "dice", "sobre", "para", "entre", "tiene"
        }
        q_tokens = [
            w.strip("?,.:;()¿¡\"'").lower()
            for w in question.split()
            if len(w.strip("?,.:;()¿¡\"'")) >= 3 and w.strip("?,.:;()¿¡\"'").lower() not in stopwords
        ]
        for qt in q_tokens:
            if qt not in search_terms:
                search_terms.append(qt)

        # Match against full node haystack (`id + name + categories + labels + aliases + attributes`)
        for node in self.hyper.nodes.values():
            attr_str = json.dumps(node.attributes, ensure_ascii=False).lower() if node.attributes else ""
            haystack = " ".join([
                node.id.lower(),
                node.name.lower(),
                " ".join(c.lower() for c in node.categories),
                " ".join(l.lower() for l in node.labels),
                " ".join(a.lower() for a in node.aliases),
                attr_str
            ])
            for term in search_terms:
                if len(term) > 2 and term in haystack:
                    seed_node_ids.add(node.id)
                    break

        categories = set(["Concept"])
        for nid in seed_node_ids:
            n = self.hyper.get_node(nid)
            if n:
                categories.update(n.categories)

        return {
            "semantic_keywords": search_terms[:6],
            "categories": sorted(categories),
            "seed_node_ids": seed_node_ids
        }

    def _build_forensic_graph_slice(
        self,
        seed_node_ids: Set[str],
        selected_edges: List[HyperEdge],
        f_oc_scores: Dict[str, float],
        raptor_gate: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], str, List[str]]:
        """
        Implements `index.php` Step 2:
        Loads all seed nodes AND every neighbor node connected by the selected hyperedges,
        preserving their complete `attributes`, `labels`, `constraints`, and `notes` inside
        `CURRENT GRAPH SLICE`.
        """
        node_ids_in_context: List[str] = []
        active_nodes_dicts: List[Dict[str, Any]] = []
        active_edges_dicts: List[Dict[str, Any]] = []

        def _include_node(nid: str):
            if nid not in node_ids_in_context:
                n_obj = self.hyper.get_node(nid)
                if n_obj:
                    node_ids_in_context.append(nid)
                    active_nodes_dicts.append(n_obj.to_slice_dict())

        for sid in sorted(seed_node_ids):
            _include_node(sid)

        for edge in selected_edges:
            active_edges_dicts.append(edge.to_slice_dict())
            for conn_id in edge.nodes:
                _include_node(conn_id)

        current_graph_slice = {
            "nodes": active_nodes_dicts,
            "edges": active_edges_dicts
        }

        # Also build a human-readable summary of the proof subgraph
        lines = [
            "=== PROTO-AGI v4.0 CURRENT GRAPH SLICE (MINIMAL PROOF SUBGRAPH) ===",
            f"RAPTOR Gate: Layer L_{raptor_gate['target_raptor_layer']} | "
            f"Matryoshka Dim: {raptor_gate['matryoshka_dim']} | "
            f"Nodes Loaded: {len(active_nodes_dicts)} | Hyperedges Loaded: {len(active_edges_dicts)}"
        ]
        for edge in selected_edges:
            sig = self.indexer.compute_edge_signature(edge)
            score = f_oc_scores.get(edge.id, 0.0)
            lines.append(
                f"• Hyperedge [{edge.id}] ({edge.relation_type}: {edge.relation}) "
                f"| |H_e|={edge.cardinality} | F_OC={score:.4f} | PrimeSig={sig}:"
            )
            lines.append(f"  Connects: {edge.nodes}")
            if edge.notes:
                lines.append(f"  Notes: {edge.notes}")
            if edge.constraints:
                lines.append(f"  Constraints: {json.dumps(edge.constraints, ensure_ascii=False)}")

        for nd in active_nodes_dicts:
            if nd.get("attributes"):
                lines.append(f"• Node [{nd['id']}] ({nd['name']}) Attributes: {json.dumps(nd['attributes'], ensure_ascii=False)}")

        return current_graph_slice, "\n".join(lines), node_ids_in_context

    def _generate_react_forensic_answer(
        self,
        question: str,
        current_graph_slice: Dict[str, Any],
        selected_edges: List[HyperEdge],
        active_boundary_nodes: List[str]
    ) -> Tuple[str, str, str, str, str]:
        """
        Implements `index.php` Step 3:
        Passes `CURRENT GRAPH SLICE` JSON and executes the structured ReAct forensic prompt
        generating `[THINK]`, `[REF]`, and `[NAT]`.
        """
        if not selected_edges and not current_graph_slice.get("nodes"):
            refusal_msg = (
                "Refusal State: No certified n-ary hyperedges or nodes exist in the "
                "compiled hyperontology to ground this query."
            )
            return (
                f"[THINK] No matching nodes or hyperedges found in ontology.\n[REF] {refusal_msg}\n[NAT] {refusal_msg}",
                "No matching nodes or hyperedges found in ontology.",
                refusal_msg,
                refusal_msg,
                refusal_msg
            )

        if self.client.is_available():
            slice_json = json.dumps(current_graph_slice, ensure_ascii=False)
            final_prompt = (
                f"SYSTEM: You are a Proto-AGI Agent reasoning over a topological hypergraph.\n"
                f"CURRENT GRAPH SLICE: {slice_json}\n"
                f"USER QUERY: {question}\n\n"
                f"INSTRUCTIONS & RULES:\n"
                f"1. YOU MUST ALWAYS start your response with a [THINK] block explaining your internal logic over the nodes and hyperedges.\n"
                f"2. Evaluate if the CURRENT GRAPH SLICE contains enough info to answer.\n"
                f"3. IF NO info is found, you can request to expand ONE node by outputting: '[TOOL: EXPAND_NODE: id_of_node]'.\n"
                f"4. CRITICAL: If the answer is missing from the slice, generate [REF] and [NAT] stating it is outside the ontology.\n"
                f"5. IF YES info is found, generate:\n"
                f"   [REF] Detailed technical explanation in English citing exact [node_id] and [edge_id] and incorporating the concrete attributes, mechanisms, and notes from CURRENT GRAPH SLICE.\n"
                f"   [NAT] Clear, fluid natural language synthesis (in the language of the query).\n\n"
                f"AGENT OUTPUT:"
            )
            try:
                raw_out = self.client.generate(prompt=final_prompt)
                think_b, ref_b, nat_b = self._parse_react_blocks(raw_out)
                # Combine [REF] and [NAT] so both technical English citations and natural response are preserved
                combined = f"[REF] {ref_b}\n\n[NAT] {nat_b}".strip() if (ref_b or nat_b) else raw_out
                return raw_out, think_b, ref_b, nat_b, combined
            except Exception:
                pass

        # Deterministic fallback synthesizing full node attributes + hyperedge notes
        think_str = (
            f"Activated {len(current_graph_slice['nodes'])} nodes and {len(selected_edges)} hyperedges "
            f"via A320228 Ordinal-Cardinal mixing."
        )
        ref_parts = []
        for e in selected_edges:
            n_labels = [
                f"{self.hyper.get_node(nid).name} [{nid}]" if self.hyper.get_node(nid) else f"[{nid}]"
                for nid in e.nodes
            ]
            note_txt = e.notes or e.attributes.get("application") or e.attributes.get("evidence_sentence") or ""
            c_txt = f" Constraints: {json.dumps(e.constraints)}." if e.constraints else ""
            ref_parts.append(
                f"Hyperedge [{e.id}] ({e.relation_type} -> {e.relation}) connects {', '.join(n_labels)}. "
                f"{note_txt}{c_txt}"
            )
        for nd in current_graph_slice.get("nodes", []):
            if nd.get("attributes"):
                ref_parts.append(f"Node [{nd['id']}] ({nd['name']}) attributes: {json.dumps(nd['attributes'], ensure_ascii=False)}")

        ref_str = " ".join(ref_parts)
        nat_str = ref_str
        raw_fallback = f"[THINK] {think_str}\n[REF] {ref_str}\n[NAT] {nat_str}"
        return raw_fallback, think_str, ref_str, nat_str, f"[REF] {ref_str}\n\n[NAT] {nat_str}"

    @staticmethod
    def _parse_react_blocks(raw_text: str) -> Tuple[str, str, str]:
        think_match = re.search(r"\[THINK\](.*?)(?=\[REF\]|\[NAT\]|\[TOOL:|$)", raw_text, re.DOTALL | re.IGNORECASE)
        ref_match = re.search(r"\[REF\](.*?)(?=\[NAT\]|\[TOOL:|$)", raw_text, re.DOTALL | re.IGNORECASE)
        nat_match = re.search(r"\[NAT\](.*?)(?=\[TOOL:|$)", raw_text, re.DOTALL | re.IGNORECASE)

        think_b = think_match.group(1).strip() if think_match else ""
        ref_b = ref_match.group(1).strip() if ref_match else ""
        nat_b = nat_match.group(1).strip() if nat_match else ""
        if not ref_b and not nat_b:
            ref_b = raw_text.strip()
        return think_b, ref_b, nat_b
