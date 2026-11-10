"""
Auto-Hyperontology Command Line Interface (CLI)
Supports building, querying, inspecting stats, exporting WebGL topological reifications,
and running the Live Hallucination & Efficiency Benchmark against RealFlatTextRAG.
"""

import argparse
import sys
import os
import json
from auto_hyperontology.rag import HyperRAG
from auto_hyperontology.core import Hyperontology
from auto_hyperontology.benchmark import run_live_rag_benchmark


def main():
    parser = argparse.ArgumentParser(
        prog="autohyperonto",
        description="Auto-Hyperontology: Build, index, distill, benchmark, and query n-ary mathematical hyperontologies."
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Command: build
    build_parser = subparsers.add_parser("build", help="Compile hyperontology from a text file or directory")
    build_parser.add_argument("--input", "-i", required=True, help="Input file or directory path")
    build_parser.add_argument("--output", "-o", default="hyperontology.json", help="Output JSON path")
    build_parser.add_argument("--webgl", "-w", default=None, help="Optional path to export WebGL reified JSON (reader-math.php)")
    build_parser.add_argument("--model", "-m", default="intuyt:macro", help="Ollama model name (default: intuyt:macro)")
    build_parser.add_argument("--endpoint", "-e", default="http://localhost:11434", help="Ollama base URL")
    build_parser.add_argument("--domain", "-d", default="General", help="Default domain name")
    build_parser.add_argument("--theta-max", type=int, default=3000, help="Max BPE tokens per chunk (default: 3000)")

    # Command: query
    query_parser = subparsers.add_parser("query", help="Execute 5-stage Proto-AGI query over compiled hyperontology")
    query_parser.add_argument("--ontology", "-g", required=True, help="Path to compiled hyperontology JSON")
    query_parser.add_argument("--question", "-q", required=True, help="Natural language question")
    query_parser.add_argument("--model", "-m", default="intuyt:macro", help="Ollama model name")
    query_parser.add_argument("--endpoint", "-e", default="http://localhost:11434", help="Ollama base URL")
    query_parser.add_argument("--max-edges", type=int, default=4, help="Max hyperedges in minimal response tree")
    query_parser.add_argument("--alpha", type=float, default=0.45, help="F_OC alpha weight for Sim(Q, H_e)")
    query_parser.add_argument("--beta", type=float, default=0.55, help="F_OC beta weight for PrimeWeight(|H_e|)/Pos(H_e)")

    # Command: stats
    stats_parser = subparsers.add_parser("stats", help="Inspect topological, coprime, and OEIS A320228 metrics")
    stats_parser.add_argument("--ontology", "-g", required=True, help="Path to compiled hyperontology JSON")

    # Command: export-webgl
    webgl_parser = subparsers.add_parser("export-webgl", help="Export bounded WebGL reified subgraph (N_limit <= 300)")
    webgl_parser.add_argument("--ontology", "-g", required=True, help="Path to compiled hyperontology JSON")
    webgl_parser.add_argument("--output", "-o", default="hiperontologia_final.json", help="Output WebGL JSON")
    webgl_parser.add_argument("--hub", default=None, help="Optional central hub node ID")
    webgl_parser.add_argument("--n-limit", type=int, default=300, help="Rendering node limit (10 <= N_limit <= 300)")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    if args.command == "build":
        print(f"[*] Compiling mathematical hyperontology from: {args.input} (theta_max={args.theta_max})")
        rag = HyperRAG(
            model=args.model,
            base_url=args.endpoint,
            default_domain=args.domain,
            theta_max=args.theta_max
        )
        if os.path.isfile(args.input):
            rag.add_file(args.input, domain=args.domain)
        elif os.path.isdir(args.input):
            rag.add_directory(args.input)
        else:
            print(f"[!] Input path not found: {args.input}")
            sys.exit(1)

        rag.save(args.output)
        st = rag.stats()
        print(f"[+] Saved compiled hyperontology to: {args.output}")
        print(f"    Vertices |X|: {st['total_nodes']} | Hyperedges |E|: {st['total_edges']} | Boundary Nodes: {st['boundary_nodes']}")
        print(f"    Coprime Adjacency Verified: {st['coprime_labeling_verified']}")
        print(f"    Distillation Curriculum: {st['curriculum_counts']}")

        if args.webgl:
            payload = rag.export_webgl_reification(args.webgl)
            print(f"[+] Exported WebGL reified graph ({payload['total_rendered']} elements) to: {args.webgl}")

    elif args.command == "query":
        rag = HyperRAG.load(args.ontology, model=args.model, base_url=args.endpoint)
        rag.alpha = args.alpha
        rag.beta = args.beta
        rag._refresh_engine()

        res = rag.query(args.question, max_edges=args.max_edges)
        print("=" * 75)
        print(f"QUERY: {res.question}")
        print("=" * 75)
        print(f"• RAPTOR Gate: Layer L_{res.raptor_gate_info['target_raptor_layer']} ({res.raptor_gate_info['complexity_tier']})")
        print(f"• Matryoshka Embedding Unfolding: {res.raptor_gate_info['matryoshka_dim']} dimensions")
        print(f"• Seed Nodes: {res.seed_nodes} | Active Boundary Anchors: {res.boundary_nodes_activated}")
        print(f"• F_OC Scheduled Hyperedges: {res.f_oc_scores}")
        print("\n" + res.minimal_response_context)
        print("\n" + "-" * 75)
        print("GROUNDED ANSWER:")
        print("-" * 75)
        print(res.answer)
        print("\n" + "=" * 75)
        print("AME & GREEN-AGI METRICS:")
        print("=" * 75)
        print(f"• Ahorro (A) Context Savings: {res.metrics.savings_percent}% ({res.metrics.context_tokens_used} tok vs {res.metrics.context_tokens_baseline} baseline)")
        print(f"• Prime Stride Sparse Attention (strides 2,3,5): {res.metrics.prime_stride_compressed_tokens} active tokens attended")
        print(f"• GreenAGI Energy (3B MoE vs 400B Dense): {res.metrics.energy_wh_3b} Wh vs {res.metrics.energy_wh_400b_baseline} Wh (~{res.metrics.compute_reduction_factor}x reduction)")
        print(f"• GreenAGI Carbon Footprint: {res.metrics.co2_kg_3b} kg CO2 vs {res.metrics.co2_kg_400b_baseline} kg CO2")
        print(f"• Evidencia (E) Traceability: {res.metrics.evidence_score * 100:.1f}% ({len(res.metrics.grounded_hyperedges)} hyperedges cited)")
        print("=" * 75)

    elif args.command == "stats":
        rag = HyperRAG.load(args.ontology)
        print(json.dumps(rag.stats(), indent=2))

    elif args.command == "export-webgl":
        rag = HyperRAG.load(args.ontology)
        payload = rag.export_webgl_reification(args.output, hub_node_id=args.hub, n_limit=args.n_limit)
        print(f"[+] Exported WebGL topological reification (hub={payload['hub']}, rendered={payload['total_rendered']}) to {args.output}")


if __name__ == "__main__":
    main()
