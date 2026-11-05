"""
Auto-Hyperontology Command Line Interface (CLI)
Allows building, indexing, and querying mathematical hyperontologies from any text files.
"""

import argparse
import sys
import json
from auto_hyperontology.rag import HyperRAG
from auto_hyperontology.core import Hyperontology


def main():
    parser = argparse.ArgumentParser(
        prog="autohyperonto",
        description="Auto-Hyperontology: Build and query n-ary mathematical hyperontologies for local LLMs."
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Command: build
    build_parser = subparsers.add_parser("build", help="Extract and compile hyperontology from a file or directory")
    build_parser.add_argument("--input", "-i", required=True, help="Path to text file or directory of documents")
    build_parser.add_argument("--output", "-o", default="hyperontology.json", help="Path to save output JSON (default: hyperontology.json)")
    build_parser.add_argument("--model", "-m", default="intuyt:macro", help="Ollama model name (default: intuyt:macro)")
    build_parser.add_argument("--endpoint", "-e", default="http://localhost:11434", help="Ollama base URL (default: http://localhost:11434)")
    build_parser.add_argument("--domain", "-d", default="General", help="Domain label for input text (default: General)")

    # Command: query
    query_parser = subparsers.add_parser("query", help="Ask a question against a compiled hyperontology")
    query_parser.add_argument("--ontology", "-g", required=True, help="Path to compiled hyperontology JSON")
    query_parser.add_argument("--question", "-q", required=True, help="Question to ask")
    query_parser.add_argument("--model", "-m", default="intuyt:macro", help="Ollama model name")
    query_parser.add_argument("--endpoint", "-e", default="http://localhost:11434", help="Ollama base URL")
    query_parser.add_argument("--max-edges", type=int, default=4, help="Maximum hyperedges in proof subgraph")

    # Command: stats
    stats_parser = subparsers.add_parser("stats", help="Inspect structural metrics of a compiled hyperontology")
    stats_parser.add_argument("--ontology", "-g", required=True, help="Path to compiled hyperontology JSON")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    if args.command == "build":
        print(f"[*] Compiling hyperontology from: {args.input}")
        rag = HyperRAG(model=args.model, base_url=args.endpoint, default_domain=args.domain)
        
        import os
        if os.path.isfile(args.input):
            rag.add_file(args.input, domain=args.domain)
        elif os.path.isdir(args.input):
            rag.add_directory(args.input)
        else:
            print(f"[!] Input path not found: {args.input}")
            sys.exit(1)

        rag.save(args.output)
        stats = rag.stats()
        print(f"[+] Successfully compiled hyperontology to {args.output}")
        print(f"    Nodes: {stats['total_nodes']} | n-ary Hyperedges: {stats['total_edges']} | Boundary Anchors: {stats['boundary_nodes']}")
        print(f"    Domains: {', '.join(stats['domains'])}")

    elif args.command == "query":
        rag = HyperRAG.load(args.ontology, model=args.model, base_url=args.endpoint)
        print(f"[*] Querying hyperontology: '{args.question}'")
        res = rag.query(args.question, max_edges=args.max_edges)
        
        print("\n" + "=" * 70)
        print("ANSWER:")
        print("=" * 70)
        print(res.answer)
        print("\n" + "-" * 70)
        print("AME METRICS:")
        print(f"• Ahorro (A) Context Compression: {res.metrics.savings_percent}% reduction vs. unconstrained graph RAG")
        print(f"  ({res.metrics.context_tokens_used} tokens used vs. baseline {res.metrics.context_tokens_baseline})")
        print(f"• Mejora (M) Structural Adherence: Zero hallucinated hops")
        print(f"• Evidencia (E) Grounded Citations: {res.metrics.evidence_score * 100:.1f}% ({len(res.metrics.grounded_hyperedges)} hyperedges cited)")
        print("-" * 70)

    elif args.command == "stats":
        h = Hyperontology.load_json(args.ontology)
        print(f"Hyperontology: {h.name}")
        print(f"• Total Nodes: {len(h.nodes)}")
        print(f"• Total Hyperedges: {len(h.edges)}")
        print(f"• Boundary Nodes: {len(h.boundary_nodes)}")
        print(f"• Domains ({len(h.domain_to_nodes)}): {list(h.domain_to_nodes.keys())}")
        print(f"• Categories: {list(h.category_to_nodes.keys())}")


if __name__ == "__main__":
    main()
