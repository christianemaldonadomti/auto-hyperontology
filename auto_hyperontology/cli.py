"""
Auto-Hyperontology Command Line Interface (CLI)
"""

import argparse
import sys
import os
import json
from auto_hyperontology.rag import HyperRAG
from auto_hyperontology.core import Hyperontology


def main():
    parser = argparse.ArgumentParser(
        prog="autohyperonto",
        description="Auto-Hyperontology: Build, index, distill, benchmark, and query n-ary mathematical hyperontologies."
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    build_parser = subparsers.add_parser("build", help="Compile hyperontology from a text file or directory")
    build_parser.add_argument("--input", "-i", required=True, help="Input file or directory path")
    build_parser.add_argument("--output", "-o", default="hyperontology.json", help="Output JSON path")
    build_parser.add_argument("--webgl", "-w", default=None, help="Optional path to export WebGL reified JSON")
    build_parser.add_argument("--model", "-m", default="qwen3.8-flash-next:125b-a6b-q4_K_M", help="Ollama extraction model")
    build_parser.add_argument("--endpoint", "-e", default="https://comic-hugely-eft.ngrok-free.app", help="Ollama base URL")
    build_parser.add_argument("--domain", "-d", default="General", help="Default domain name")
    build_parser.add_argument("--theta-max", type=int, default=3000, help="Max BPE tokens per chunk")

    query_parser = subparsers.add_parser("query", help="Execute Proto-AGI v4.0 query over compiled hyperontology")
    query_parser.add_argument("--ontology", "-g", required=True, help="Path to compiled hyperontology JSON (or biol1.json)")
    query_parser.add_argument("--question", "-q", required=True, help="Natural language question")
    query_parser.add_argument("--model", "-m", default="intuyt:macro", help="Ollama reasoning model")
    query_parser.add_argument("--endpoint", "-e", default="https://comic-hugely-eft.ngrok-free.app", help="Ollama base URL")
    query_parser.add_argument("--max-edges", type=int, default=5, help="Max hyperedges in slice")

    stats_parser = subparsers.add_parser("stats", help="Inspect topological, coprime, and OEIS A320228 metrics")
    stats_parser.add_argument("--ontology", "-g", required=True, help="Path to compiled hyperontology JSON")

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(1)

    if args.command == "build":
        rag = HyperRAG(model=args.model, base_url=args.endpoint, default_domain=args.domain, theta_max=args.theta_max)
        if os.path.isfile(args.input):
            rag.add_file(args.input, domain=args.domain)
        elif os.path.isdir(args.input):
            rag.add_directory(args.input)
        rag.save(args.output)
        print(f"[+] Saved compiled hyperontology to: {args.output}")
        if args.webgl:
            rag.export_webgl_reification(args.webgl)

    elif args.command == "query":
        rag = HyperRAG.load(args.ontology, model=args.model, base_url=args.endpoint)
        res = rag.query(args.question, max_edges=args.max_edges)
        print(res.raw_react_output)

    elif args.command == "stats":
        rag = HyperRAG.load(args.ontology)
        print(json.dumps(rag.stats(), indent=2))


if __name__ == "__main__":
    main()
