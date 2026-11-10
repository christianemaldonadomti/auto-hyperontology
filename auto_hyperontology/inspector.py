"""
Auto-Hyperontology: Inspection and Visual Audit Utilities
"""

import json
import os
from typing import Dict, Any, List, Optional
from auto_hyperontology.core import Hyperontology, HyperEdge, HyperNode


def inspect_hyperontology(hyper: Hyperontology) -> Dict[str, Any]:
    nodes = list(hyper.nodes.values())
    edges = list(hyper.edges.values())

    cat_counts: Dict[str, int] = {}
    for n in nodes:
        for c in n.categories:
            cat_counts[c] = cat_counts.get(c, 0) + 1

    domain_counts: Dict[str, int] = {}
    for n in nodes:
        for d in n.domains:
            domain_counts[d] = domain_counts.get(d, 0) + 1

    cardinality_hist: Dict[int, int] = {}
    for e in edges:
        k = e.cardinality
        cardinality_hist[k] = cardinality_hist.get(k, 0) + 1

    boundary_list = [
        {
            "id": n.id,
            "name": n.name,
            "domains": n.domains,
            "categories": n.categories,
            "degree": n.connectivity_degree,
            "prime_label": n.prime_label
        }
        for n in nodes if n.is_boundary
    ]

    edge_table = [
        {
            "edge_id": e.id,
            "relation_type": e.relation_type,
            "relation": e.relation,
            "cardinality": e.cardinality,
            "factorization_pos": e.factorization_pos,
            "raptor_layer": e.raptor_layer,
            "domain_scope": e.domain_scope,
            "participating_nodes": [
                f"{hyper.get_node(nid).name if hyper.get_node(nid) else nid} (p={hyper.get_node(nid).prime_label if hyper.get_node(nid) else '?'})"
                for nid in e.nodes
            ],
            "roles": e.roles,
            "node_weights": e.node_weights,
            "constraints": e.constraints,
            "evidence": e.notes or e.attributes.get("application") or e.attributes.get("evidence_sentence") or ""
        }
        for e in sorted(edges, key=lambda x: (x.factorization_pos, -x.cardinality))
    ]

    return {
        "total_vertices": len(nodes),
        "total_hyperedges": len(edges),
        "boundary_nodes_count": len(boundary_list),
        "boundary_nodes": boundary_list,
        "categories_distribution": cat_counts,
        "domains_distribution": domain_counts,
        "cardinality_spectrum": dict(sorted(cardinality_hist.items())),
        "edges": edge_table
    }


def print_hyperontology_audit(hyper: Hyperontology, max_edges: int = 15):
    audit = inspect_hyperontology(hyper)
    print("=" * 85)
    print(f"HYPERONTOLOGY TOPOLOGICAL AUDIT: '{hyper.name}'")
    print("=" * 85)
    print(f"• Total Vertices |X|: {audit['total_vertices']}")
    print(f"• Total n-ary Hyperedges |E|: {audit['total_hyperedges']}")
    print(f"• Boundary Nodes (Nodos Frontera): {audit['boundary_nodes_count']}")
    print(f"• Foundational Categories: {audit['categories_distribution']}")
    print(f"• Domain Coverage: {audit['domains_distribution']}")
    print(f"• Cardinality Spectrum (|H_e|): {audit['cardinality_spectrum']}")

    if audit["boundary_nodes"]:
        print("\n" + "-" * 85)
        print("ACTIVE BOUNDARY NODES (Nodos Frontera):")
        print("-" * 85)
        for b in audit["boundary_nodes"][:8]:
            print(f"  ⚓ {b['name']} [{b['id']}] -> Categories: {b['categories']} | Domains: {b['domains']} | Deg={b['degree']} | Prime p={b['prime_label']}")

    print("\n" + "-" * 85)
    print("TOP n-ary HYPEREDGES (Ordered by Sabidussi-Vizing Factorization Pos):")
    print("-" * 85)
    for e in audit["edges"][:max_edges]:
        print(f"[{e['edge_id']}] ({e['relation_type']} -> {e['relation']}, |H_e|={e['cardinality']}-ary, L_{e['raptor_layer']})")
        print(f"  Participants: {', '.join(e['participating_nodes'])}")
        if e["evidence"]:
            print(f"  Notes/Evidence: {e['evidence'][:140]}...")
        print()
    print("=" * 85)


def generate_interactive_html(hyper: Hyperontology, output_html_path: str = "hyperontology_graph.html"):
    nodes_data = []
    edges_data = []

    for nid, node in hyper.nodes.items():
        color = "#e3a008" if node.is_boundary else ("#1c64f2" if "Process" in node.categories else "#31c48d")
        size = 18 + min(20, node.connectivity_degree * 3)
        nodes_data.append({
            "id": nid,
            "label": f"{node.name}\n(p={node.prime_label})",
            "title": f"Categories: {node.categories}<br>Labels: {node.labels}<br>Attributes: {json.dumps(node.attributes)}",
            "color": color,
            "shape": "dot",
            "size": size,
            "font": {"color": "#f8fafc", "size": 12}
        })

    for edge in hyper.edges.values():
        hub_id = f"hub_{edge.id}"
        nodes_data.append({
            "id": hub_id,
            "label": f"{edge.relation_type}\n({edge.cardinality}-ary)",
            "title": f"Hyperedge: {edge.id}<br>Relation: {edge.relation}<br>Notes: {edge.notes}",
            "color": "#9061f9",
            "shape": "diamond",
            "size": 14 + edge.cardinality * 2,
            "font": {"color": "#e9d5ff", "size": 11}
        })

        for p_id in edge.nodes:
            if p_id in hyper.nodes:
                edges_data.append({
                    "from": p_id,
                    "to": hub_id,
                    "color": {"color": "#64748b", "highlight": "#9061f9"},
                    "width": 1.5
                })

    html_content = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>Hyperontology Interactive Topological Graph</title>
  <script type="text/javascript" src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
  <style>
    body {{ margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: #0f172a; color: #f8fafc; }}
    #header {{ padding: 14px 20px; background: #1e293b; border-bottom: 1px solid #334155; display: flex; justify-content: space-between; align-items: center; }}
    .badge {{ padding: 4px 10px; border-radius: 9999px; font-size: 12px; font-weight: 600; margin-left: 8px; }}
    .badge-gold {{ background: #78350f; color: #fde68a; }}
    .badge-purple {{ background: #581c87; color: #e9d5ff; }}
    .badge-green {{ background: #064e3b; color: #a7f3d0; }}
    #network {{ width: 100vw; height: 600px; background: #020617; }}
  </style>
</head>
<body>
  <div id="header">
    <div>
      <strong>Hyperontology H = (X, E) Interactive Topological Graph</strong>: {hyper.name}
      <span class="badge badge-gold">Gold = Multi-Category / Boundary Node</span>
      <span class="badge badge-purple">Purple Diamond = n-ary Hyperedge Hub</span>
      <span class="badge badge-green">Green/Blue = Concept / Process</span>
    </div>
    <div>Vertices: {len(hyper.nodes)} | Hyperedges: {len(hyper.edges)}</div>
  </div>
  <div id="network"></div>
  <script type="text/javascript">
    const nodes = new vis.DataSet({json.dumps(nodes_data)});
    const edges = new vis.DataSet({json.dumps(edges_data)});
    const container = document.getElementById('network');
    const data = {{ nodes: nodes, edges: edges }};
    const options = {{
      physics: {{
        barnesHut: {{ gravitationalConstant: -3500, centralGravity: 0.35, springLength: 95, damping: 0.09 }}
      }},
      interaction: {{ hover: true, tooltipDelay: 100, navigationButtons: true }}
    }};
    const network = new vis.Network(container, data, options);
  </script>
</body>
</html>"""

    with open(output_html_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    return output_html_path
