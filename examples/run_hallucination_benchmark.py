"""
Standalone Live Benchmark Runner: Auto-Hyperontology (Hyper-RAG) vs. RealFlatTextRAG (Okapi BM25)
Run via:
    PYTHONPATH=. python3 examples/run_hallucination_benchmark.py
"""

from auto_hyperontology.benchmark import run_live_rag_benchmark

SAMPLE_PUBMED_QA = [
    {
        "id": "PubMed_1",
        "question": "Does Delayed Diagnostic Protocol in Myocardial Infarction cause Medical Malpractice Claims and Workplace Productivity Loss?",
        "context": (
            "In Clinical Cardiology, Myocardial Infarction occurs when Coronary Ischemia triggers Cellular Necrosis. "
            "When Myocardial Infarction is compounded by Delayed Diagnostic Protocol in Hospital Administration, "
            "patients file Medical Malpractice Claims under Healthcare Law, and long-term disability induces "
            "Workplace Productivity Loss in Health Economics."
        ),
        "long_answer": (
            "Yes, when Myocardial Infarction is compounded by Delayed Diagnostic Protocol, it leads to "
            "Medical Malpractice Claims and Workplace Productivity Loss."
        ),
        "final_decision": "yes"
    },
    {
        "id": "PubMed_2",
        "question": "What is the clinical outcome of Low-Dose Aspirin Prophylaxis in Myocardial Infarction?",
        "context": (
            "In Clinical Cardiology, Low-Dose Aspirin Prophylaxis administered during acute Myocardial Infarction "
            "inhibits platelet aggregation and promotes Uncomplicated Clinical Recovery."
        ),
        "long_answer": (
            "Low-Dose Aspirin Prophylaxis in Myocardial Infarction inhibits platelet aggregation and promotes "
            "Uncomplicated Clinical Recovery."
        ),
        "final_decision": "yes"
    }
]


def main():
    report = run_live_rag_benchmark(SAMPLE_PUBMED_QA)
    print("\n=== BENCHMARK SUMMARY ===")
    for k, v in report["summary"].items():
        print(f"• {k}: {v}")


if __name__ == "__main__":
    main()
