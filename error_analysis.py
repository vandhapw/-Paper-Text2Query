"""
Error Analysis Module
=====================
Generates confusion matrix, failure taxonomy, and per-operation breakdown
from benchmark results. Produces publication-ready figures.

Usage:
    python error_analysis.py
    python error_analysis.py --input benchmark_rigorous_results.json
"""
from __future__ import annotations
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import json, os
from datetime import datetime
from typing import Dict, List, Any, Optional
from collections import Counter, defaultdict
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio

from query_plan import VALID_OPS


# ══════════════════════════════════════════════════════════════════════════════
# CONFUSION MATRIX
# ══════════════════════════════════════════════════════════════════════════════

def build_confusion_matrix(results: List[dict]) -> dict:
    """
    Build 10×10 confusion matrix: rows=expected, cols=predicted.
    Input: list of QuestionResult dicts (from benchmark_rigorous_results.json)
    """
    all_ops = sorted(VALID_OPS)
    op_idx = {op: i for i, op in enumerate(all_ops)}
    n = len(all_ops)
    matrix = [[0] * n for _ in range(n)]

    for r in results:
        exp = r.get("expected_op", "")
        pred = r.get("s1", {}).get("predicted_op", "ERROR")
        if exp in op_idx and pred in op_idx:
            matrix[op_idx[exp]][op_idx[pred]] += 1

    return {"operations": all_ops, "matrix": matrix}


def print_confusion_matrix(cm: dict):
    """Pretty-print confusion matrix."""
    ops = cm["operations"]
    matrix = cm["matrix"]
    n = len(ops)

    # Header
    header = f"{'':>22}|"
    for op in ops:
        header += f"{op.replace('Detect','')[:6]:>7}"
    print(header)
    print("-" * (22 + 7 * n + 1))

    for i, op in enumerate(ops):
        row = f"{op.replace('Detect','Detect'):>22}|"
        for j in range(n):
            val = matrix[i][j]
            cell = f"{val:>7}" if val > 0 else f"{'·':>7}"
            row += cell
        print(row)

    # Per-operation accuracy
    print(f"\n  Per-Operation Accuracy:")
    for i, op in enumerate(ops):
        total = sum(matrix[i])
        correct = matrix[i][i]
        acc = correct / total if total > 0 else 0
        print(f"    {op:<22}: {acc:.1%} ({correct}/{total})")


# ══════════════════════════════════════════════════════════════════════════════
# FAILURE TAXONOMY
# ══════════════════════════════════════════════════════════════════════════════

FAILURE_CATEGORIES = [
    "S1_OP_WRONG",
    "S1_COLLECTION_WRONG",
    "S1_FIELDS_WRONG",
    "S1_TIME_WRONG",
    "S1_LLM_FAILED",
    "S2_NO_DATA",
    "S2_QUERY_ERROR",
    "S3_GENERATION_FAILED",
    "S4_CHART_FAILED",
    "PIPELINE_ERROR",
]


def classify_failures(results: List[dict]) -> Dict[str, List[dict]]:
    """Classify every failure into taxonomy categories."""
    taxonomy: Dict[str, List[dict]] = {cat: [] for cat in FAILURE_CATEGORIES}

    for r in results:
        qid = r.get("question_id", "?")
        q = r.get("question", "")
        s1 = r.get("s1", {})
        s2 = r.get("s2", {})
        s3 = r.get("s3", {})
        s4 = r.get("s4", {})

        # Pipeline error
        if r.get("error_message"):
            taxonomy["PIPELINE_ERROR"].append({
                "id": qid, "question": q,
                "error": r["error_message"],
            })
            continue

        # S1: Wrong operation
        if not r.get("s1_op_correct", True):
            taxonomy["S1_OP_WRONG"].append({
                "id": qid, "question": q,
                "expected": r.get("expected_op"),
                "predicted": s1.get("predicted_op"),
                "plan_source": s1.get("plan_source"),
                "was_repaired": s1.get("was_repaired"),
            })

        # S1: Wrong collection
        if not r.get("s1_cols_correct", True):
            taxonomy["S1_COLLECTION_WRONG"].append({
                "id": qid, "question": q,
                "expected": r.get("expected_cols"),
                "predicted": s1.get("predicted_cols"),
            })

        # S1: Wrong fields (Jaccard < 0.5)
        if r.get("s1_fields_jaccard", 1.0) < 0.5:
            taxonomy["S1_FIELDS_WRONG"].append({
                "id": qid, "question": q,
                "expected": r.get("expected_fields"),
                "predicted": s1.get("predicted_fields"),
                "jaccard": round(r.get("s1_fields_jaccard", 0), 3),
            })

        # S1: Wrong time scope
        if not r.get("s1_time_correct", True):
            taxonomy["S1_TIME_WRONG"].append({
                "id": qid, "question": q,
                "expected": r.get("expected_time"),
                "predicted": s1.get("predicted_time"),
            })

        # S1: LLM failed (fell back to regex)
        if s1.get("plan_source") == "regex" and s1.get("predicted_op") != "ERROR":
            taxonomy["S1_LLM_FAILED"].append({
                "id": qid, "question": q,
                "operation": s1.get("predicted_op"),
            })

        # S2: No data returned
        if not s2.get("has_data", True):
            taxonomy["S2_NO_DATA"].append({
                "id": qid, "question": q,
                "operation": s1.get("predicted_op"),
                "time_scope": s1.get("predicted_time"),
            })

        # S4: Chart generated but file doesn't exist
        if s4.get("chart_generated") and not s4.get("chart_exists", False):
            taxonomy["S4_CHART_FAILED"].append({
                "id": qid, "question": q,
                "chart_path": s4.get("chart_path"),
            })

    return taxonomy


def print_failure_taxonomy(taxonomy: Dict[str, List[dict]]):
    """Pretty-print failure taxonomy with examples."""
    total = sum(len(v) for v in taxonomy.values())
    print(f"\n  FAILURE TAXONOMY ({total} total failures)")
    print(f"  {'─'*60}")

    for cat, items in sorted(taxonomy.items(), key=lambda x: -len(x[1])):
        if not items:
            continue
        pct = len(items) / total * 100 if total > 0 else 0
        print(f"  {cat:<25}: {len(items):>4} ({pct:>5.1f}%)")

        # Show representative examples
        for ex in items[:3]:
            qid = ex.get("id", "?")
            q_text = ex.get("question", "")[:45]
            extra = ""
            if "expected" in ex and "predicted" in ex:
                extra = f" → exp={ex['expected']} pred={ex['predicted']}"
            elif "error" in ex:
                extra = f" → {str(ex['error'])[:40]}"
            print(f"    • {qid}: {q_text}{extra}")
        if len(items) > 3:
            print(f"    ... and {len(items)-3} more")
        print()


# ══════════════════════════════════════════════════════════════════════════════
# PER-OPERATION BREAKDOWN
# ══════════════════════════════════════════════════════════════════════════════

def per_operation_breakdown(results: List[dict]) -> Dict[str, dict]:
    """Compute accuracy metrics per expected operation group."""
    by_op: Dict[str, dict] = defaultdict(lambda: {
        "total": 0, "op_correct": 0, "cols_correct": 0,
        "fields_jaccard": [], "time_correct": 0, "latency": [],
    })

    for r in results:
        exp_op = r.get("expected_op", "Unknown")
        d = by_op[exp_op]
        d["total"] += 1
        d["op_correct"] += int(r.get("s1_op_correct", False))
        d["cols_correct"] += int(r.get("s1_cols_correct", False))
        d["fields_jaccard"].append(r.get("s1_fields_jaccard", 0))
        d["time_correct"] += int(r.get("s1_time_correct", False))
        d["latency"].append(r.get("total_latency_ms", 0))

    summary = {}
    for op, d in sorted(by_op.items()):
        n = d["total"]
        summary[op] = {
            "n": n,
            "op_accuracy": round(d["op_correct"] / n, 4) if n else 0,
            "cols_accuracy": round(d["cols_correct"] / n, 4) if n else 0,
            "avg_fields_jaccard": round(np.mean(d["fields_jaccard"]), 4) if d["fields_jaccard"] else 0,
            "time_accuracy": round(d["time_correct"] / n, 4) if n else 0,
            "avg_latency_ms": round(np.mean(d["latency"]), 1) if d["latency"] else 0,
        }
    return summary


def print_per_operation(summary: Dict[str, dict]):
    """Pretty-print per-operation breakdown."""
    print(f"\n  PER-OPERATION BREAKDOWN")
    print(f"  {'─'*80}")
    print(f"  {'Operation':<22} {'N':>4} {'OpAcc':>7} {'ColAcc':>7} {'FldJac':>7} {'TimeAcc':>7} {'Lat ms':>8}")
    print(f"  {'─'*80}")
    for op, m in summary.items():
        print(f"  {op:<22} {m['n']:>4} {m['op_accuracy']:>6.1%} {m['cols_accuracy']:>6.1%} "
              f"{m['avg_fields_jaccard']:>6.2f} {m['time_accuracy']:>6.1%} {m['avg_latency_ms']:>7.0f}")


# ══════════════════════════════════════════════════════════════════════════════
# REPAIR IMPACT ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

def repair_impact(results: List[dict]) -> dict:
    """Analyze how self-repair affects outcomes."""
    repaired_count = sum(1 for r in results if r.get("s1", {}).get("was_repaired", False))
    total = len(results)

    # Among repaired, how many ended up correct?
    repaired_results = [r for r in results if r.get("s1", {}).get("was_repaired", False)]
    repaired_correct = sum(1 for r in repaired_results if r.get("s1_op_correct", False))

    # Among non-repaired, how many correct?
    non_repaired = [r for r in results if not r.get("s1", {}).get("was_repaired", False)]
    non_repaired_correct = sum(1 for r in non_repaired if r.get("s1_op_correct", False))

    return {
        "total": total,
        "repaired_count": repaired_count,
        "repair_rate": round(repaired_count / total, 4) if total else 0,
        "repaired_op_accuracy": round(repaired_correct / len(repaired_results), 4) if repaired_results else 0,
        "non_repaired_op_accuracy": round(non_repaired_correct / len(non_repaired), 4) if non_repaired else 0,
        "repair_notes": Counter(
            note
            for r in results
            for note in r.get("s1", {}).get("repair_notes", [])
        ),
    }


def print_repair_impact(ri: dict):
    print(f"\n  SELF-REPAIR IMPACT")
    print(f"  {'─'*50}")
    print(f"  Repair rate:           {ri['repair_rate']:.1%} ({ri['repaired_count']}/{ri['total']})")
    print(f"  Repaired → op correct: {ri['repaired_op_accuracy']:.1%}")
    print(f"  Non-repaired → correct: {ri['non_repaired_op_accuracy']:.1%}")
    print(f"  Common repair notes:")
    for note, count in ri["repair_notes"].most_common(5):
        print(f"    • {note}: {count}×")


# ══════════════════════════════════════════════════════════════════════════════
# PUBLICATION-READY FIGURES
# ══════════════════════════════════════════════════════════════════════════════

def generate_figures(cm: dict, taxonomy: dict, per_op: dict, ri: dict,
                     output_dir: str = "error_analysis_figures"):
    """Generate Plotly figures for publication."""
    os.makedirs(output_dir, exist_ok=True)
    try:
        from plotly.subplots import make_subplots
    except ImportError:
        print("  ⚠️ Plotly not available, skipping figure generation")
        return

    ops = cm["operations"]
    matrix = cm["matrix"]

    # ── Figure 1: Confusion Matrix Heatmap ──────────────────────────────────
    short_ops = [op.replace("Detect", "D.") for op in ops]
    fig = go.Figure(data=go.Heatmap(
        z=matrix,
        x=short_ops,
        y=short_ops,
        colorscale="Blues",
        text=[[str(v) if v > 0 else "" for v in row] for row in matrix],
        texttemplate="%{text}",
        hovertemplate="Expected: %{y}<br>Predicted: %{x}<br>Count: %{z}<extra></extra>",
    ))
    fig.update_layout(
        title="Operation Detection Confusion Matrix",
        xaxis_title="Predicted Operation",
        yaxis_title="Expected Operation",
        width=700, height=700,
        template="plotly_white",
        font=dict(size=11),
    )
    fig.write_image(os.path.join(output_dir, "confusion_matrix.png"), scale=2)
    fig.write_image(os.path.join(output_dir, "confusion_matrix.pdf"))
    print(f"  ✅ Confusion matrix → {output_dir}/confusion_matrix.png")

    # ── Figure 2: Per-Operation Accuracy Bar Chart ──────────────────────────
    fig2 = go.Figure()
    op_names = list(per_op.keys())
    op_accs = [per_op[op]["op_accuracy"] for op in op_names]
    col_accs = [per_op[op]["cols_accuracy"] for op in op_names]

    fig2.add_trace(go.Bar(
        name="Operation Acc", x=op_names, y=op_accs,
        marker_color="#2E86AB",
    ))
    fig2.add_trace(go.Bar(
        name="Collection Acc", x=op_names, y=col_accs,
        marker_color="#A23B72",
    ))
    fig2.update_layout(
        title="Accuracy by Operation Type",
        yaxis_title="Accuracy",
        yaxis_range=[0, 1.05],
        barmode="group",
        width=900, height=450,
        template="plotly_white",
    )
    fig2.write_image(os.path.join(output_dir, "per_operation_accuracy.png"), scale=2)
    print(f"  ✅ Per-operation accuracy → {output_dir}/per_operation_accuracy.png")

    # ── Figure 3: Failure Distribution Pie ───────────────────────────────────
    failure_counts = {cat: len(items) for cat, items in taxonomy.items() if items}
    if failure_counts:
        fig3 = go.Figure(data=go.Pie(
            labels=list(failure_counts.keys()),
            values=list(failure_counts.values()),
            hole=0.4,
        ))
        fig3.update_layout(
            title="Failure Distribution by Category",
            width=700, height=500,
            template="plotly_white",
        )
        fig3.write_image(os.path.join(output_dir, "failure_distribution.png"), scale=2)
        print(f"  ✅ Failure distribution → {output_dir}/failure_distribution.png")


def generate_figures_html(cm, taxonomy, per_op, ri):
    """Save figures as interactive HTML when Kaleido/Chrome is not available."""
    output_dir = "figures"
    os.makedirs(output_dir, exist_ok=True)

    # Parse confusion matrix
    ops = cm.get("operations", list(cm.keys()))
    short_ops = [k.replace("Detect", "") for k in ops]
    matrix = cm.get("matrix", [])
    if not matrix:
        matrix = []
        for row_key in ops:
            row_data = cm.get(row_key, {})
            if isinstance(row_data, list):
                matrix.append(row_data)
            else:
                matrix.append([row_data.get(col_key, 0) for col_key in ops])

    # Figure 1: Confusion Matrix
    fig = go.Figure(data=go.Heatmap(
        z=matrix, x=short_ops, y=short_ops, colorscale="Blues",
        text=[[str(v) if v > 0 else "" for v in row] for row in matrix],
        texttemplate="%{text}",
    ))
    fig.update_layout(title="Operation Detection Confusion Matrix",
        xaxis_title="Predicted", yaxis_title="Expected", width=700, height=700)
    fig.write_html(os.path.join(output_dir, "confusion_matrix.html"))
    print(f"  ✅ Confusion matrix → {output_dir}/confusion_matrix.html")

    # Figure 2: Per-Operation Accuracy
    op_names = list(per_op.keys())
    op_accs = [per_op[op]["op_accuracy"] for op in op_names]
    col_accs = [per_op[op]["cols_accuracy"] for op in op_names]
    fig2 = go.Figure()
    fig2.add_trace(go.Bar(name="Operation Acc", x=op_names, y=op_accs, marker_color="#2E86AB"))
    fig2.add_trace(go.Bar(name="Collection Acc", x=op_names, y=col_accs, marker_color="#A23B72"))
    fig2.update_layout(title="Accuracy by Operation Type", yaxis_title="Accuracy",
        yaxis_range=[0, 1.05], barmode="group", width=900, height=450)
    fig2.write_html(os.path.join(output_dir, "per_operation_accuracy.html"))
    print(f"  ✅ Per-operation accuracy → {output_dir}/per_operation_accuracy.html")

    # Figure 3: Failure Distribution
    failure_counts = {cat: len(items) for cat, items in taxonomy.items() if items}
    if failure_counts:
        fig3 = go.Figure(data=go.Pie(
            labels=list(failure_counts.keys()), values=list(failure_counts.values()), hole=0.4))
        fig3.update_layout(title="Failure Distribution by Category", width=700, height=500)
        fig3.write_html(os.path.join(output_dir, "failure_distribution.html"))
        print(f"  ✅ Failure distribution → {output_dir}/failure_distribution.html")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def run_error_analysis(input_file: str = "benchmark_rigorous_results.json"):
    print(f"\n{'='*72}")
    print(f"  ERROR ANALYSIS — {input_file}")
    print(f"{'='*72}\n")

    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    results = data.get("results", [])

    # Confusion matrix
    cm = build_confusion_matrix(results)
    print_confusion_matrix(cm)

    # Failure taxonomy
    taxonomy = classify_failures(results)
    print_failure_taxonomy(taxonomy)

    # Per-operation breakdown
    per_op = per_operation_breakdown(results)
    print_per_operation(per_op)

    # Repair impact
    ri = repair_impact(results)
    print_repair_impact(ri)

    # Save analysis JSON first (always succeeds)
    analysis = {
        "timestamp": datetime.now().isoformat(),
        "confusion_matrix": cm,
        "failure_taxonomy": {k: len(v) for k, v in taxonomy.items()},
        "failure_details": taxonomy,
        "per_operation": per_op,
        "repair_impact": ri,
    }
    out_path = input_file.replace(".json", "_error_analysis.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(analysis, f, indent=2, default=str, ensure_ascii=False)
    print(f"\n  📄 Error analysis saved → {out_path}")

    # Generate figures (try PNG/PDF via Kaleido, fall back to HTML)
    try:
        generate_figures(cm, taxonomy, per_op, ri)
    except Exception as e:
        print(f"\n  ⚠️ Kaleido/Chrome not available: {e}")
        print(f"  📊 Saving figures as HTML instead...")
        generate_figures_html(cm, taxonomy, per_op, ri)

    return analysis


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="benchmark_rigorous_results.json")
    args = parser.parse_args()

    run_error_analysis(args.input)