"""
Rigorous Benchmark — Fixed Version (No Circular Evaluation)
=============================================================
Runs the ACTUAL pipeline (S1→S2→S3→S4) on all 200 benchmark questions.
Captures real S1 output and compares against ground truth.
Includes ablation modes: full pipeline, LLM-only, regex-only.

Usage:
    python benchmark_rigorous.py
    python benchmark_rigorous.py --ablation
    python benchmark_rigorous.py --group DetectTrend
    python benchmark_rigorous.py --skip-stage3 --skip-stage4
    python benchmark_rigorous.py --output results_v3.json
"""
from __future__ import annotations
import sys, io
if __name__ == '__main__':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import json, time, os, re
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any, Tuple
from collections import Counter, defaultdict
import numpy as np

from dotenv import load_dotenv
load_dotenv(override=True)
from config_utils import get_benchmark_model, get_db_name, get_ollama_host, get_ollama_key, required_env

from pymongo import MongoClient
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

from s1_decomposer import (
    QuestionDecomposer, DecomposedQuery, format_decomposition,
    _regex_decompose, _validate_and_repair, _repair_operation,
)
from s2_query import run_stage2, _execute_on_collection, _safety_repair_op
from s3_answer import run_stage3
from s4_visualization import run_stage4
from query_plan import VALID_OPS, SCHEMA_REGISTRY, DataSource

# Import the 200-question suite from benchmark.py
from benchmark import BENCHMARK_QUESTIONS


# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════

MONGODB_URI = os.getenv("MONGODB_URI")
DB_NAME     = get_db_name()
OLLAMA_HOST = get_ollama_host()
OLLAMA_KEY  = get_ollama_key()

HEADERS_OLLAMA = {"Authorization": f"Bearer {OLLAMA_KEY}"} if OLLAMA_KEY else {}


# ══════════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class Stage1Result:
    """Real S1 output captured from the pipeline."""
    predicted_op: str
    predicted_cols: List[str]
    predicted_fields: List[str]
    predicted_time: str
    predicted_intent: str
    plan_source: str          # "llm" | "regex" | "repaired"
    confidence: float
    was_repaired: bool
    repair_notes: List[str]
    latency_ms: float


@dataclass
class Stage2Result:
    has_data: bool
    result_preview: str       # first 200 chars of context
    latency_ms: float


@dataclass
class Stage3Result:
    generated: bool
    answer_preview: str       # first 300 chars
    latency_ms: float


@dataclass
class Stage4Result:
    chart_generated: bool
    chart_path: Optional[str]
    chart_exists: bool        # file actually exists on disk?
    latency_ms: float


@dataclass
class QuestionResult:
    question_id: str
    group: str
    question: str
    expected_op: str
    expected_cols: List[str]
    expected_fields: List[str]
    expected_time: str

    # Real predictions
    s1: Stage1Result
    s2: Stage2Result
    s3: Stage3Result
    s4: Stage4Result

    # Per-stage correctness (computed from real output vs ground truth)
    s1_op_correct: bool
    s1_cols_correct: bool
    s1_fields_jaccard: float
    s1_time_correct: bool

    total_latency_ms: float
    error_message: Optional[str] = None


@dataclass
class AblationResult:
    mode: str            # "full" | "llm_only" | "regex_only"
    question_id: str
    question: str
    predicted_op: str
    predicted_cols: List[str]
    predicted_fields: List[str]
    predicted_time: str
    op_correct: bool
    cols_correct: bool
    fields_jaccard: float
    time_correct: bool
    latency_ms: float


# ══════════════════════════════════════════════════════════════════════════════
# SCORING HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _jaccard(set_a: set, set_b: set) -> float:
    """Jaccard similarity between two sets."""
    if not set_a and not set_b:
        return 1.0
    union = set_a | set_b
    if not union:
        return 0.0
    return len(set_a & set_b) / len(union)


def _cols_match(predicted: List[str], expected: List[str]) -> bool:
    """Check if predicted collections are a superset of expected (allow extra)."""
    pred_set = set(predicted)
    exp_set = set(expected)
    return exp_set.issubset(pred_set)


def _normalize_op(op: str) -> str:
    """Normalize operation string for comparison."""
    op = op.strip()
    if op in VALID_OPS:
        return op
    # Try repair
    repaired, _ = _repair_operation(op, "")
    return repaired


def _context_has_data(context: str) -> bool:
    """Return False for empty contexts and known Stage 2 failure/no-data markers."""
    if not context or not context.strip():
        return False
    lower = context.lower()
    failure_markers = (
        "no data found",
        "query error",
        "unknown collection",
        "insufficient data",
        "no data retrieved",
    )
    return not any(marker in lower for marker in failure_markers)


# ══════════════════════════════════════════════════════════════════════════════
# CONNECTION SETUP
# ══════════════════════════════════════════════════════════════════════════════

def _setup_connections():
    """Set up MongoDB + LLM connections."""
    mongo_client = MongoClient(required_env("MONGODB_URI"), serverSelectionTimeoutMS=5000)
    db = mongo_client[get_db_name()]
    klaen_col = db["plalion_klaen_sensor"]
    company_col = db["plalion_company_sensor"]
    weather_col = db["lighting_weatherapi"]

    # Use local Ollama — model set via BENCHMARK_MODEL env var (default: qwen3:8b)
    model_name = get_benchmark_model()
    ollama_base = get_ollama_host()
    # Cloud models need API key — set via env var for ollama client
    ollama_key = get_ollama_key()
    if ollama_key:
        os.environ["OLLAMA_API_KEY"] = ollama_key
    os.environ["OLLAMA_HOST"] = ollama_base
    llm_kwargs = dict(
        model=model_name,
        base_url=ollama_base,
        temperature=0.3,
    )
    if ollama_key:
        llm_kwargs["api_key"] = ollama_key
        llm_kwargs["client_kwargs"] = {"headers": {"Authorization": f"Bearer {ollama_key}"}}
    llm = ChatOllama(**llm_kwargs)

    decomposer = QuestionDecomposer(groq_llm=llm)

    return {
        "mongo_client": mongo_client,
        "db": db,
        "klaen_col": klaen_col,
        "company_col": company_col,
        "weather_col": weather_col,
        "llm": llm,
        "decomposer": decomposer,
    }


# ══════════════════════════════════════════════════════════════════════════════
# MAIN BENCHMARK RUNNER — ACTUAL PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def run_single_question(
    question_data: dict,
    decomposer: QuestionDecomposer,
    llm: ChatOllama,
    klaen_col,
    company_col,
    weather_col,
    skip_stage3: bool = False,
    skip_stage4: bool = False,
) -> QuestionResult:
    """Run the ACTUAL pipeline on a single question. No shortcuts."""

    q = question_data["question"]
    exp_op = question_data["expected_op"]
    exp_cols = question_data["expected_cols"]
    exp_fields = question_data["expected_fields"]
    exp_time = question_data["expected_time"]
    qid = question_data["id"]
    group = question_data["group"]

    t_total = time.monotonic()
    error_msg = None

    # ── STAGE 1: Real decomposer call ────────────────────────────────────────
    t1 = time.monotonic()
    try:
        dq = decomposer.decompose(q)
        s1_latency = (time.monotonic() - t1) * 1000

        # Capture REAL predictions from the DecomposedQuery
        s1_result = Stage1Result(
            predicted_op=dq.operation,
            predicted_cols=list(dq.collections),
            predicted_fields=list(dq.fields),
            predicted_time=dq.time_type,
            predicted_intent=dq.plan.semantic.intent_type if dq.plan else "unknown",
            plan_source=dq.plan.plan_source if dq.plan else "unknown",
            confidence=dq.plan.validation.confidence if dq.plan else 0.0,
            was_repaired=dq.plan.validation.repaired if dq.plan else False,
            repair_notes=dq.plan.validation.repair_notes if dq.plan else [],
            latency_ms=s1_latency,
        )

        # Compute correctness against ground truth
        s1_op_correct = (_normalize_op(dq.operation) == exp_op)
        s1_cols_correct = _cols_match(list(dq.collections), exp_cols)
        s1_fields_jaccard = _jaccard(set(dq.fields), set(exp_fields))
        s1_time_correct = (dq.time_type == exp_time)

    except Exception as e:
        error_msg = f"S1 error: {e}"
        s1_result = Stage1Result(
            predicted_op="ERROR", predicted_cols=[], predicted_fields=[],
            predicted_time="ERROR", predicted_intent="error",
            plan_source="error", confidence=0.0, was_repaired=False,
            repair_notes=[error_msg], latency_ms=0.0,
        )
        s1_op_correct = False
        s1_cols_correct = False
        s1_fields_jaccard = 0.0
        s1_time_correct = False
        dq = None

    # ── STAGE 2: Real query execution ────────────────────────────────────────
    t2 = time.monotonic()
    try:
        if dq is not None:
            context = run_stage2(dq, klaen_col, company_col, weather_col)
        else:
            context = ""
        s2_latency = (time.monotonic() - t2) * 1000
        s2_result = Stage2Result(
            has_data=_context_has_data(context),
            result_preview=context[:200],
            latency_ms=s2_latency,
        )
    except Exception as e:
        s2_result = Stage2Result(has_data=False, result_preview=f"Error: {e}", latency_ms=0.0)

    # ── STAGE 3: Real answer generation ──────────────────────────────────────
    t3 = time.monotonic()
    if skip_stage3 or dq is None:
        s3_result = Stage3Result(generated=False, answer_preview="", latency_ms=0.0)
    else:
        try:
            answer = run_stage3(q, context, dq, llm)
            s3_latency = (time.monotonic() - t3) * 1000
            s3_result = Stage3Result(
                generated=True,
                answer_preview=answer[:300],
                latency_ms=s3_latency,
            )
        except Exception as e:
            s3_result = Stage3Result(generated=False, answer_preview=f"Error: {e}", latency_ms=0.0)

    # ── STAGE 4: Real visualization ──────────────────────────────────────────
    t4 = time.monotonic()
    if skip_stage4 or dq is None:
        s4_result = Stage4Result(chart_generated=False, chart_path=None, chart_exists=False, latency_ms=0.0)
    else:
        try:
            chart_path = run_stage4(context, dq, q)
            s4_latency = (time.monotonic() - t4) * 1000
            chart_exists = chart_path is not None and os.path.exists(chart_path) if chart_path else False
            s4_result = Stage4Result(
                chart_generated=chart_path is not None,
                chart_path=chart_path,
                chart_exists=chart_exists,
                latency_ms=s4_latency,
            )
        except Exception as e:
            s4_result = Stage4Result(chart_generated=False, chart_path=None, chart_exists=False, latency_ms=0.0)

    total_latency = (time.monotonic() - t_total) * 1000

    return QuestionResult(
        question_id=qid,
        group=group,
        question=q,
        expected_op=exp_op,
        expected_cols=exp_cols,
        expected_fields=exp_fields,
        expected_time=exp_time,
        s1=s1_result,
        s2=s2_result,
        s3=s3_result,
        s4=s4_result,
        s1_op_correct=s1_op_correct,
        s1_cols_correct=s1_cols_correct,
        s1_fields_jaccard=s1_fields_jaccard,
        s1_time_correct=s1_time_correct,
        total_latency_ms=total_latency,
        error_message=error_msg,
    )


# ══════════════════════════════════════════════════════════════════════════════
# ABLATION RUNNER
# ══════════════════════════════════════════════════════════════════════════════

def run_ablation(
    questions: List[dict],
    decomposer: QuestionDecomposer,
    modes: List[str] = None,
) -> Dict[str, List[AblationResult]]:
    """
    Run ablation study comparing:
      - "full":       LLM + regex fallback + self-repair
      - "llm_only":  LLM only (disable regex fallback — if LLM fails, mark error)
      - "regex_only": Regex decomposition only (no LLM call)
    """
    modes = modes or ["full", "llm_only", "regex_only"]
    all_results: Dict[str, List[AblationResult]] = {m: [] for m in modes}

    for q_data in questions:
        q = q_data["question"]
        exp_op = q_data["expected_op"]
        exp_cols = q_data["expected_cols"]
        exp_fields = q_data["expected_fields"]
        exp_time = q_data["expected_time"]

        # ── Mode: full pipeline ────────────────────────────────────────────
        if "full" in modes:
            t0 = time.monotonic()
            try:
                dq = decomposer.decompose(q)
                latency = (time.monotonic() - t0) * 1000
                pred = AblationResult(
                    mode="full", question_id=q_data["id"], question=q,
                    predicted_op=dq.operation,
                    predicted_cols=list(dq.collections),
                    predicted_fields=list(dq.fields),
                    predicted_time=dq.time_type,
                    op_correct=(_normalize_op(dq.operation) == exp_op),
                    cols_correct=_cols_match(list(dq.collections), exp_cols),
                    fields_jaccard=_jaccard(set(dq.fields), set(exp_fields)),
                    time_correct=(dq.time_type == exp_time),
                    latency_ms=latency,
                )
            except Exception as e:
                pred = AblationResult(
                    mode="full", question_id=q_data["id"], question=q,
                    predicted_op="ERROR", predicted_cols=[], predicted_fields=[],
                    predicted_time="ERROR",
                    op_correct=False, cols_correct=False, fields_jaccard=0.0,
                    time_correct=False, latency_ms=0.0,
                )
            all_results["full"].append(pred)

        # ── Mode: regex_only ─────────────────────────────────────────────────
        if "regex_only" in modes:
            t0 = time.monotonic()
            parsed, conf = _regex_decompose(q)
            latency = (time.monotonic() - t0) * 1000

            # Apply validation to regex output too (for fair comparison)
            cols, flds, op, tt, vr = _validate_and_repair(
                parsed.get("collections", []),
                parsed.get("fields", []),
                parsed.get("operation", "DetectLatest"),
                parsed.get("time_type", "global"),
                q,
            )
            all_results["regex_only"].append(AblationResult(
                mode="regex_only", question_id=q_data["id"], question=q,
                predicted_op=op,
                predicted_cols=cols,
                predicted_fields=flds,
                predicted_time=tt,
                op_correct=(_normalize_op(op) == exp_op),
                cols_correct=_cols_match(cols, exp_cols),
                fields_jaccard=_jaccard(set(flds), set(exp_fields)),
                time_correct=(tt == exp_time),
                latency_ms=latency,
            ))

        # ── Mode: llm_only (simulated — we run decomposer but track fallback) ─
        if "llm_only" in modes:
            # Run full pipeline, but mark as "llm_only failure" if it fell back to regex
            t0 = time.monotonic()
            try:
                dq = decomposer.decompose(q)
                latency = (time.monotonic() - t0) * 1000
                plan_source = dq.plan.plan_source if dq.plan else "unknown"

                # If pipeline fell back to regex, mark as LLM failure
                if plan_source == "regex":
                    pred = AblationResult(
                        mode="llm_only", question_id=q_data["id"], question=q,
                        predicted_op="LLM_FAILED", predicted_cols=[], predicted_fields=[],
                        predicted_time="LLM_FAILED",
                        op_correct=False, cols_correct=False, fields_jaccard=0.0,
                        time_correct=False, latency_ms=latency,
                    )
                else:
                    pred = AblationResult(
                        mode="llm_only", question_id=q_data["id"], question=q,
                        predicted_op=dq.operation,
                        predicted_cols=list(dq.collections),
                        predicted_fields=list(dq.fields),
                        predicted_time=dq.time_type,
                        op_correct=(_normalize_op(dq.operation) == exp_op),
                        cols_correct=_cols_match(list(dq.collections), exp_cols),
                        fields_jaccard=_jaccard(set(dq.fields), set(exp_fields)),
                        time_correct=(dq.time_type == exp_time),
                        latency_ms=latency,
                    )
            except Exception:
                pred = AblationResult(
                    mode="llm_only", question_id=q_data["id"], question=q,
                    predicted_op="LLM_FAILED", predicted_cols=[], predicted_fields=[],
                    predicted_time="LLM_FAILED",
                    op_correct=False, cols_correct=False, fields_jaccard=0.0,
                    time_correct=False, latency_ms=0.0,
                )
            all_results["llm_only"].append(pred)

    return all_results


# ══════════════════════════════════════════════════════════════════════════════
# CONFUSION MATRIX + ERROR ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

def build_confusion_matrix(results: List[QuestionResult]) -> dict:
    """Build 10×10 confusion matrix for operation detection."""
    all_ops = sorted(VALID_OPS)
    op_idx = {op: i for i, op in enumerate(all_ops)}
    n = len(all_ops)
    matrix = [[0] * n for _ in range(n)]

    for r in results:
        exp = r.expected_op
        pred = _normalize_op(r.s1.predicted_op)
        if exp in op_idx and pred in op_idx:
            matrix[op_idx[exp]][op_idx[pred]] += 1

    return {
        "operations": all_ops,
        "matrix": matrix,
    }


def build_error_taxonomy(results: List[QuestionResult]) -> dict:
    """Classify every failure into a taxonomy."""
    taxonomy = {
        "s1_wrong_operation": [],
        "s1_wrong_collection": [],
        "s1_wrong_fields": [],
        "s1_wrong_time": [],
        "s2_no_data": [],
        "s4_chart_failure": [],
        "pipeline_error": [],
    }

    for r in results:
        if r.error_message:
            taxonomy["pipeline_error"].append({
                "id": r.question_id, "question": r.question,
                "error": r.error_message,
            })
            continue

        if not r.s1_op_correct:
            taxonomy["s1_wrong_operation"].append({
                "id": r.question_id, "question": r.question,
                "expected": r.expected_op, "predicted": r.s1.predicted_op,
                "plan_source": r.s1.plan_source,
            })
        if not r.s1_cols_correct:
            taxonomy["s1_wrong_collection"].append({
                "id": r.question_id, "question": r.question,
                "expected": r.expected_cols, "predicted": r.s1.predicted_cols,
            })
        if r.s1_fields_jaccard < 0.5:
            taxonomy["s1_wrong_fields"].append({
                "id": r.question_id, "question": r.question,
                "expected": r.expected_fields, "predicted": r.s1.predicted_fields,
                "jaccard": round(r.s1_fields_jaccard, 3),
            })
        if not r.s1_time_correct:
            taxonomy["s1_wrong_time"].append({
                "id": r.question_id, "question": r.question,
                "expected": r.expected_time, "predicted": r.s1.predicted_time,
            })
        if not r.s2.has_data:
            taxonomy["s2_no_data"].append({
                "id": r.question_id, "question": r.question,
                "operation": r.s1.predicted_op,
            })
        if r.s4.chart_generated and not r.s4.chart_exists:
            taxonomy["s4_chart_failure"].append({
                "id": r.question_id, "question": r.question,
                "chart_path": r.s4.chart_path,
            })

    return taxonomy


# ══════════════════════════════════════════════════════════════════════════════
# AGGREGATION + REPORTING
# ══════════════════════════════════════════════════════════════════════════════

def compute_aggregated(results: List[QuestionResult]) -> dict:
    """Compute aggregate metrics from real pipeline results."""
    n = len(results)
    if n == 0:
        return {}

    op_correct = sum(1 for r in results if r.s1_op_correct)
    cols_correct = sum(1 for r in results if r.s1_cols_correct)
    fields_jaccard_avg = np.mean([r.s1_fields_jaccard for r in results])
    time_correct = sum(1 for r in results if r.s1_time_correct)
    has_data = sum(1 for r in results if r.s2.has_data)
    charts_gen = sum(1 for r in results if r.s4.chart_generated)
    charts_exist = sum(1 for r in results if r.s4.chart_exists)
    repaired = sum(1 for r in results if r.s1.was_repaired)
    llm_source = sum(1 for r in results if r.s1.plan_source == "llm")
    errors = sum(1 for r in results if r.error_message)

    # By operation group
    by_group = defaultdict(list)
    for r in results:
        by_group[r.group].append(r)

    group_metrics = {}
    for g, grp_results in by_group.items():
        gn = len(grp_results)
        group_metrics[g] = {
            "n": gn,
            "op_accuracy": round(sum(1 for r in grp_results if r.s1_op_correct) / gn, 3),
            "cols_accuracy": round(sum(1 for r in grp_results if r.s1_cols_correct) / gn, 3),
            "avg_latency_ms": round(np.mean([r.total_latency_ms for r in grp_results]), 1),
        }

    # Confidence intervals (Wilson score interval for proportions)
    from scipy import stats as sp_stats

    def wilson_ci(p_hat, n_val, z=1.96):
        if n_val == 0:
            return [0.0, 0.0]
        denom = 1 + z**2 / n_val
        center = (p_hat + z**2 / (2 * n_val)) / denom
        margin = z * np.sqrt((p_hat * (1 - p_hat) + z**2 / (4 * n_val)) / n_val) / denom
        return [round(max(0, center - margin), 4), round(min(1, center + margin), 4)]

    op_rate = op_correct / n
    cols_rate = cols_correct / n

    return {
        "total_questions": n,
        "pipeline_errors": errors,
        "s1_operation_accuracy": round(op_rate, 4),
        "s1_operation_ci_95": wilson_ci(op_rate, n),
        "s1_collections_accuracy": round(cols_rate, 4),
        "s1_collections_ci_95": wilson_ci(cols_rate, n),
        "s1_fields_jaccard_avg": round(float(fields_jaccard_avg), 4),
        "s1_time_accuracy": round(time_correct / n, 4),
        "s2_data_hit_rate": round(has_data / n, 4),
        "s4_chart_generation_rate": round(charts_gen / n, 4),
        "s4_chart_file_exists_rate": round(charts_exist / n, 4),
        "s1_repair_rate": round(repaired / n, 4),
        "s1_llm_success_rate": round(llm_source / n, 4),
        "avg_total_latency_ms": round(np.mean([r.total_latency_ms for r in results]), 1),
        "median_total_latency_ms": round(float(np.median([r.total_latency_ms for r in results])), 1),
        "avg_s1_latency_ms": round(np.mean([r.s1.latency_ms for r in results]), 1),
        "avg_s2_latency_ms": round(np.mean([r.s2.latency_ms for r in results]), 1),
        "avg_s3_latency_ms": round(np.mean([r.s3.latency_ms for r in results if r.s3.generated]), 1),
        "avg_s4_latency_ms": round(np.mean([r.s4.latency_ms for r in results if r.s4.chart_generated]), 1),
        "by_group": group_metrics,
    }


def compute_ablation_summary(ablation_results: Dict[str, List[AblationResult]]) -> dict:
    """Summarize ablation results across modes."""
    summary = {}
    for mode, results in ablation_results.items():
        n = len(results)
        if n == 0:
            continue
        summary[mode] = {
            "n": n,
            "op_accuracy": round(sum(1 for r in results if r.op_correct) / n, 4),
            "cols_accuracy": round(sum(1 for r in results if r.cols_correct) / n, 4),
            "fields_jaccard_avg": round(np.mean([r.fields_jaccard for r in results]), 4),
            "time_accuracy": round(sum(1 for r in results if r.time_correct) / n, 4),
            "avg_latency_ms": round(np.mean([r.latency_ms for r in results]), 1),
        }
    return summary


# ══════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def run_full_benchmark(
    questions: List[dict] = None,
    skip_stage3: bool = False,
    skip_stage4: bool = True,
    group_filter: str = None,
    output_file: str = "benchmark_rigorous_results.json",
    run_ablation_flag: bool = True,
):
    questions = questions or BENCHMARK_QUESTIONS

    if group_filter:
        questions = [q for q in questions if q["group"] == group_filter]
        print(f"  Filtered to group '{group_filter}': {len(questions)} questions")

    print(f"\n{'='*72}")
    print(f"  RIGOROUS BENCHMARK — {len(questions)} Questions")
    print(f"  Skip S3: {skip_stage3} | Skip S4: {skip_stage4} | Ablation: {run_ablation_flag}")
    print(f"{'='*72}\n")

    conn = _setup_connections()
    decomposer = conn["decomposer"]
    llm = conn["llm"]

    # ── Run each question through ACTUAL pipeline ───────────────────────────
    results: List[QuestionResult] = []
    for i, q_data in enumerate(questions, 1):
        qid = q_data["id"]
        q_preview = q_data["question"][:60]
        print(f"  [{i:3d}/{len(questions)}] {qid}: {q_preview}...", end="", flush=True)

        result = run_single_question(
            q_data, decomposer, llm,
            conn["klaen_col"], conn["company_col"], conn["weather_col"],
            skip_stage3=skip_stage3,
            skip_stage4=skip_stage4,
        )
        results.append(result)

        op_mark = "✓" if result.s1_op_correct else "✗"
        print(f" → op:{op_mark} ({result.s1.predicted_op}) src:{result.s1.plan_source} "
              f"lat:{result.total_latency_ms:.0f}ms")

    # ── Aggregated metrics ───────────────────────────────────────────────────
    agg = compute_aggregated(results)
    print(f"\n{'='*72}")
    print("  AGGREGATED RESULTS")
    print(f"{'='*72}")
    for k, v in agg.items():
        if k == "by_group":
            continue
        if isinstance(v, float) and 0 < v <= 1:
            print(f"  {k:<38}: {v:.1%}")
        elif isinstance(v, list) and len(v) == 2:
            print(f"  {k:<38}: [{v[0]:.1%}, {v[1]:.1%}]")
        else:
            print(f"  {k:<38}: {v}")

    print(f"\n  By Operation Group:")
    for g, m in agg.get("by_group", {}).items():
        print(f"    {g:<20}: op_acc={m['op_accuracy']:.1%}  col_acc={m['cols_accuracy']:.1%}  "
              f"lat={m['avg_latency_ms']:.0f}ms")

    # ── Confusion matrix ─────────────────────────────────────────────────────
    cm = build_confusion_matrix(results)

    # ── Error taxonomy ──────────────────────────────────────────────────────
    taxonomy = build_error_taxonomy(results)
    print(f"\n{'='*72}")
    print("  ERROR TAXONOMY")
    print(f"{'='*72}")
    for cat, items in taxonomy.items():
        print(f"  {cat:<25}: {len(items)} failures")
        # Show up to 3 examples
        for ex in items[:3]:
            print(f"    - {ex.get('id','?')}: {ex.get('question','')[:50]}")
            if 'predicted' in ex:
                print(f"      expected={ex.get('expected')} → predicted={ex.get('predicted')}")

    # ── Ablation ─────────────────────────────────────────────────────────────
    ablation_summary = None
    ablation_results = None
    if run_ablation_flag:
        print(f"\n{'='*72}")
        print("  ABLATION STUDY")
        print(f"{'='*72}")
        ablation_results = run_ablation(questions, decomposer)
        ablation_summary = compute_ablation_summary(ablation_results)
        for mode, m in ablation_summary.items():
            print(f"  {mode:<15}: op={m['op_accuracy']:.1%}  cols={m['cols_accuracy']:.1%}  "
                  f"fields_j={m['fields_jaccard_avg']:.2f}  time={m['time_accuracy']:.1%}  "
                  f"lat={m['avg_latency_ms']:.0f}ms")

    # ── Save JSON report ─────────────────────────────────────────────────────
    report = {
        "run_timestamp": datetime.now().isoformat(),
        "config": {
            "skip_stage3": skip_stage3,
            "skip_stage4": skip_stage4,
            "num_questions": len(questions),
            "group_filter": group_filter,
        },
        "aggregated": agg,
        "confusion_matrix": cm,
        "error_taxonomy": taxonomy,
        "ablation": ablation_summary,
        "results": [asdict(r) for r in results],
    }

    if ablation_results:
        report["ablation_details"] = {mode: [asdict(r) for r in res]
                                       for mode, res in ablation_results.items()}

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str, ensure_ascii=False)
    print(f"\n  📄 Full report saved → {output_file}")

    conn["mongo_client"].close()
    return results, agg, cm, taxonomy, ablation_summary


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Rigorous Benchmark — Fixed, No Circular Eval")
    parser.add_argument("--group", type=str, default=None,
                        help="Run one group only")
    parser.add_argument("--skip-stage3", action="store_true")
    parser.add_argument("--skip-stage4", action="store_true", default=True)
    parser.add_argument("--include-s4", action="store_true",
                        help="Enable S4 visualization evaluation")
    parser.add_argument("--no-ablation", action="store_true",
                        help="Skip ablation study")
    parser.add_argument("--output", type=str, default="benchmark_rigorous_results.json")
    args = parser.parse_args()

    run_full_benchmark(
        skip_stage3=args.skip_stage3,
        skip_stage4=not args.include_s4,
        group_filter=args.group,
        output_file=args.output,
        run_ablation_flag=not args.no_ablation,
    )
