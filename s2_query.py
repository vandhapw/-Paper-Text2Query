"""
Stage 2 – Query Execution (IR-Driven, Time-Aware, Multi-Collection Grounded)
Includes execution-time compound operation safety net.
"""
from __future__ import annotations
import re, time
from datetime import datetime
from typing import Any
from pymongo.collection import Collection
from s1_decomposer import DecomposedQuery
from query_plan import TimeGranularity, SCHEMA_REGISTRY, VALID_OPS


def _fmt_ts(ts: Any) -> str:
    if isinstance(ts, datetime):
        return ts.strftime("%Y-%m-%d %H:%M UTC")
    return str(ts)


def _doc_to_str(doc: dict, fields: list[str]) -> str:
    ts    = _fmt_ts(doc.get("timestamp", "N/A"))
    parts = [f"[{ts}]"]
    for f in fields:
        if f in doc:
            val = doc[f]
            if isinstance(val, float):
                val = round(val, 2)
            parts.append(f"{f}={val}")
    if "condition" in doc:
        cond = doc["condition"]
        parts.append(
            f"condition={cond.get('text', cond) if isinstance(cond, dict) else cond}"
        )
    return "  " + " | ".join(parts)


def _build_projection(fields: list[str], col_name: str) -> dict:
    schema     = SCHEMA_REGISTRY.get(col_name, {})
    valid_flds = (
        schema.get("numeric_fields", []) +
        schema.get("categorical_fields", []) +
        schema.get("identifier_fields", [])
    )
    proj = {"timestamp": 1, "_id": 0}
    for f in fields:
        if not valid_flds or f in valid_flds:
            proj[f] = 1
    if "condition" in valid_flds:
        proj["condition"] = 1
    return proj


def _get_bucket(dq: DecomposedQuery) -> tuple[str, str, int]:
    """Returns (bucket_unit, bucket_label, max_buckets) from IR or time_type fallback."""
    if dq.plan:
        gran = dq.plan.execution.granularity
        mb   = dq.plan.execution.max_buckets
        return {
            TimeGranularity.MINUTE: ("minute", "Per-Minute", mb),
            TimeGranularity.HOUR:   ("hour",   "Hourly",     mb),
            TimeGranularity.DAY:    ("day",    "Daily",      mb),
            TimeGranularity.WEEK:   ("week",   "Weekly",     mb),
            TimeGranularity.MONTH:  ("month",  "Monthly",    mb),
        }.get(gran, ("hour", "Hourly", 24))
    tt = dq.time_type
    if tt == "last_hour":                                     return "minute", "Per-Minute", 60
    if tt in ("today", "yesterday", "last_24h"):              return "hour",   "Hourly",     24
    if tt in ("this_week", "last_week", "last_7d"):           return "day",    "Daily",       7
    if tt in ("this_month", "last_30d"):                      return "day",    "Daily",      31
    if tt == "global":                                         return "month",  "Monthly",    60
    return "hour", "Hourly", 24


def _safety_repair_op(op: str, question: str) -> str:
    """Last-resort operation repair at execution time."""
    if op in VALID_OPS:
        return op
    q = question.lower()
    if any(sep in op for sep in ["|", "/", "+", ","]):
        parts      = re.split(r"[|/+,]", op)
        has_oldest = any("oldest" in p.lower() for p in parts)
        has_latest = any("latest" in p.lower() for p in parts)
        if has_oldest and has_latest:
            print(f"  Execution safety-net: '{op}' → DetectRange")
            return "DetectRange"
        for p in parts:
            if p.strip() in VALID_OPS:
                print(f"  Execution safety-net: '{op}' → {p.strip()}")
                return p.strip()
    # Infer from question
    if re.search(r"\boldest\b.{0,30}\blatest\b|\blatest\b.{0,30}\boldest\b", q):
        repaired = "DetectRange"
    elif re.search(r"\boldest\b|\bfirst record\b", q):
        repaired = "DetectOldest"
    elif re.search(r"\blatest\b|\brecent\b", q):
        repaired = "DetectLatest"
    else:
        repaired = "DetectLatest"
    print(f"  Execution safety-net: unknown op '{op}' → {repaired}")
    return repaired


def _execute_on_collection(col: Collection, col_name: str, dq: DecomposedQuery) -> str:
    filt = dq.to_mongo_filter()
    flds = dq.fields
    proj = _build_projection(flds, col_name)

    # ── Execution-time safety net for invalid/compound operations ─────────────
    op = _safety_repair_op(dq.operation, dq.raw_question)

    lines: list[str] = [f"\n Collection: {col_name} | Operation: {op}"]

    try:
        # ── DetectLatest ──────────────────────────────────────────────────────
        if op == "DetectLatest":
            doc = col.find_one(filt, proj, sort=[("timestamp", -1)])
            lines.append(_doc_to_str(doc, flds) if doc else "  (no data found)")

        # ── DetectOldest ──────────────────────────────────────────────────────
        elif op == "DetectOldest":
            doc = col.find_one(filt, proj, sort=[("timestamp", 1)])
            lines.append(_doc_to_str(doc, flds) if doc else "  (no data found)")

        # ── DetectRange ───────────────────────────────────────────────────────
        elif op == "DetectRange":
            oldest = col.find_one(filt, proj, sort=[("timestamp", 1)])
            latest = col.find_one(filt, proj, sort=[("timestamp", -1)])
            if oldest: lines.append("  OLDEST: " + _doc_to_str(oldest, flds).strip())
            if latest: lines.append("  LATEST: " + _doc_to_str(latest, flds).strip())
            if not oldest and not latest:
                lines.append("  (no data found)")

        # ── DetectAverage ─────────────────────────────────────────────────────
        elif op == "DetectAverage":
            group_expr = {f"avg_{f}": {"$avg": f"${f}"} for f in flds}
            result = list(col.aggregate([
                {"$match": filt},
                {"$group": {"_id": None, **group_expr}},
            ]))
            if result:
                parts = [
                    f"{k.replace('avg_', '')}={round(v, 2)}"
                    for k, v in result[0].items()
                    if k != "_id" and v is not None
                ]
                lines.append("  AVG: " + " | ".join(parts))
            else:
                lines.append("  (no data found)")

        # ── DetectMaximum ─────────────────────────────────────────────────────
        elif op == "DetectMaximum":
            f0  = flds[0] if flds else "temperature"
            doc = col.find_one(filt, proj, sort=[(f0, -1)])
            lines.append("  MAX: " + _doc_to_str(doc, flds).strip() if doc else "  (no data found)")

        # ── DetectMinimum ─────────────────────────────────────────────────────
        elif op == "DetectMinimum":
            f0  = flds[0] if flds else "temperature"
            doc = col.find_one(filt, proj, sort=[(f0, 1)])
            lines.append("  MIN: " + _doc_to_str(doc, flds).strip() if doc else "  (no data found)")

        # ── DetectAnomaly ─────────────────────────────────────────────────────
        elif op == "DetectAnomaly":
            f0    = flds[0] if flds else "temperature"
            stats = list(col.aggregate([
                {"$match": filt},
                {"$group": {
                    "_id":  None,
                    "mean": {"$avg": f"${f0}"},
                    "std":  {"$stdDevPop": f"${f0}"},
                }},
            ]))
            if not stats:
                lines.append("  (insufficient data for anomaly detection)")
            else:
                mean, std = stats[0]["mean"] or 0, stats[0]["std"] or 1
                lo, hi    = mean - 2.5 * std, mean + 2.5 * std
                docs = list(col.find(
                    {**filt, f0: {"$not": {"$gte": lo, "$lte": hi}}},
                    proj,
                ).sort("timestamp", -1).limit(dq.limit))
                lines.append(
                    f"  Mean={round(mean,2)} Std={round(std,2)}"
                    f" Threshold=[{round(lo,2)}, {round(hi,2)}]"
                )
                if docs:
                    for d in docs: lines.append(_doc_to_str(d, flds))
                else:
                    lines.append("  (no anomalies detected)")

        # ── DetectTrend ───────────────────────────────────────────────────────
        elif op == "DetectTrend":
            bucket_unit, bucket_label, max_buckets = _get_bucket(dq)
            avg_expr = {f"avg_{f}": {"$avg": f"${f}"} for f in flds if f != "timestamp"}

            group_id: dict  = {
                "year":  {"$year":  "$timestamp"},
                "month": {"$month": "$timestamp"},
            }
            sort_keys: dict = {"_id.year": 1, "_id.month": 1}

            if bucket_unit in ("day", "hour", "minute"):
                group_id["day"]  = {"$dayOfMonth": "$timestamp"}
                sort_keys["_id.day"] = 1
            if bucket_unit in ("hour", "minute"):
                group_id["hour"] = {"$hour": "$timestamp"}
                sort_keys["_id.hour"] = 1
            if bucket_unit == "minute":
                group_id["minute"] = {"$minute": "$timestamp"}
                sort_keys["_id.minute"] = 1

            results = list(col.aggregate([
                {"$match": filt},
                {"$group": {"_id": group_id, **avg_expr, "count": {"$sum": 1}}},
                {"$sort":  sort_keys},
                {"$limit": max_buckets},
            ]))

            lines.append(f"  {bucket_label} trend ({len(results)} buckets):")
            for r in results:
                d = r["_id"]
                if bucket_unit == "minute":
                    lbl = (f"{d['year']}-{d['month']:02d}-{d['day']:02d}"
                           f" {d.get('hour',0):02d}:{d.get('minute',0):02d} UTC")
                elif bucket_unit == "hour":
                    lbl = (f"{d['year']}-{d['month']:02d}-{d['day']:02d}"
                           f" {d.get('hour',0):02d}:00 UTC")
                elif bucket_unit == "day":
                    lbl = f"{d['year']}-{d['month']:02d}-{d['day']:02d}"
                else:
                    lbl = f"{d['year']}-{d['month']:02d}"
                vals = [
                    f"{k.replace('avg_','')}={round(v,2)}" 
                    for k, v in r.items()
                    if k.startswith("avg_") and v is not None
                ]
                lines.append(f"  [{lbl}] {' | '.join(vals)} (n={r.get('count','?')})")
            if not results:
                lines.append("  (no data found)")

        # ── DetectCount ───────────────────────────────────────────────────────
        elif op == "DetectCount":
            q_lower    = dq.raw_question.lower()
            is_unique  = any(kw in q_lower for kw in [
                "unique", "distinct", "berbeda", "unik", "how many different",
            ])
            is_groupby = any(kw in q_lower for kw in [
                "most record", "most records", "most data", "highest count",
                "per device", "per serial", "by device", "by serial",
                "which device", "which serial", "terbanyak", "paling banyak",
                "breakdown", "per unit", "each device", "each serial",
            ])
            if   any(kw in q_lower for kw in ["serial", "serial number", "nomor seri"]):
                group_field = "serial_number"
            elif any(kw in q_lower for kw in ["device", "unit", "perangkat"]):
                group_field = "device_id"
            elif dq.fields and dq.fields[0] not in (
                "temperature", "humidity", "co2", "dust", "voc", "ozone"
            ):
                group_field = dq.fields[0]
            else:
                group_field = None

            if is_groupby and group_field:
                results = list(col.aggregate([
                    {"$match": filt},
                    {"$group": {"_id": f"${group_field}", "count": {"$sum": 1}}},
                    {"$sort":  {"count": -1}},
                    {"$limit": max(dq.limit, 10)},
                ]))
                lines.append(f"  Top {len(results)} by '{group_field}':")
                for rank, r in enumerate(results, 1):
                    lines.append(f"  #{rank:>2} {r['_id']} → {r['count']:,} records")
                if not results:
                    lines.append(f"  (no data or field '{group_field}' not found)")

            elif is_unique and group_field:
                vals = col.distinct(group_field, filt)
                lines.append(f"  Unique '{group_field}' values: {len(vals)}")
                if len(vals) <= 30:
                    lines.append(f"  Values: {vals}")

            else:
                count = col.count_documents(filt)
                lines.append(f"  Total records: {count:,}")

        # ── DetectCompare ─────────────────────────────────────────────────────
        elif op == "DetectCompare":
            doc = col.find_one(filt, proj, sort=[("timestamp", -1)])
            lines.append(_doc_to_str(doc, flds) if doc else "  (no data found)")

        else:
            lines.append(f"  Unhandled operation: {op}")

    except Exception as e:
        lines.append(f"  Query error: {e}")

    return "\n".join(lines)


def run_stage2(
    dq:          DecomposedQuery,
    klaen_col:   Collection,
    company_col: Collection,
    weather_col: Collection,
) -> str:
    t0 = time.monotonic()
    collection_map = {
        "plalion_klaen_sensor":   klaen_col,
        "plalion_company_sensor": company_col,
        "lighting_weatherapi":    weather_col,
    }
    results: list[str] = []
    for col_name in dq.collections:
        col = collection_map.get(col_name)
        if col is None:
            results.append(f"\n Unknown collection: {col_name}")
            continue
        results.append(_execute_on_collection(col, col_name, dq))

    if dq.plan:
        dq.plan.stage2_latency_ms = (time.monotonic() - t0) * 1000

    combined = "\n".join(results)
    return combined if combined.strip() else "No data retrieved from the database."
