"""Stage 1 - Query Synthesis + Validation + Self-Repair
Produces a formal QueryPlan (IR). Includes compound-operation repair.
"""
from __future__ import annotations
import os, re, json, time, sys, io
if __name__ == '__main__':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass, field
from typing import Optional
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

from query_plan import (
    QueryPlan, SemanticIntent, ExecutionIntent, ValidationResult,
    TimeGranularity, DataSource, SCHEMA_REGISTRY, VALID_OPS,
    TIME_TO_GRANULARITY, TIME_TO_MAX_BUCKETS,
)

load_dotenv(override=True)

# ── Backward-compatible DecomposedQuery ───────────────────────────────────────

@dataclass
class DecomposedQuery:
    collections:  list[str]
    fields:       list[str]
    operation:    str
    time_type:    str
    time_start:   Optional[datetime]
    time_end:     Optional[datetime]
    limit:        int  = 10
    raw_question: str  = ""
    mongo_filter: dict = field(default_factory=dict)
    aggregation:  list = field(default_factory=list)
    plan:         Optional[QueryPlan] = field(default=None, repr=False)

    def to_mongo_filter(self) -> dict:
        if self.time_type == "global":
            return {}
        if self.time_start and self.time_end:
            return {"timestamp": {"$gte": self.time_start, "$lte": self.time_end}}
        if self.time_start:
            return {"timestamp": {"$gte": self.time_start}}
        return {}


# ── Constants ──────────────────────────────────────────────────────────────────

COLLECTION_ALIASES = {
    "klaen":   DataSource.KLAEN,   "indoor":  DataSource.KLAEN,
    "company": DataSource.COMPANY, "office":  DataSource.COMPANY,
    "weather": DataSource.WEATHER, "outdoor": DataSource.WEATHER,
    "outside": DataSource.WEATHER, "cuaca":   DataSource.WEATHER,
}

FIELD_ALIASES = {
    "temperature":   ["temp", "temperature", "suhu", "panas"],
    "humidity":      ["humidity", "humid", "kelembaban", "lembab"],
    "co2":           ["co2", "carbon dioxide", "karbondioksida"],
    "dust":          ["dust", "particulate matter", "pm"],
    "voc":           ["voc", "volatile organic compounds"],
    "ozone":         ["ozone", "o3"],
    "aqi":           ["aqi", "air quality", "kualitas udara"],
    "pm2_5":         ["pm2.5", "pm2_5", "fine dust"],
    "pm10":          ["pm10"],
    "wind_kph":      ["wind", "angin"],
    "precip_mm":     ["rain", "precip", "hujan"],
    "condition":     ["condition", "cuaca", "weather condition"],
    "serial_number": ["serial", "serial number", "nomor seri"],
    "device_id":     ["device id", "device", "perangkat"],
    "pressure_mb":   ["pressure", "pressure_mb", "hpa", "barometric"],
    "feelslike_c":   ["feels like", "feelslike", "feels like temperature", "apparent temperature"],
}

COLLECTION_FIELD_ALIASES = {
    DataSource.WEATHER: {
        "temperature": "temp_c",
        "ozone": "o3",
    },
}


def _schema_fields(collection: str) -> set[str]:
    schema = SCHEMA_REGISTRY.get(collection, {})
    return set(
        schema.get("numeric_fields", [])
        + schema.get("categorical_fields", [])
        + schema.get("identifier_fields", [])
    )


def _default_field_for_collections(collections: list[str]) -> str:
    if collections and all(c == DataSource.WEATHER for c in collections):
        return "temp_c"
    return "temperature"


def _normalize_fields_for_collections(fields: list[str], collections: list[str]) -> list[str]:
    normalized: list[str] = []
    for field in fields:
        if field == "timestamp":
            continue

        kept_original = False
        for collection in collections:
            valid_fields = _schema_fields(collection)
            if field in valid_fields:
                if field not in normalized:
                    normalized.append(field)
                kept_original = True
                continue

            mapped = COLLECTION_FIELD_ALIASES.get(collection, {}).get(field)
            if mapped and mapped in valid_fields and mapped not in normalized:
                normalized.append(mapped)

        if not kept_original and not any(
            COLLECTION_FIELD_ALIASES.get(collection, {}).get(field) in normalized
            for collection in collections
        ):
            normalized.append(field)

    return normalized

# DetectRange must be checked BEFORE DetectOldest/DetectLatest
OPERATION_PATTERNS = {
    "DetectRange":   [r"\boldest.{0,25}latest\b", r"\blatest.{0,25}oldest\b",
                      r"\bfirst.{0,25}last\b",    r"\boldest.{0,25}newest\b"],
    "DetectOldest":  [r"\boldest\b", r"\bpertama\b", r"\bfirst record\b", r"\bawal\b"],
    "DetectLatest":  [r"\blatest\b", r"\bterbaru\b", r"\bcurrent\b", r"\bnow\b",
                      r"\bsekarang\b", r"\brecent\b"],
    "DetectAverage": [r"\baverage\b", r"\bmean\b", r"\brata.rata\b", r"\bavg\b"],
    "DetectMaximum": [r"\bmaximum\b", r"\bmax\b", r"\bhighest\b", r"\btertinggi\b"],
    "DetectMinimum": [r"\bminimum\b", r"\bmin\b", r"\blowest\b",  r"\bterendah\b"],
    "DetectAnomaly": [r"\banomaly\b", r"\babnormal\b", r"\bspike\b", r"\banomal\b"],
    "DetectTrend":   [r"\btrend\b", r"\bover time\b", r"\bhistory\b", r"\bhistorik\b"],
    "DetectCount":   [r"\bhow many\b", r"\bcount\b", r"\bjumlah\b", r"\bberapa\b"],
    "DetectCompare": [r"\bcompare\b", r"\bdifference\b", r"\bbanding\b",
                      r"\bvs\b", r"\bversus\b"],
}

TIME_PATTERNS = {
    "last_hour":  r"\blast hour\b|\bjam terakhir\b",
    "today":      r"\btoday\b|\bhari ini\b",
    "yesterday":  r"\byesterday\b|\bkemarin\b",
    "this_week":  r"\bthis week\b|\bminggu ini\b",
    "last_week":  r"\blast week\b|\bminggu lalu\b",
    "this_month": r"\bthis month\b|\bbulan ini\b",
    "last_24h":   r"\blast 24 hours?\b|\b24 jam\b",
    "last_7d":    r"\blast 7 days?\b|\b7 hari\b",
    "last_30d":   r"\blast 30 days?\b|\b30 hari\b",
    "global":     r"\ball time\b|\ball data\b|\bsemua data\b|\bseluruh\b|\bhistorical\b",
}

OPERATION_TO_INTENT = {
    "DetectLatest":  "point_lookup",
    "DetectOldest":  "point_lookup",
    "DetectRange":   "boundary_lookup",
    "DetectAverage": "aggregation",
    "DetectMaximum": "aggregation",
    "DetectMinimum": "aggregation",
    "DetectAnomaly": "anomaly_detection",
    "DetectTrend":   "trend_analysis",
    "DetectCount":   "counting",
    "DetectCompare": "comparison",
}

SCHEMA_CONTEXT = """
Collections and their fields:
1. plalion_klaen_sensor   → temperature, humidity, co2, dust, voc, ozone, timestamp
2. plalion_company_sensor → temperature, humidity, co2, dust, voc, ozone, serial_number, timestamp
3. lighting_weatherapi    → temp_c, feelslike_c, humidity, wind_kph, precip_mm,
                            condition, aqi, pm2_5, pm10, o3, co, no2, so2, timestamp
"""

SYSTEM_PROMPT = f"""You are a MongoDB query planner for an IoT sensor database.
{SCHEMA_CONTEXT}

You MUST always respond with a valid JSON object. Never return empty.
Respond ONLY with this exact JSON structure (no markdown, no explanation):
{{
  "collections": ["<collection_name>"],
  "fields": ["<field_name>"],
  "operation": "<single operation — see VALID OPERATIONS below>",
  "time_type": "<last_hour|today|yesterday|this_week|last_week|this_month|last_24h|last_7d|last_30d|global>",
  "limit": <integer 1-100>,
  "intent_type": "<point_lookup|aggregation|trend_analysis|comparison|anomaly_detection|counting|boundary_lookup>"
}}

VALID OPERATIONS (choose exactly ONE string):
  DetectLatest  → most recent single record
  DetectOldest  → oldest single record
  DetectRange   → BOTH oldest AND latest record (use for "oldest and latest", "first and last")
  DetectAverage → average value over time range
  DetectMaximum → record with highest field value
  DetectMinimum → record with lowest field value
  DetectAnomaly → statistical outlier detection
  DetectTrend   → time-bucketed aggregation series
  DetectCount   → count records or distinct values
  DetectCompare → compare latest values across collections

CRITICAL RULES:
- "operation" field must be a SINGLE string from VALID OPERATIONS above.
- NEVER combine operations using |, /, +, comma, or any separator.
- "oldest and latest" / "first and last" / "oldest and newest" → ALWAYS DetectRange.
- Compare questions with both indoor sensors → include both klaen and company collections.
- Weather/outdoor questions → use lighting_weatherapi.
- Weather/outdoor temperature questions -> use field "temp_c" (not "temperature").
- Weather/outdoor ozone questions -> use field "o3" (not "ozone").
- "global" time_type = ALL historical data, no date filter.
- "daily trend" → time_type=today, limit=24.
- "weekly trend" → time_type=last_7d, limit=7.
- "monthly trend" → time_type=this_month, limit=31.
- "all time trend" → time_type=global, limit=60.
- Never include "timestamp" in the fields list.
- No time keyword in question → default time_type=global.
- "unique/distinct/most records by device/serial" → DetectCount + fields=["serial_number"] + time_type=global.
"""


# ── Time Resolver ──────────────────────────────────────────────────────────────

def _resolve_time(time_type: str) -> tuple[Optional[datetime], Optional[datetime]]:
    # Allow a fixed reference date for benchmark reproducibility.
    # If REFERENCE_DATE is set (format: YYYY-MM-DD), use it instead of "now".
    # This ensures time-scoped queries hit data that actually exists in the DB.
    _ref = os.getenv("REFERENCE_DATE", "").strip()
    if _ref:
        try:
            # When using a fixed reference date, set "now" to end of that day
            # so "today" queries cover the full day (00:00–23:59:59)
            ref_dt = datetime.strptime(_ref, "%Y-%m-%d").replace(tzinfo=timezone.utc)
            now = ref_dt.replace(hour=23, minute=59, second=59)
        except ValueError:
            now = datetime.now(timezone.utc)
    else:
        now = datetime.now(timezone.utc)
    today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    mapping = {
        "last_hour":  (now - timedelta(hours=1), now),
        "today":      (today_start, now),
        "yesterday":  (today_start - timedelta(days=1), today_start),
        "this_week":  (today_start - timedelta(days=now.weekday()), now),
        "last_week":  (today_start - timedelta(days=now.weekday() + 7),
                       today_start - timedelta(days=now.weekday())),
        "this_month": (today_start.replace(day=1), now),
        "last_24h":   (now - timedelta(hours=24), now),
        "last_7d":    (now - timedelta(days=7), now),
        "last_30d":   (now - timedelta(days=30), now),
        "global":     (None, None),
    }
    return mapping.get(time_type, (now - timedelta(hours=24), now))


# ── Compound Operation Repairer ────────────────────────────────────────────────

def _repair_operation(operation: str, question: str) -> tuple[str, Optional[str]]:
    """
    Detect and repair invalid/compound operation strings.
    Returns (repaired_operation, repair_note) or (original, None) if already valid.
    """
    if operation in VALID_OPS:
        return operation, None

    q = question.lower()
    original = operation

    # Case 1: pipe/slash/plus-combined ops (e.g. "DetectOldest|DetectLatest")
    if any(sep in operation for sep in ["|", "/", "+", ","]):
        parts = re.split(r"[|/+,]", operation)
        parts = [p.strip() for p in parts if p.strip()]
        has_oldest = any("oldest" in p.lower() for p in parts)
        has_latest = any("latest" in p.lower() for p in parts)
        if has_oldest and has_latest:
            return "DetectRange", f"Compound op '{original}' → DetectRange"
        # Pick first valid part
        for p in parts:
            if p in VALID_OPS:
                return p, f"Compound op '{original}' → {p}"

    # Case 2: infer from question text
    if re.search(r"\boldest\b.{0,30}\blatest\b|\blatest\b.{0,30}\boldest\b"
                 r"|\bfirst\b.{0,30}\blast\b|\boldest\b.{0,30}\bnewest\b", q):
        return "DetectRange", f"Unknown op '{original}' → DetectRange (from question)"
    if re.search(r"\boldest\b|\bfirst record\b|\bawal\b|\bpertama\b", q):
        return "DetectOldest", f"Unknown op '{original}' → DetectOldest (from question)"
    if re.search(r"\blatest\b|\brecent\b|\bcurrent\b|\bterbaru\b", q):
        return "DetectLatest", f"Unknown op '{original}' → DetectLatest (from question)"
    if re.search(r"\btrend\b|\bover time\b", q):
        return "DetectTrend", f"Unknown op '{original}' → DetectTrend (from question)"
    if re.search(r"\bcompare\b|\bvs\b|\bversus\b", q):
        return "DetectCompare", f"Unknown op '{original}' → DetectCompare (from question)"

    return "DetectLatest", f"Unknown op '{original}' → defaulted to DetectLatest"


# ── Schema Validator + Self-Repair ─────────────────────────────────────────────

def _validate_and_repair(
    collections: list[str],
    fields:      list[str],
    operation:   str,
    time_type:   str,
    question:    str,
) -> tuple[list[str], list[str], str, str, ValidationResult]:
    warnings, errors, repair_notes = [], [], []
    repaired = False
    q = question.lower()

    # ── 0. Repair compound/invalid operation ──────────────────────────────────
    operation, repair_note = _repair_operation(operation, question)
    if repair_note:
        repair_notes.append(repair_note)
        repaired = True
        if "Unknown op" in repair_note:
            errors.append(repair_note)

    # ── 1. Collection validation ───────────────────────────────────────────────
    valid_collections = list(SCHEMA_REGISTRY.keys())
    invalid_cols = [c for c in collections if c not in valid_collections]
    if invalid_cols:
        errors.append(f"Unknown collections: {invalid_cols}")
        collections = [c for c in collections if c in valid_collections]
        if not collections:
            collections = [DataSource.KLAEN]
            repair_notes.append("Defaulted collections to plalion_klaen_sensor")
            repaired = True

    fields = _normalize_fields_for_collections(fields, collections)
    if not fields:
        fields = [_default_field_for_collections(collections)]
        repair_notes.append(f"Defaulted fields to {fields}")
        repaired = True

    # ── 2. Field validation against collection schemas ─────────────────────────
    all_valid_fields: set[str] = set()
    for col in collections:
        schema = SCHEMA_REGISTRY.get(col, {})
        all_valid_fields.update(schema.get("numeric_fields", []))
        all_valid_fields.update(schema.get("categorical_fields", []))
        all_valid_fields.update(schema.get("identifier_fields", []))

    invalid_fields = [f for f in fields if f not in all_valid_fields and f != "timestamp"]
    if invalid_fields:
        warnings.append(f"Fields not in schema: {invalid_fields}")
        fields = [f for f in fields if f in all_valid_fields]
        if not fields:
            fields = [_default_field_for_collections(collections)]
            repair_notes.append(f"Defaulted fields to {fields}")
            repaired = True

    # ── 3. Multi-collection grounding for compare queries ─────────────────────
    if operation == "DetectCompare" and len(collections) < 2:
        has_indoor = any(k in q for k in ["indoor", "klaen", "company", "office"])
        has_outdoor = any(k in q for k in ["outdoor", "outside", "weather"])
        if has_indoor and has_outdoor:
            collections = [DataSource.KLAEN, DataSource.WEATHER]
        elif has_indoor:
            collections = [DataSource.KLAEN, DataSource.COMPANY]
        elif has_outdoor:
            collections = [DataSource.WEATHER, DataSource.KLAEN]
        else:
            collections = [DataSource.KLAEN, DataSource.COMPANY, DataSource.WEATHER]
        fields = _normalize_fields_for_collections(fields, collections)
        if not fields:
            fields = [_default_field_for_collections(collections)]
        repair_notes.append(f"Auto-expanded collections for comparison: {collections}")
        repaired = True
        warnings.append("Compare op requires multi-collection; auto-expanded")

    # ── 4. Time-aware operator warnings ───────────────────────────────────────
    if operation == "DetectTrend" and time_type == "global":
        warnings.append("DetectTrend with global scope may be slow on large collections")
    if operation in ("DetectLatest", "DetectOldest") and time_type not in (
        "global", "today", "last_24h", "last_hour"
    ):
        warnings.append(f"Point lookup with '{time_type}' may return no data if collection is sparse")

    # ── 5. Anomaly single-field enforcement ───────────────────────────────────
    if operation == "DetectAnomaly" and len(fields) > 1:
        warnings.append("DetectAnomaly uses only the first field; extra fields ignored")

    # ── 6. Confidence scoring ──────────────────────────────────────────────────
    penalty = 0.0
    penalty += 0.15 * len(errors)
    penalty += 0.05 * len(warnings)
    penalty += 0.10 if repaired else 0.0
    confidence = max(0.0, 1.0 - penalty)

    vr = ValidationResult(
        is_valid     = len(errors) == 0,
        confidence   = confidence,
        warnings     = warnings,
        errors       = errors,
        repaired     = repaired,
        repair_notes = repair_notes,
    )
    return collections, fields, operation, time_type, vr


# ── Regex Fallback ─────────────────────────────────────────────────────────────

def _regex_decompose(question: str) -> tuple[dict, float]:
    q = question.lower()

    collections = []
    for alias, col in COLLECTION_ALIASES.items():
        if alias in q and col not in collections:
            collections.append(col)
    if not collections:
        collections = [DataSource.KLAEN, DataSource.COMPANY, DataSource.WEATHER]

    detected_fields: list[str] = []
    for canonical, aliases in FIELD_ALIASES.items():
        if any(a in q for a in aliases):
            detected_fields.append(canonical)
    detected_fields = [f for f in detected_fields if f != "timestamp"]
    if not detected_fields:
        detected_fields = ["temperature"]
    detected_fields = _normalize_fields_for_collections(detected_fields, collections)

    # Check DetectRange first (before individual oldest/latest)
    operation = "DetectLatest"
    for op, patterns in OPERATION_PATTERNS.items():
        for pat in patterns:
            if re.search(pat, q):
                operation = op
                break
        if operation != "DetectLatest":
            break

    time_type = "global"
    for tt, pat in TIME_PATTERNS.items():
        if re.search(pat, q):
            time_type = tt
            break

    return {
        "collections": collections,
        "fields":      detected_fields,
        "operation":   operation,
        "time_type":   time_type,
        "limit":       10,
        "intent_type": OPERATION_TO_INTENT.get(operation, "point_lookup"),
    }, 0.70


# ── Main Decomposer ────────────────────────────────────────────────────────────

class QuestionDecomposer:
    def __init__(self, groq_llm: ChatOpenAI, max_retries: int = 3):
        self.llm         = groq_llm
        self.max_retries = max_retries

    def decompose(self, question: str) -> DecomposedQuery:
        t0 = time.monotonic()
        parsed, plan_source, base_confidence = self._llm_parse(question)
        latency_ms = (time.monotonic() - t0) * 1000

        collections  = parsed.get("collections", [DataSource.KLAEN])
        fields       = [f for f in parsed.get("fields", ["temperature"]) if f != "timestamp"]
        operation    = parsed.get("operation", "DetectLatest")
        time_type    = parsed.get("time_type", "global")
        limit        = int(parsed.get("limit", 10))
        intent_type  = parsed.get("intent_type", OPERATION_TO_INTENT.get(operation, "point_lookup"))

        # Validate + self-repair (including compound op repair)
        collections, fields, operation, time_type, vr = _validate_and_repair(
            collections, fields, operation, time_type, question
        )
        if vr.repaired:
            plan_source = "repaired"

        vr.confidence = min(vr.confidence, base_confidence)

        time_start, time_end = _resolve_time(time_type)
        mongo_filter = {} if time_type == "global" else (
            {"timestamp": {"$gte": time_start, "$lte": time_end}}
            if time_start and time_end else {}
        )

        granularity = TIME_TO_GRANULARITY.get(time_type, TimeGranularity.HOUR)
        max_buckets = TIME_TO_MAX_BUCKETS.get(time_type, 24)

        semantic = SemanticIntent(
            raw_question    = question,
            intent_type     = intent_type,
            subject_fields  = fields,
            subject_sources = [
                SCHEMA_REGISTRY.get(c, {}).get("source_type", "unknown")
                for c in collections
            ],
            time_expression = time_type,
            is_multi_source = len(collections) > 1,
            is_multi_field  = len(fields) > 1,
            requires_viz    = operation in {
                "DetectTrend", "DetectCompare", "DetectAverage",
                "DetectMaximum", "DetectMinimum", "DetectAnomaly", "DetectCount",
            },
            confidence      = vr.confidence,
        )

        execution = ExecutionIntent(
            collections  = collections,
            fields       = fields,
            operation    = operation,
            mongo_filter = mongo_filter,
            aggregation  = [],
            time_type    = time_type,
            time_start   = time_start,
            time_end     = time_end,
            granularity  = granularity,
            max_buckets  = max_buckets,
            limit        = limit,
            group_field  = None,
        )

        plan = QueryPlan(
            semantic          = semantic,
            execution         = execution,
            validation        = vr,
            plan_source       = plan_source,
            stage1_latency_ms = latency_ms,
        )

        dq = DecomposedQuery(
            collections  = collections,
            fields       = fields,
            operation    = operation,
            time_type    = time_type,
            time_start   = time_start,
            time_end     = time_end,
            limit        = limit,
            raw_question = question,
            mongo_filter = mongo_filter,
            plan         = plan,
        )
        return dq

    def _llm_parse(self, question: str) -> tuple[dict, str, float]:
        last_error = None
        for attempt in range(1, self.max_retries + 1):
            try:
                raw = self.llm.invoke([
                    ("system", SYSTEM_PROMPT),
                    ("human",  question),
                ]).content.strip()
                raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
                raw = re.sub(r"^```(?:json)?|```$",  "", raw, flags=re.MULTILINE).strip()
                if not raw:
                    raise ValueError("Empty LLM response after stripping")
                return json.loads(raw), "llm", 0.95
            except Exception as e:
                last_error = e
                if attempt < self.max_retries:
                    wait = attempt * 1.5
                    print(f"  LLM attempt {attempt}/{self.max_retries} failed ({e})"
                          f" — retrying in {wait}s...")
                    time.sleep(wait)

        print(f"  LLM failed ({last_error}) → regex fallback")
        parsed, conf = _regex_decompose(question)
        return parsed, "regex", conf


def format_decomposition(dq: DecomposedQuery) -> str:
    if dq.plan:
        return dq.plan.to_debug_str()
    return "\n".join([
        f"  Collections : {dq.collections}",
        f"  Fields      : {dq.fields}",
        f"  Operation   : {dq.operation}",
        f"  Time Type   : {dq.time_type}",
        f"  Mongo Filter: {dq.mongo_filter}",
    ])
