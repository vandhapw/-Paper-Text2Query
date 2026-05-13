"""
Formal Intermediate Representation (IR) for TAG query planning.
Separates semantic intent (what the user means) from execution intent (how to query MongoDB).
"""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
from enum import Enum


class AnalyticOp(str, Enum):
    LATEST  = "DetectLatest"
    OLDEST  = "DetectOldest"
    RANGE   = "DetectRange"
    AVERAGE = "DetectAverage"
    MAXIMUM = "DetectMaximum"
    MINIMUM = "DetectMinimum"
    ANOMALY = "DetectAnomaly"
    TREND   = "DetectTrend"
    COUNT   = "DetectCount"
    COMPARE = "DetectCompare"


class TimeGranularity(str, Enum):
    SECOND = "second"
    MINUTE = "minute"
    HOUR   = "hour"
    DAY    = "day"
    WEEK   = "week"
    MONTH  = "month"


class DataSource(str, Enum):
    KLAEN   = "plalion_klaen_sensor"
    COMPANY = "plalion_company_sensor"
    WEATHER = "lighting_weatherapi"


SCHEMA_REGISTRY: dict[str, dict] = {
    DataSource.KLAEN: {
        "numeric_fields":      ["temperature", "humidity", "co2", "dust", "voc", "ozone"],
        "categorical_fields":  [],
        "identifier_fields":   ["serial_number"],
        "timestamp_field":     "timestamp",
        "source_type":         "indoor",
        "location_label":      "Klaen (Indoor)",
        "sampling_interval_sec": 10,
    },
    DataSource.COMPANY: {
        "numeric_fields":      ["temperature", "humidity", "co2", "dust", "voc", "ozone"],
        "categorical_fields":  [],
        "identifier_fields":   ["serial_number", "device_id"],
        "timestamp_field":     "timestamp",
        "source_type":         "indoor",
        "location_label":      "Company (Office)",
        "sampling_interval_sec": 5,
    },
    DataSource.WEATHER: {
        "numeric_fields":      [
            "temp_c", "feelslike_c", "humidity", "wind_kph", "precip_mm",
            "aqi", "pm2_5", "pm10", "o3", "co", "no2", "so2", "pressure_mb",
        ],
        "categorical_fields":  ["condition"],
        "identifier_fields":   [],
        "timestamp_field":     "timestamp",
        "source_type":         "outdoor",
        "location_label":      "WeatherAPI (Outdoor)",
        "sampling_interval_sec": 1800,
    },
}

TIME_TO_GRANULARITY: dict[str, TimeGranularity] = {
    "last_hour":  TimeGranularity.MINUTE,
    "today":      TimeGranularity.HOUR,
    "yesterday":  TimeGranularity.HOUR,
    "last_24h":   TimeGranularity.HOUR,
    "this_week":  TimeGranularity.DAY,
    "last_week":  TimeGranularity.DAY,
    "last_7d":    TimeGranularity.DAY,
    "this_month": TimeGranularity.DAY,
    "last_30d":   TimeGranularity.DAY,
    "global":     TimeGranularity.MONTH,
}

TIME_TO_MAX_BUCKETS: dict[str, int] = {
    "last_hour":  60,
    "today":      24,
    "yesterday":  24,
    "last_24h":   24,
    "this_week":  7,
    "last_week":  7,
    "last_7d":    7,
    "this_month": 31,
    "last_30d":   30,
    "global":     60,
}

VALID_OPS = {
    "DetectLatest", "DetectOldest", "DetectRange", "DetectAverage",
    "DetectMaximum", "DetectMinimum", "DetectAnomaly", "DetectTrend",
    "DetectCount", "DetectCompare",
}


@dataclass
class SemanticIntent:
    raw_question:    str
    intent_type:     str
    subject_fields:  list[str]
    subject_sources: list[str]
    time_expression: str
    is_multi_source: bool  = False
    is_multi_field:  bool  = False
    requires_viz:    bool  = True
    confidence:      float = 1.0


@dataclass
class ExecutionIntent:
    collections:  list[str]
    fields:       list[str]
    operation:    str
    mongo_filter: dict
    aggregation:  list
    time_type:    str
    time_start:   Optional[datetime]
    time_end:     Optional[datetime]
    granularity:  TimeGranularity
    max_buckets:  int
    limit:        int
    group_field:  Optional[str] = None


@dataclass
class ValidationResult:
    is_valid:     bool
    confidence:   float
    warnings:     list[str] = field(default_factory=list)
    errors:       list[str] = field(default_factory=list)
    repaired:     bool      = False
    repair_notes: list[str] = field(default_factory=list)


@dataclass
class QueryPlan:
    semantic:              SemanticIntent
    execution:             ExecutionIntent
    validation:            ValidationResult
    plan_source:           str   = "llm"
    stage1_latency_ms:     float = 0.0
    stage2_latency_ms:     float = 0.0
    stage3_latency_ms:     float = 0.0
    stage4_latency_ms:     float = 0.0
    total_records_retrieved: int = 0

    def to_debug_str(self) -> str:
        lines = [
            f"  ┌─ QUERY PLAN {'─'*40}",
            f"  │ Source      : {self.plan_source}",
            f"  │ Confidence  : {self.validation.confidence:.0%}",
            f"  ├─ SEMANTIC INTENT",
            f"  │ Intent Type : {self.semantic.intent_type}",
            f"  │ Fields      : {self.semantic.subject_fields}",
            f"  │ Sources     : {self.semantic.subject_sources}",
            f"  │ Time Expr   : {self.semantic.time_expression}",
            f"  │ Multi-src   : {self.semantic.is_multi_source}",
            f"  │ Multi-field : {self.semantic.is_multi_field}",
            f"  ├─ EXECUTION INTENT",
            f"  │ Collections : {self.execution.collections}",
            f"  │ Fields      : {self.execution.fields}",
            f"  │ Operation   : {self.execution.operation}",
            f"  │ Time Type   : {self.execution.time_type}",
            f"  │ Time Start  : {self.execution.time_start}",
            f"  │ Time End    : {self.execution.time_end}",
            f"  │ Granularity : {self.execution.granularity.value}",
            f"  │ Max Buckets : {self.execution.max_buckets}",
            f"  │ Mongo Filter: {self.execution.mongo_filter}",
        ]
        if self.validation.warnings:
            lines.append(f"  ├─ ⚠️  WARNINGS : {'; '.join(self.validation.warnings)}")
        if self.validation.errors:
            lines.append(f"  ├─ ❌ ERRORS   : {'; '.join(self.validation.errors)}")
        if self.validation.repaired:
            lines.append(f"  ├─ 🔧 REPAIRED : {'; '.join(self.validation.repair_notes)}")
        lines.append(f"  └─{'─'*44}")
        return "\n".join(lines)
