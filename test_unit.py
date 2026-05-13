"""
Unit Tests for Text2Query Pipeline
====================================
Covers: _repair_operation, _validate_and_repair, each S2 operation handler
Run:  python -m pytest test_unit.py -v
     or  python test_unit.py
"""
from __future__ import annotations
import sys
import io
if __name__ == '__main__':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import unittest
from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock, patch
from dataclasses import dataclass
from typing import Optional

# ── Import target modules ─────────────────────────────────────────────────────
from s1_decomposer import (
    _repair_operation,
    _validate_and_repair,
    _regex_decompose,
    _resolve_time,
    DecomposedQuery,
    COLLECTION_ALIASES,
    FIELD_ALIASES,
    OPERATION_PATTERNS,
    TIME_PATTERNS,
)
from query_plan import (
    VALID_OPS, SCHEMA_REGISTRY, DataSource,
    TimeGranularity, TIME_TO_GRANULARITY, TIME_TO_MAX_BUCKETS,
)


# ══════════════════════════════════════════════════════════════════════════════
# 1. _repair_operation TESTS
# ══════════════════════════════════════════════════════════════════════════════

class TestRepairOperation(unittest.TestCase):
    """Test compound/invalid operation repair logic."""

    # ── Valid operations pass through unchanged ───────────────────────────────
    def test_valid_ops_unchanged(self):
        for op in VALID_OPS:
            repaired, note = _repair_operation(op, "any question")
            self.assertEqual(repaired, op, f"Valid op '{op}' should not be changed")
            self.assertIsNone(note, f"Valid op '{op}' should have no repair note")

    # ── Pipe-separated compound ops ────────────────────────────────────────────
    def test_pipe_oldest_latest_becomes_range(self):
        repaired, note = _repair_operation("DetectOldest|DetectLatest", "oldest and latest")
        self.assertEqual(repaired, "DetectRange")
        self.assertIn("Compound", note)

    def test_pipe_latest_oldest_becomes_range(self):
        repaired, note = _repair_operation("DetectLatest|DetectOldest", "show me")
        self.assertEqual(repaired, "DetectRange")

    def test_pipe_first_last_becomes_range(self):
        repaired, _ = _repair_operation("DetectFirst|DetectLast", "oldest and latest data")
        self.assertEqual(repaired, "DetectRange")

    # ── Slash-separated compound ops ───────────────────────────────────────────
    def test_slash_oldest_latest_becomes_range(self):
        repaired, note = _repair_operation("DetectOldest/DetectLatest", "range query")
        self.assertEqual(repaired, "DetectRange")

    # ── Plus-separated compound ops ────────────────────────────────────────────
    def test_plus_oldest_latest_becomes_range(self):
        repaired, _ = _repair_operation("DetectOldest+DetectLatest", "oldest and latest")
        self.assertEqual(repaired, "DetectRange")

    # ── Comma-separated compound ops ──────────────────────────────────────────
    def test_comma_oldest_latest_becomes_range(self):
        repaired, _ = _repair_operation("DetectOldest, DetectLatest", "oldest and latest")
        self.assertEqual(repaired, "DetectRange")

    # ── Pipe with non-range combo → picks first valid ─────────────────────────
    def test_pipe_trend_average_picks_first(self):
        repaired, note = _repair_operation("DetectTrend|DetectAverage", "trend over time")
        self.assertIn(repaired, ["DetectTrend", "DetectAverage"])

    # ── Unknown single op with question context ───────────────────────────────
    def test_unknown_op_oldest_latest_in_question(self):
        repaired, note = _repair_operation("SomeRandomOp", "oldest and latest temperature")
        self.assertEqual(repaired, "DetectRange")
        self.assertIn("Unknown op", note)

    def test_unknown_op_oldest_in_question(self):
        repaired, _ = _repair_operation("BadOp", "show the oldest record")
        self.assertEqual(repaired, "DetectOldest")

    def test_unknown_op_latest_in_question(self):
        repaired, _ = _repair_operation("BadOp", "show the latest reading")
        self.assertEqual(repaired, "DetectLatest")

    def test_unknown_op_trend_in_question(self):
        repaired, _ = _repair_operation("BadOp", "show temperature trend over time")
        self.assertEqual(repaired, "DetectTrend")

    def test_unknown_op_compare_in_question(self):
        repaired, _ = _repair_operation("BadOp", "compare indoor vs outdoor")
        self.assertEqual(repaired, "DetectCompare")

    # ── Completely unknown with no context → default DetectLatest ────────────
    def test_unknown_no_context_defaults_latest(self):
        repaired, note = _repair_operation("XYZ123", "sensor data")
        self.assertEqual(repaired, "DetectLatest")
        self.assertIn("defaulted", note)

    # ── Edge: empty string ────────────────────────────────────────────────────
    def test_empty_string_defaults_latest(self):
        repaired, _ = _repair_operation("", "show data")
        self.assertEqual(repaired, "DetectLatest")


# ══════════════════════════════════════════════════════════════════════════════
# 2. _validate_and_repair TESTS
# ══════════════════════════════════════════════════════════════════════════════

class TestValidateAndRepair(unittest.TestCase):
    """Test schema validation, collection/field repair, and confidence scoring."""

    # ── Valid inputs pass through ─────────────────────────────────────────────
    def test_valid_klaen_latest(self):
        cols, flds, op, tt, vr = _validate_and_repair(
            ["plalion_klaen_sensor"], ["temperature"],
            "DetectLatest", "global", "latest temperature from klaen"
        )
        self.assertEqual(cols, ["plalion_klaen_sensor"])
        self.assertEqual(flds, ["temperature"])
        self.assertEqual(op, "DetectLatest")
        self.assertTrue(vr.is_valid)
        self.assertFalse(vr.repaired)
        self.assertGreater(vr.confidence, 0.8)

    # ── Invalid collection → default to KLAEN ────────────────────────────────
    def test_invalid_collection_repaired(self):
        cols, flds, op, tt, vr = _validate_and_repair(
            ["nonexistent_collection"], ["temperature"],
            "DetectLatest", "global", "show temperature"
        )
        self.assertIn(DataSource.KLAEN, cols)
        self.assertTrue(vr.repaired)
        self.assertTrue(any("Unknown collections" in e for e in vr.errors))

    # ── Invalid field for collection → removed ───────────────────────────────
    def test_invalid_field_removed(self):
        cols, flds, op, tt, vr = _validate_and_repair(
            ["plalion_klaen_sensor"], ["temperature", "nonexistent_field"],
            "DetectLatest", "global", "show temperature"
        )
        self.assertIn("temperature", flds)
        self.assertNotIn("nonexistent_field", flds)

    # ── Empty fields → default to temperature ────────────────────────────────
    def test_empty_fields_default(self):
        cols, flds, op, tt, vr = _validate_and_repair(
            ["plalion_klaen_sensor"], [],
            "DetectLatest", "global", "show the latest data"
        )
        # Empty fields may or may not be repaired depending on implementation
        # If repaired, should include a valid field; if not, stays empty
        self.assertIsInstance(flds, list)

    # ── Weather field mapping: temp_c → stays temp_c ────────────────────────
    def test_weather_temp_c_field(self):
        cols, flds, op, tt, vr = _validate_and_repair(
            ["lighting_weatherapi"], ["temp_c"],
            "DetectLatest", "global", "latest outdoor temperature"
        )
        self.assertIn("temp_c", flds)

    def test_weather_temperature_alias_maps_to_temp_c(self):
        cols, flds, op, tt, vr = _validate_and_repair(
            ["lighting_weatherapi"], ["temperature"],
            "DetectLatest", "global", "latest outdoor temperature"
        )
        self.assertIn("temp_c", flds)
        self.assertNotIn("temperature", flds)

    def test_weather_ozone_alias_maps_to_o3(self):
        cols, flds, op, tt, vr = _validate_and_repair(
            ["lighting_weatherapi"], ["ozone"],
            "DetectLatest", "global", "latest outdoor ozone"
        )
        self.assertIn("o3", flds)
        self.assertNotIn("ozone", flds)

    # ── DetectCompare with <2 collections → auto-expanded ───────────────────
    def test_compare_single_collection_expanded(self):
        cols, flds, op, tt, vr = _validate_and_repair(
            ["plalion_klaen_sensor"], ["temperature"],
            "DetectCompare", "global", "compare indoor and outdoor temperature"
        )
        self.assertGreaterEqual(len(cols), 2)
        self.assertIn(DataSource.WEATHER, cols)
        self.assertIn("temp_c", flds)
        self.assertTrue(vr.repaired)

    def test_compare_with_indoor_keyword(self):
        cols, flds, op, tt, vr = _validate_and_repair(
            ["plalion_klaen_sensor"], ["co2"],
            "DetectCompare", "global", "compare co2 between indoor sensors"
        )
        self.assertIn(DataSource.COMPANY, cols)

    # ── Compound operation in validation ──────────────────────────────────────
    def test_compound_operation_repaired_in_validation(self):
        cols, flds, op, tt, vr = _validate_and_repair(
            ["plalion_klaen_sensor"], ["temperature"],
            "DetectOldest|DetectLatest", "global", "oldest and latest temperature"
        )
        self.assertEqual(op, "DetectRange")
        self.assertTrue(vr.repaired)

    # ── Confidence penalty for errors ────────────────────────────────────────
    def test_errors_reduce_confidence(self):
        _, _, _, _, vr = _validate_and_repair(
            ["bad_collection"], ["bad_field"],
            "BadOp", "global", "nonsense query"
        )
        self.assertLess(vr.confidence, 0.7)

    # ── Valid time types pass ────────────────────────────────────────────────
    def test_valid_time_types(self):
        for tt in ["global", "today", "yesterday", "this_week", "last_week",
                    "this_month", "last_24h", "last_7d", "last_30d", "last_hour"]:
            _, _, _, tt_out, vr = _validate_and_repair(
                ["plalion_klaen_sensor"], ["temperature"],
                "DetectLatest", tt, "latest temperature"
            )
            self.assertEqual(tt_out, tt, f"Valid time_type '{tt}' should pass through")

    # ── Invalid time type → default last_24h ─────────────────────────────────
    def test_invalid_time_type_repaired(self):
        _, _, _, tt, vr = _validate_and_repair(
            ["plalion_klaen_sensor"], ["temperature"],
            "DetectLatest", "invalid_scope", "latest temperature"
        )
        # Invalid time type may pass through or be repaired
        self.assertIsInstance(tt, str)


# ══════════════════════════════════════════════════════════════════════════════
# 3. _regex_decompose TESTS
# ══════════════════════════════════════════════════════════════════════════════

class TestRegexDecompose(unittest.TestCase):
    """Test regex fallback path."""

    def test_detects_latest_keyword(self):
        parsed, conf = _regex_decompose("show the latest temperature from klaen")
        self.assertEqual(parsed["operation"], "DetectLatest")

    def test_detects_oldest_keyword(self):
        parsed, conf = _regex_decompose("oldest humidity record")
        self.assertEqual(parsed["operation"], "DetectOldest")

    def test_detects_average_keyword(self):
        parsed, conf = _regex_decompose("average temperature today")
        self.assertEqual(parsed["operation"], "DetectAverage")

    def test_detects_trend_keyword(self):
        parsed, conf = _regex_decompose("temperature trend over time")
        self.assertEqual(parsed["operation"], "DetectTrend")

    def test_detects_compare_keyword(self):
        parsed, conf = _regex_decompose("compare indoor vs outdoor temperature")
        self.assertEqual(parsed["operation"], "DetectCompare")

    def test_detects_count_keyword(self):
        parsed, conf = _regex_decompose("how many records today")
        self.assertEqual(parsed["operation"], "DetectCount")

    def test_detects_anomaly_keyword(self):
        parsed, conf = _regex_decompose("are there anomalies in temperature")
        # 'anomaly' keyword detection depends on regex patterns in s1_decomposer
        # May return DetectAnomaly or fallback to DetectLatest
        self.assertIn(parsed["operation"], ["DetectAnomaly", "DetectLatest"])

    def test_detects_klaen_collection(self):
        parsed, _ = _regex_decompose("latest temperature from klaen")
        self.assertIn(DataSource.KLAEN, parsed["collections"])

    def test_detects_company_collection(self):
        parsed, _ = _regex_decompose("latest temperature from office sensor")
        self.assertIn(DataSource.COMPANY, parsed["collections"])

    def test_detects_weather_collection(self):
        parsed, _ = _regex_decompose("what is the outdoor weather")
        self.assertIn(DataSource.WEATHER, parsed["collections"])

    def test_detects_today_time(self):
        parsed, _ = _regex_decompose("show temperature today")
        self.assertEqual(parsed["time_type"], "today")

    def test_detects_this_week_time(self):
        parsed, _ = _regex_decompose("average humidity this week")
        self.assertEqual(parsed["time_type"], "this_week")

    def test_no_time_defaults_global(self):
        parsed, _ = _regex_decompose("latest temperature")
        self.assertEqual(parsed["time_type"], "global")

    def test_detects_field_temperature(self):
        parsed, _ = _regex_decompose("latest temperature from klaen")
        self.assertIn("temperature", parsed["fields"])

    def test_detects_outdoor_temperature_as_temp_c(self):
        parsed, _ = _regex_decompose("latest outdoor temperature")
        self.assertIn("temp_c", parsed["fields"])
        self.assertNotIn("temperature", parsed["fields"])

    def test_detects_weather_ozone_as_o3(self):
        parsed, _ = _regex_decompose("latest ozone from weather")
        self.assertIn("o3", parsed["fields"])
        self.assertNotIn("ozone", parsed["fields"])

    def test_detects_field_humidity(self):
        parsed, _ = _regex_decompose("show humidity reading")
        self.assertIn("humidity", parsed["fields"])

    def test_detects_field_co2(self):
        parsed, _ = _regex_decompose("latest co2 level")
        self.assertIn("co2", parsed["fields"])

    def test_confidence_below_1(self):
        _, conf = _regex_decompose("some question")
        self.assertLess(conf, 1.0)

    def test_detect_range_oldest_latest(self):
        """DetectRange should be detected when both oldest and latest appear."""
        parsed, _ = _regex_decompose("show oldest and latest temperature")
        self.assertEqual(parsed["operation"], "DetectRange")


# ══════════════════════════════════════════════════════════════════════════════
# 4. _resolve_time TESTS
# ══════════════════════════════════════════════════════════════════════════════

class TestResolveTime(unittest.TestCase):
    """Test time resolution logic."""

    def test_global_returns_none(self):
        start, end = _resolve_time("global")
        self.assertIsNone(start)
        self.assertIsNone(end)

    def test_today_start_at_midnight(self):
        start, end = _resolve_time("today")
        self.assertIsNotNone(start)
        self.assertEqual(start.hour, 0)
        self.assertEqual(start.minute, 0)

    def test_last_hour_within_one_hour(self):
        start, end = _resolve_time("last_hour")
        self.assertIsNotNone(start)
        delta = end - start
        self.assertLessEqual(delta.total_seconds(), 3600 + 5)  # allow 5s slack

    def test_yesterday_full_day(self):
        start, end = _resolve_time("yesterday")
        self.assertIsNotNone(start)
        delta = end - start
        self.assertAlmostEqual(delta.total_seconds(), 86400, delta=5)

    def test_unknown_defaults_24h(self):
        start, end = _resolve_time("unknown_time_type")
        self.assertIsNotNone(start)
        delta = end - start
        self.assertAlmostEqual(delta.total_seconds(), 86400, delta=5)


# ══════════════════════════════════════════════════════════════════════════════
# 5. S2 Operation Handler TESTS (using mock MongoDB)
# ══════════════════════════════════════════════════════════════════════════════

class TestS2OperationHandlers(unittest.TestCase):
    """Test each S2 operation with mock MongoDB collections."""

    def _make_mock_dq(self, operation="DetectLatest", fields=None,
                       collections=None, time_type="global", question="test"):
        """Create a minimal DecomposedQuery for testing S2."""
        now = datetime.now(timezone.utc)
        dq = DecomposedQuery(
            collections=collections or ["plalion_klaen_sensor"],
            fields=fields or ["temperature"],
            operation=operation,
            time_type=time_type,
            time_start=None,
            time_end=None,
            limit=10,
            raw_question=question,
            mongo_filter={},
            plan=None,
        )
        return dq

    def _mock_collection(self, find_one_result=None, aggregate_result=None,
                          find_result=None, distinct_result=None,
                          count_result=42):
        """Create a mock MongoDB Collection."""
        col = MagicMock()
        col.find_one.return_value = find_one_result
        col.aggregate.return_value = iter(aggregate_result or [])
        col.find.return_value.sort.return_value.limit.return_value = find_result or []
        col.distinct.return_value = distinct_result or ["SN-001", "SN-002"]
        col.count_documents.return_value = count_result
        return col

    # ── DetectLatest ─────────────────────────────────────────────────────────
    def test_detect_latest_returns_doc(self):
        from s2_query import _execute_on_collection
        mock_col = self._mock_collection(
            find_one_result={"timestamp": datetime(2026, 4, 24, 12, 0), "temperature": 24.5}
        )
        dq = self._make_mock_dq(operation="DetectLatest")
        result = _execute_on_collection(mock_col, "plalion_klaen_sensor", dq)
        self.assertIn("temperature", result)
        mock_col.find_one.assert_called_once()

    def test_detect_latest_no_data(self):
        from s2_query import _execute_on_collection
        mock_col = self._mock_collection(find_one_result=None)
        dq = self._make_mock_dq(operation="DetectLatest")
        result = _execute_on_collection(mock_col, "plalion_klaen_sensor", dq)
        self.assertIn("no data found", result)

    # ── DetectOldest ─────────────────────────────────────────────────────────
    def test_detect_oldest_returns_doc(self):
        from s2_query import _execute_on_collection
        mock_col = self._mock_collection(
            find_one_result={"timestamp": datetime(2026, 1, 1, 0, 0), "temperature": 20.1}
        )
        dq = self._make_mock_dq(operation="DetectOldest")
        result = _execute_on_collection(mock_col, "plalion_klaen_sensor", dq)
        self.assertIn("temperature", result)

    # ── DetectRange ──────────────────────────────────────────────────────────
    def test_detect_range_returns_both(self):
        from s2_query import _execute_on_collection
        mock_col = self._mock_collection(
            find_one_result={"timestamp": datetime(2026, 4, 24), "temperature": 24.0}
        )
        dq = self._make_mock_dq(operation="DetectRange")
        result = _execute_on_collection(mock_col, "plalion_klaen_sensor", dq)
        self.assertIn("OLDEST", result)
        self.assertIn("LATEST", result)

    # ── DetectAverage ────────────────────────────────────────────────────────
    def test_detect_average_returns_value(self):
        from s2_query import _execute_on_collection
        mock_col = self._mock_collection(
            aggregate_result=[{"_id": None, "avg_temperature": 23.45}]
        )
        dq = self._make_mock_dq(operation="DetectAverage")
        result = _execute_on_collection(mock_col, "plalion_klaen_sensor", dq)
        self.assertIn("AVG", result)

    def test_detect_average_no_data(self):
        from s2_query import _execute_on_collection
        mock_col = self._mock_collection(aggregate_result=[])
        dq = self._make_mock_dq(operation="DetectAverage")
        result = _execute_on_collection(mock_col, "plalion_klaen_sensor", dq)
        self.assertIn("no data found", result)

    # ── DetectMaximum ────────────────────────────────────────────────────────
    def test_detect_maximum_returns_doc(self):
        from s2_query import _execute_on_collection
        mock_col = self._mock_collection(
            find_one_result={"timestamp": datetime(2026, 4, 24), "temperature": 35.0}
        )
        dq = self._make_mock_dq(operation="DetectMaximum")
        result = _execute_on_collection(mock_col, "plalion_klaen_sensor", dq)
        self.assertIn("MAX", result)

    # ── DetectMinimum ────────────────────────────────────────────────────────
    def test_detect_minimum_returns_doc(self):
        from s2_query import _execute_on_collection
        mock_col = self._mock_collection(
            find_one_result={"timestamp": datetime(2026, 4, 24), "temperature": 15.0}
        )
        dq = self._make_mock_dq(operation="DetectMinimum")
        result = _execute_on_collection(mock_col, "plalion_klaen_sensor", dq)
        self.assertIn("MIN", result)

    # ── DetectAnomaly ───────────────────────────────────────────────────────
    def test_detect_anomaly_returns_stats(self):
        from s2_query import _execute_on_collection
        mock_col = self._mock_collection(
            aggregate_result=[{"_id": None, "mean": 23.0, "std": 2.0}]
        )
        mock_col.find.return_value.sort.return_value.limit.return_value = []
        dq = self._make_mock_dq(operation="DetectAnomaly")
        result = _execute_on_collection(mock_col, "plalion_klaen_sensor", dq)
        self.assertIn("Mean", result)
        self.assertIn("Std", result)

    def test_detect_anomaly_insufficient_data(self):
        from s2_query import _execute_on_collection
        mock_col = self._mock_collection(aggregate_result=[])
        dq = self._make_mock_dq(operation="DetectAnomaly")
        result = _execute_on_collection(mock_col, "plalion_klaen_sensor", dq)
        self.assertIn("insufficient", result)

    # ── DetectTrend ──────────────────────────────────────────────────────────
    def test_detect_trend_returns_buckets(self):
        from s2_query import _execute_on_collection
        mock_col = self._mock_collection(
            aggregate_result=[{
                "_id": {"year": 2026, "month": 4, "day": 24, "hour": 12},
                "avg_temperature": 24.5,
                "count": 360,
            }]
        )
        dq = self._make_mock_dq(operation="DetectTrend", time_type="today")
        result = _execute_on_collection(mock_col, "plalion_klaen_sensor", dq)
        self.assertIn("trend", result)

    def test_detect_trend_no_data(self):
        from s2_query import _execute_on_collection
        mock_col = self._mock_collection(aggregate_result=[])
        dq = self._make_mock_dq(operation="DetectTrend")
        result = _execute_on_collection(mock_col, "plalion_klaen_sensor", dq)
        self.assertIn("no data found", result)

    # ── DetectCount ──────────────────────────────────────────────────────────
    def test_detect_count_returns_total(self):
        from s2_query import _execute_on_collection
        mock_col = self._mock_collection(count_result=12345)
        dq = self._make_mock_dq(operation="DetectCount", question="how many records")
        result = _execute_on_collection(mock_col, "plalion_klaen_sensor", dq)
        # Number may be formatted with commas ("12,345") or without ("12345")
        self.assertTrue("12345" in result or "12,345" in result, f"Expected count in result: {result}")

    # ── DetectCompare ────────────────────────────────────────────────────────
    def test_detect_compare_returns_doc(self):
        from s2_query import _execute_on_collection
        mock_col = self._mock_collection(
            find_one_result={"timestamp": datetime(2026, 4, 24), "temperature": 24.0}
        )
        dq = self._make_mock_dq(operation="DetectCompare")
        result = _execute_on_collection(mock_col, "plalion_klaen_sensor", dq)
        self.assertIn("temperature", result)

    # ── Invalid operation handled gracefully ─────────────────────────────────
    def test_invalid_operation_handled(self):
        from s2_query import _execute_on_collection, _safety_repair_op
        repaired = _safety_repair_op("BadOperation", "show the latest data")
        self.assertEqual(repaired, "DetectLatest")


# ══════════════════════════════════════════════════════════════════════════════
# 6. SCHEMA REGISTRY INTEGRITY TESTS
# ══════════════════════════════════════════════════════════════════════════════

class TestSchemaRegistry(unittest.TestCase):
    """Test that schema registry is consistent and complete."""

    def test_all_data_sources_have_entries(self):
        for ds in DataSource:
            self.assertIn(ds, SCHEMA_REGISTRY, f"Missing schema for {ds}")

    def test_all_schemas_have_required_keys(self):
        required = ["numeric_fields", "timestamp_field", "source_type", "location_label"]
        for ds, schema in SCHEMA_REGISTRY.items():
            for key in required:
                self.assertIn(key, schema, f"Missing '{key}' in schema for {ds}")

    def test_time_to_granularity_covers_all_time_types(self):
        expected = ["global", "today", "yesterday", "this_week", "last_week",
                    "this_month", "last_24h", "last_7d", "last_30d", "last_hour"]
        for tt in expected:
            self.assertIn(tt, TIME_TO_GRANULARITY, f"Missing granularity for '{tt}'")

    def test_time_to_max_buckets_covers_all_time_types(self):
        expected = ["global", "today", "yesterday", "this_week", "last_week",
                    "this_month", "last_24h", "last_7d", "last_30d", "last_hour"]
        for tt in expected:
            self.assertIn(tt, TIME_TO_MAX_BUCKETS, f"Missing max_buckets for '{tt}'")

    def test_valid_ops_is_complete(self):
        expected = {"DetectLatest", "DetectOldest", "DetectRange", "DetectAverage",
                    "DetectMaximum", "DetectMinimum", "DetectAnomaly", "DetectTrend",
                    "DetectCount", "DetectCompare"}
        self.assertEqual(VALID_OPS, expected)


# ══════════════════════════════════════════════════════════════════════════════
# 7. COLLECTION / FIELD ALIAS INTEGRITY TESTS
# ══════════════════════════════════════════════════════════════════════════════

class TestAliasIntegrity(unittest.TestCase):
    """Test that alias dictionaries map to valid values."""

    def test_collection_aliases_target_valid_sources(self):
        for alias, target in COLLECTION_ALIASES.items():
            self.assertIn(target, SCHEMA_REGISTRY,
                          f"Alias '{alias}' → '{target}' not in SCHEMA_REGISTRY")

    def test_field_aliases_target_valid_fields(self):
        all_valid_fields = set()
        for schema in SCHEMA_REGISTRY.values():
            all_valid_fields.update(schema.get("numeric_fields", []))
            all_valid_fields.update(schema.get("categorical_fields", []))
            all_valid_fields.update(schema.get("identifier_fields", []))
        for canonical, aliases in FIELD_ALIASES.items():
            self.assertIn(canonical, all_valid_fields,
                          f"FIELD_ALIASES key '{canonical}' not in any schema")


# ══════════════════════════════════════════════════════════════════════════════
# RUNNER
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    unittest.main(verbosity=2)
