# Text2Query — Idea Summary

## Title
Text2Query: A Four-Stage Natural Language Query Pipeline for IoT Sensor Data with MongoDB Integration

## Problem Statement
IoT sensor networks generate massive volumes of multivariate time-series data stored in databases like MongoDB. Non-technical users (facility managers, building operators, environmental scientists) cannot query this data using MongoDB's query language. Existing text-to-SQL tools (Spider, DIN-SQL) target tabular SQL databases and cannot handle time-series aggregations, anomaly detection, or cross-collection comparisons that IoT sensor queries require.

## Proposed Solution
Text2Query is a four-stage pipeline that converts natural language questions about IoT sensor data into structured answers with automatic visualizations:

1. **S1 Decomposer** — Uses LLM (Qwen3.5:397B-cloud) with regex fallback to parse NL questions into a structured QueryPlan (intermediate representation) specifying operation type, target collections, fields, and time scope.

2. **S2 Query** — Executes the QueryPlan against MongoDB with time-aware filtering, handling 10 operation types (DetectLatest, DetectOldest, DetectRange, DetectAverage, DetectMaximum, DetectMinimum, DetectAnomaly, DetectTrend, DetectCount, DetectCompare).

3. **S3 Answer** — LLM generates a natural language answer from retrieved data context.

4. **S4 Visualization** — Auto-generates Plotly charts for trend, compare, and anomaly queries.

## Key Innovations
- **Hybrid NL-to-Query architecture**: LLM parsing with regex fallback ensures robustness (99% LLM success, 100% pipeline success)
- **QueryPlan intermediate representation**: Structured IR with self-repair capability enables stage-independent evaluation
- **10 IoT-specific operations**: Beyond CRUD — includes statistical aggregations, anomaly detection, trend analysis, and cross-collection comparison
- **Automatic visualization**: Context-aware chart generation without user specification
- **Stage-independent evaluation**: Each pipeline stage evaluated independently with ground truth comparison

## Experimental Results
- **205 benchmark questions** across 10 operation types + edge cases
- **S1 Operation Detection**: 100% accuracy (95% CI: [98.2%, 100%])
- **S1 Collection Detection**: 96.1% accuracy (95% CI: [92.5%, 98.0%])
- **S1 Field Extraction**: 87.6% Jaccard similarity
- **S2 Data Hit Rate**: 63.9% (time-scoped queries are the primary limitation)
- **LLM Success Rate**: 99% (only 2/205 required regex fallback)
- **Self-Repair Rate**: 0.98% (only 2/205 queries needed repair)
- **Confusion Matrix**: Near-diagonal (perfect operation classification for 9/10 types)
- **Latency**: Median 18.7s end-to-end (S1 dominates at ~44s with cloud LLM)

## Target Application
Indoor air quality (IAQ) monitoring across three MongoDB collections:
- plalion_klaen_sensor (temperature, humidity, CO2, VOC, dust, ozone)
- plalion_company_sensor (temperature, humidity, CO2, dust, VOC, ozone, device_id)
- lighting_weatherapi (outdoor weather: temperature, humidity, wind, precipitation, AQI, PM2.5, PM10)

## Reproducibility
- Open-source code on GitHub
- Docker container for reproducibility
- 205-question benchmark with ground truth annotations
- Unit tests (71 tests)
- MongoDB seeding script for test data