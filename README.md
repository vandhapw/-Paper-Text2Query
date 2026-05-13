# Text2Query — IoT Sensor Natural Language Query System

A 4-stage pipeline for querying IoT sensor data in natural language:

1. **S1 Decomposer** — LLM + regex fallback parses NL questions into a structured QueryPlan (IR)
2. **S2 Query** — Executes the IR against MongoDB with time-aware filtering
3. **S3 Answer** — LLM generates a natural language answer from the retrieved data
4. **S4 Visualization** — Auto-generates Plotly charts for trend, compare, and anomaly queries

## Quick Start

```bash
# Recommended Python: 3.11 or 3.12

# Set up environment
cp .env.example .env   # Edit with your MongoDB URI and Ollama host

# Create an isolated environment
python -m venv .venv
.\.venv\Scripts\activate   # Windows PowerShell
pip install -r requirements.txt

# Seed MongoDB with test data
python seed_mongodb.py

# Run interactive chat
python final_chat.py

# Run full benchmark (205 questions)
python benchmark_rigorous.py --no-ablation

# Run unit tests
python -m pytest test_unit.py -v
```

## Configuration

All config is in `.env`:

| Variable | Description |
|----------|-------------|
| `MONGODB_URI` | MongoDB connection string |
| `DB_NAME` | Database name (default: `server_db`) |
| `OLLAMA_HOST` | Ollama API base URL |
| `BENCHMARK_MODEL` | LLM model for S1/S3 (default: `qwen3.5:397b-cloud`) |
| `REFERENCE_DATE` | Fixed date for benchmark (format: `YYYY-MM-DD`). If set, time-scoped queries use this date instead of "now". |

Do not hardcode MongoDB passwords or API keys in Python files. Keep real secrets only in `.env` or a secret manager, and rotate credentials if they were ever committed or shared.

## Collections

| Collection | Description | Fields |
|------------|-------------|--------|
| `plalion_klaen_sensor` | Indoor Klaen sensor | temperature, humidity, co2, voc, dust, ozone |
| `plalion_company_sensor` | Office sensor | temperature, humidity, co2, dust, voc, ozone, serial_number, device_id |
| `lighting_weatherapi` | Outdoor weather | temp_c, feelslike_c, humidity, wind_kph, precip_mm, condition, aqi, pm2_5, pm10, co, no2, so2, o3, pressure_mb |

## Supported Operations

| Operation | Description |
|-----------|-------------|
| DetectLatest | Most recent record |
| DetectOldest | Oldest record |
| DetectRange | Oldest AND latest |
| DetectAverage | Mean value over time range |
| DetectMaximum | Record with highest field value |
| DetectMinimum | Record with lowest field value |
| DetectAnomaly | Statistical outlier detection |
| DetectTrend | Time-bucketed aggregation |
| DetectCount | Count records or distinct values |
| DetectCompare | Compare values across collections |

## Benchmark Results

| Metric | v1 (qwen2.5-coder:1.5b) | v2 (qwen3.5:397b-cloud) |
|--------|--------------------------|--------------------------|
| S1 Op Accuracy | 78.5% | 100.0% |
| S1 Collections | 91.2% | 96.1% |
| S1 Fields Jaccard | 80.4% | 86.9% |
| S2 Data Hit Rate | 50.7% | 64.4% |
| Pipeline Errors | 4/205 | 0/205 |

### Known Limitations

- **S2_NO_DATA 35.6%**: Time-scoped queries (today, this_week, etc.) use the current date, but seeded data is from 2025. Set `REFERENCE_DATE=2025-02-15` in `.env` for benchmark runs to match data range.
- **DetectAnomaly** uses only the first requested field; additional fields are ignored with a warning.
- **S4 Visualization** requires Kaleido for PNG export; falls back to HTML if unavailable.
- **Ground Truth v2**: Annotator B for groups D–K is algorithmic perturbation, not human. Use `human_only_agreement_report()` for publication-grade κ.

## Architecture

```
User Question
     │
     ▼
[S1 Decomposer] ──→ QueryPlan (IR) ──→ Validation & Self-Repair
     │
     ▼
[S2 Query] ──→ MongoDB aggregation ──→ Raw data context
     │
     ▼
[S3 Answer] ──→ LLM-generated natural language answer
     │
     ▼
[S4 Visualization] ──→ Plotly chart (optional)
```

## File Reference

| File | Purpose |
|------|---------|
| `s1_decomposer.py` | Stage 1: LLM + regex NL→QueryPlan |
| `s2_query.py` | Stage 2: MongoDB query execution |
| `s3_answer.py` | Stage 3: LLM answer generation |
| `s4_visualization.py` | Stage 4: Auto chart generation |
| `query_plan.py` | IR definitions, schema registry |
| `final_chat.py` | Interactive REPL chat |
| `benchmark_rigorous.py` | Full 205-question benchmark |
| `ground_truth_v2.py` | Dual-annotated ground truth |
| `test_unit.py` | 71 unit tests |
| `error_analysis.py` | Confusion matrix & failure taxonomy |
| `seed_mongodb.py` | Database seeding script |
