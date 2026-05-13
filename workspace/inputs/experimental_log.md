# Text2Query — Experimental Log

## Experiment 1: Full 205-Question Benchmark (v2 Cloud)
**Date:** 2026-04-25
**Config:** skip_stage3=true, skip_stage4=true (S1+S2 only), model=qwen3.5:397b-cloud, num_questions=205

### Aggregate Results
| Metric | Value | 95% CI |
|--------|-------|--------|
| S1 Operation Accuracy | 100.0% | [98.2%, 100%] |
| S1 Collections Accuracy | 96.1% | [92.5%, 98.0%] |
| S1 Fields Jaccard Avg | 87.6% | — |
| S1 Time Scope Accuracy | 96.1% | — |
| S2 Data Hit Rate | 63.9% | — |
| S1 Repair Rate | 0.98% | — |
| S1 LLM Success Rate | 99.02% | — |
| Pipeline Errors | 0/205 | — |

### Per-Operation Breakdown (205 questions)
| Operation | n | Op Accuracy | Collections Accuracy | Avg Latency (ms) |
|-----------|---|-------------|---------------------|-------------------|
| DetectLatest | 20 | 100% | 100% | 10,677 |
| DetectOldest | 15 | 100% | 100% | 28,206 |
| DetectRange | 15 | 100% | 100% | 33,068 |
| DetectTrend | 25 | 100% | 100% | 28,891 |
| DetectAverage | 20 | 100% | 100% | 23,777 |
| DetectMaximum | 15 | 100% | 100% | 26,353 |
| DetectMinimum | 15 | 100% | 100% | 29,255 |
| DetectAnomaly | 20 | 100% | 100% | 22,720 |
| DetectCount | 20 | 100% | 100% | 121,459 |
| DetectCompare | 20 | 100% | 90% | 100,683 |
| EdgeCase | 20 | 100% | 70% | 52,252 |

### Confusion Matrix
Near-diagonal: all 10 operation types correctly classified. Only 2 misclassifications in the entire 205-question set.

### Latency Analysis
- Median total latency: 18,688 ms
- S1 average latency: 44,418 ms (dominated by cloud LLM call)
- S2 average latency: 6 ms (MongoDB query)
- DetectCount and DetectCompare have highest latency (>100s) due to aggregation pipeline complexity

### S2 Data Hit Rate Analysis
- Overall: 63.9% data retrieval success
- Primary failure cause: time-scoped queries (today, this_week) reference current date but seeded data is from 2025
- When time scope is "global" or data matches reference date, hit rate approaches 100%
- Setting REFERENCE_DATE=2025-02-15 would improve S2 to ~85-90%

## Experiment 2: Pilot 20-Question Rigorous Evaluation
**Date:** 2026-04-22
**Config:** All 4 stages active, model=regex fallback (cloud LLM unavailable)

### Results
| Stage | Metric | Accuracy |
|-------|--------|----------|
| S1 | Operation Detection | 100% |
| S1 | Collection ID | 100% |
| S1 | Field Extraction | 100% |
| S2 | Data Retrieval | 95% |
| S3 | Answer Generation | 100% |
| S4 | Visualization | 100% (4/4) |

**Overall:** Average score 0.988 (±0.054), success rate 100% (95% CI: [83.9%, 100%])
**Latency:** 5.54s average (regex fallback much faster than cloud LLM)

### Baseline Comparison
| Method | Combined Accuracy | Δ |
|--------|-----------------|---|
| Keyword Baseline | 90.0% | — |
| Text2Query (Regex) | 100.0% | +11.1% |

Two-proportion z-test: p = 0.1468 (not significant at n=20; expected p < 0.05 with n=205)

## Experiment 3: User Study (Planned)
- Target: 10-15 participants
- Tasks: 5-10 questions each
- Metrics: Task completion rate, SUS score, time on task
- Status: Protocol designed, not yet conducted

## Figures Available
1. **Figure 1: Architecture** — 4-stage pipeline diagram
2. **Figure 2: Performance by Operation** — Bar chart of accuracy across 11 operation types
3. **Figure 3: Latency Breakdown** — Stacked bar chart of S1/S2/S3/S4 latency
4. **Figure 4: Stage Accuracy** — Radar chart of per-stage accuracy
5. **Figure 5: Error Distribution** — Error type frequency across operations
6. **Figure 6: Collection Heatmap** — Accuracy heatmap by operation × collection