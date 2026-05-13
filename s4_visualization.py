"""
Stage 4 – Auto Visualization (Multi-variable + Multi-collection)
Supports:
  - Single field,  single collection   → simple line/bar
  - Single field,  multi collection    → subplots per collection (same y-axis)
  - Multi field,   single collection   → subplots per field (different y-axes)
  - Multi field,   multi collection    → grid subplots (field × collection)
"""

from __future__ import annotations
import os
import re
from datetime import datetime
from typing import Optional

import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── Config ─────────────────────────────────────────────────────────────────────
CHART_DIR = os.environ.get("CHART_DIR", "charts")
os.makedirs(CHART_DIR, exist_ok=True)

FIELD_UNITS = {
    "temperature": "°C",   "temp_c": "°C",    "feelslike_c": "°C",
    "humidity":    "%",
    "co2":         "ppm",
    "dust":        "µg/m³","pm2_5": "µg/m³",  "pm10": "µg/m³",
    "voc":         "ppb",
    "ozone":       "ppb",  "o3": "ppb",        "no2": "ppb",  "so2": "ppb",
    "wind_kph":    "km/h",
    "precip_mm":   "mm",
    "aqi":         "AQI",
    "pressure_mb": "hPa",
}

COLLECTION_LABELS = {
    "plalion_klaen_sensor":   "Klaen (Indoor)",
    "plalion_company_sensor": "Company (Office)",
    "lighting_weatherapi":    "WeatherAPI (Outdoor)",
}

# Distinct color per collection
COLLECTION_COLORS = {
    "plalion_klaen_sensor":   "#00B4D8",
    "plalion_company_sensor": "#F4845F",
    "lighting_weatherapi":    "#74C69D",
}

# Distinct marker symbol per collection (for overlaid plots)
COLLECTION_SYMBOLS = {
    "plalion_klaen_sensor":   "circle",
    "plalion_company_sensor": "square",
    "lighting_weatherapi":    "diamond",
}

# Distinct dash style per field (when multiple fields share one subplot)
FIELD_DASHES = ["solid", "dash", "dot", "dashdot", "longdash"]

# Plotly dark theme palette for extra fields
EXTRA_COLORS = [
    "#CBA6F7", "#FAB387", "#A6E3A1", "#89DCEB",
    "#F38BA8", "#F9E2AF", "#B4BEFE", "#94E2D5",
]


# ══════════════════════════════════════════════════════════════════════════════
# CONTEXT PARSER
# ══════════════════════════════════════════════════════════════════════════════

def _parse_context(context: str) -> dict[str, dict]:
    """
    Parse Stage 2 context string into:
    {
      "plalion_klaen_sensor": {
          "operation": "DetectTrend",
          "timestamps": [datetime, ...],
          "values": {
              "co2":         [128.4, 130.1, ...],
              "temperature": [24.1,  24.5,  ...],
          },
          "__labels__": [...],   # for group-by count charts
          "__counts__": [...],
      },
      ...
    }
    """
    data: dict[str, dict] = {}
    current_col: Optional[str] = None

    for line in context.splitlines():
        # ── Collection header ────────────────────────────────────────────
        col_match = re.search(r"Collection:\s*([\w_]+)", line)
        if col_match:
            current_col = col_match.group(1)
            data[current_col] = {
                "operation":  "",
                "timestamps": [],
                "values":     {},
                "__labels__": [],
                "__counts__": [],
            }
            op_match = re.search(r"Operation:\s*(\w+)", line)
            if op_match:
                data[current_col]["operation"] = op_match.group(1)
            continue

        if current_col is None:
            continue

        # ── Bucket line: [2026-03-11 07:00 UTC] co2=128.4 (n=360) ───────
        bucket = re.match(
            r"\s*\[(\d{4}-\d{2}-\d{2}(?:\s+\d{2}:\d{2})?(?:\s*UTC)?)\]\s*(.*)",
            line,
        )
        if bucket:
            ts_str = bucket.group(1).replace(" UTC", "").strip()
            try:
                ts = (
                    datetime.strptime(ts_str, "%Y-%m-%d %H:%M")
                    if len(ts_str) > 10
                    else datetime.strptime(ts_str, "%Y-%m-%d")
                )
            except ValueError:
                continue

            data[current_col]["timestamps"].append(ts)
            for k, v in re.findall(r"([\w_]+)=([-\d.]+)", bucket.group(2)):
                if k in ("count", "n"):
                    continue
                data[current_col]["values"].setdefault(k, []).append(float(v))
            continue

        # ── Group-by count line: #1 SN-00412 → 124,302 records ───────────
        rank = re.match(r"\s*#\s*\d+\s+(.+?)\s+→\s+([\d,]+)\s+records", line)
        if rank:
            data[current_col]["__labels__"].append(rank.group(1).strip())
            data[current_col]["__counts__"].append(
                int(rank.group(2).replace(",", ""))
            )

    return data


# ══════════════════════════════════════════════════════════════════════════════
# SUBPLOT GRID BUILDER
# ══════════════════════════════════════════════════════════════════════════════

def _make_grid(
    parsed:     dict[str, dict],
    fields:     list[str],
    chart_type: str,   # "line" | "bar" | "scatter"
    question:   str,
) -> Optional[go.Figure]:
    """
    Universal grid builder:
      rows = fields  (one row per field/variable)
      cols = 1       (all collections overlaid with distinct color+symbol)

    This gives:
      - 1 field  × 1 collection  → 1×1 single plot
      - 1 field  × N collections → 1×1 with N traces overlaid
      - M fields × 1 collection  → M×1 subplots
      - M fields × N collections → M×1 subplots, N traces each
    """
    # Filter to collections that actually have timestamps
    active_cols = [c for c in parsed if parsed[c]["timestamps"]]
    if not active_cols:
        return None

    # Filter to fields that exist in at least one collection
    active_fields = [
        f for f in fields
        if any(f in parsed[c]["values"] for c in active_cols)
    ]
    if not active_fields:
        return None

    n_rows = len(active_fields)
    subplot_titles = [
        f"{f.upper()}  ({FIELD_UNITS.get(f, '')})" for f in active_fields
    ]

    fig = make_subplots(
        rows=n_rows, cols=1,
        shared_xaxes=True,
        subplot_titles=subplot_titles,
        vertical_spacing=0.08 if n_rows <= 4 else 0.05,
    )

    legend_added: set[str] = set()

    for row_idx, field in enumerate(active_fields, start=1):
        for col_name in active_cols:
            d     = parsed[col_name]
            vals  = d["values"].get(field)
            times = d["timestamps"]

            if not vals or len(vals) != len(times):
                continue

            label      = COLLECTION_LABELS.get(col_name, col_name)
            color      = COLLECTION_COLORS.get(col_name, EXTRA_COLORS[0])
            symbol     = COLLECTION_SYMBOLS.get(col_name, "circle")
            show_legend = label not in legend_added
            legend_added.add(label)

            if chart_type == "line":
                fig.add_trace(
                    go.Scatter(
                        x=times, y=vals,
                        mode="lines+markers",
                        name=label,
                        line=dict(color=color, width=2),
                        marker=dict(size=5, symbol=symbol),
                        legendgroup=label,
                        showlegend=show_legend,
                        hovertemplate=(
                            f"<b>{label}</b><br>"
                            f"{field}: %{{y:.2f}} {FIELD_UNITS.get(field,'')}<br>"
                            "%{x}<extra></extra>"
                        ),
                    ),
                    row=row_idx, col=1,
                )

            elif chart_type == "bar":
                fig.add_trace(
                    go.Bar(
                        x=times, y=vals,
                        name=label,
                        marker_color=color,
                        legendgroup=label,
                        showlegend=show_legend,
                        hovertemplate=(
                            f"<b>{label}</b><br>"
                            f"{field}: %{{y:.2f}} {FIELD_UNITS.get(field,'')}<br>"
                            "%{x}<extra></extra>"
                        ),
                    ),
                    row=row_idx, col=1,
                )

            elif chart_type == "scatter":
                fig.add_trace(
                    go.Scatter(
                        x=times, y=vals,
                        mode="markers",
                        name=label,
                        marker=dict(
                            color=vals,
                            colorscale="RdYlGn_r",
                            size=8,
                            symbol=symbol,
                            showscale=(row_idx == 1 and col_name == active_cols[0]),
                            colorbar=dict(
                                title=FIELD_UNITS.get(field, ""),
                                len=0.4,
                                y=1 - (row_idx - 1) / n_rows,
                            ),
                        ),
                        legendgroup=label,
                        showlegend=show_legend,
                        hovertemplate=(
                            f"<b>{label}</b><br>"
                            f"{field}: %{{y:.2f}} {FIELD_UNITS.get(field,'')}<br>"
                            "%{x}<extra></extra>"
                        ),
                    ),
                    row=row_idx, col=1,
                )

        # Y-axis label per row
        unit = FIELD_UNITS.get(field, "")
        fig.update_yaxes(
            title_text=f"{field} ({unit})" if unit else field,
            row=row_idx, col=1,
            gridcolor="#313244",
            zerolinecolor="#45475A",
        )

    fig.update_xaxes(
        gridcolor="#313244",
        zerolinecolor="#45475A",
    )

    return fig


# ══════════════════════════════════════════════════════════════════════════════
# CHART BUILDERS (per operation)
# ══════════════════════════════════════════════════════════════════════════════

def _chart_trend(parsed, fields, question) -> Optional[str]:
    fig = _make_grid(parsed, fields, "line", question)
    if fig is None:
        return None
    n_cols = len([c for c in parsed if parsed[c]["timestamps"]])
    n_flds = len(fields)
    fig.update_layout(
        title=dict(
            text=f"📈 CO₂/Sensor Trend — {question[:70]}",
            font=dict(size=16, color="#CBA6F7"),
        ),
        height=280 * len(fields) + 100,
    )
    _apply_theme(fig)
    return _save_chart(fig, f"trend_{'_'.join(fields[:3])}")


def _chart_compare(parsed, fields, question) -> Optional[str]:
    """
    For DetectCompare / Average / Max / Min:
    Bar chart with grouped bars — one group per field, one bar per collection.
    """
    active_cols = [c for c in parsed if parsed[c]["timestamps"] or parsed[c]["values"]]
    active_fields = [
        f for f in fields
        if any(f in parsed[c]["values"] for c in active_cols)
    ]
    if not active_fields or not active_cols:
        return None

    fig = go.Figure()
    for col_name in active_cols:
        d      = parsed[col_name]
        label  = COLLECTION_LABELS.get(col_name, col_name)
        color  = COLLECTION_COLORS.get(col_name, "#888")
        y_vals = []
        x_labs = []
        for field in active_fields:
            vals = d["values"].get(field, [])
            if vals:
                x_labs.append(f"{field} ({FIELD_UNITS.get(field,'')})")
                y_vals.append(round(vals[-1], 2))

        if y_vals:
            fig.add_trace(go.Bar(
                name=label,
                x=x_labs, y=y_vals,
                marker_color=color,
                text=[f"{v}" for v in y_vals],
                textposition="outside",
            ))

    fig.update_layout(
        barmode="group",
        title=dict(
            text=f"📊 Sensor Comparison — {question[:70]}",
            font=dict(size=16, color="#CBA6F7"),
        ),
        xaxis_title="Field",
        yaxis_title="Value",
        height=500,
    )
    _apply_theme(fig)
    return _save_chart(fig, f"compare_{'_'.join(fields[:3])}")


def _chart_anomaly(parsed, fields, question) -> Optional[str]:
    fig = _make_grid(parsed, fields, "scatter", question)
    if fig is None:
        return None
    fig.update_layout(
        title=dict(
            text=f"🔴 Anomaly Detection — {question[:70]}",
            font=dict(size=16, color="#F38BA8"),
        ),
        height=280 * len(fields) + 100,
    )
    _apply_theme(fig)
    return _save_chart(fig, f"anomaly_{'_'.join(fields[:3])}")


def _chart_count_groupby(parsed, question) -> Optional[str]:
    """Horizontal bar — top devices by record count."""
    for col_name, d in parsed.items():
        labels = d["__labels__"]
        counts = d["__counts__"]
        if not labels:
            continue
        fig = go.Figure(go.Bar(
            x=counts, y=labels,
            orientation="h",
            marker_color=COLLECTION_COLORS.get(col_name, "#00B4D8"),
            text=[f"{c:,}" for c in counts],
            textposition="outside",
        ))
        fig.update_layout(
            title=dict(
                text=f"📦 Record Count by Device — {COLLECTION_LABELS.get(col_name, col_name)}",
                font=dict(size=16, color="#CBA6F7"),
            ),
            xaxis_title="Record Count",
            yaxis=dict(autorange="reversed"),
            height=max(400, 40 * len(labels)),
        )
        _apply_theme(fig)
        return _save_chart(fig, "count_groupby")
    return None


# ══════════════════════════════════════════════════════════════════════════════
# THEME & SAVE
# ══════════════════════════════════════════════════════════════════════════════

def _apply_theme(fig: go.Figure):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#1E1E2E",
        plot_bgcolor="#1E1E2E",
        font=dict(family="Inter, sans-serif", color="#CDD6F4", size=13),
        legend=dict(
            bgcolor="rgba(30,30,46,0.8)",
            bordercolor="#45475A",
            borderwidth=1,
            orientation="h",
            yanchor="bottom", y=1.01,
            xanchor="right",  x=1,
        ),
        margin=dict(l=70, r=50, t=80, b=60),
        hovermode="x unified",
    )


def _save_chart(fig: go.Figure, name: str) -> str:
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_path = os.path.join(CHART_DIR, f"{name}_{ts}")

    # Try PNG first (requires Kaleido/Chrome), fall back to HTML
    try:
        png_path = f"{base_path}.png"
        fig.write_image(png_path, width=1100, height=fig.layout.height or 500, scale=2)
        return png_path
    except Exception as e:
        print(f"  ⚠️ PNG export failed ({e}), saving as HTML")
        html_path = f"{base_path}.html"
        fig.write_html(html_path)
        return html_path


# ══════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def run_stage4(context: str, dq, question: str) -> Optional[str]:
    """
    Auto-detect chart type from DecomposedQuery.operation and render.
    Returns path to saved PNG or None if not visualizable.
    """
    op     = dq.operation
    fields = [f for f in dq.fields if f != "timestamp"]
    parsed = _parse_context(context)

    CHART_DISPATCH = {
        "DetectTrend":   lambda: _chart_trend(parsed, fields, question),
        "DetectCompare": lambda: _chart_compare(parsed, fields, question),
        "DetectAverage": lambda: _chart_compare(parsed, fields, question),
        "DetectMaximum": lambda: _chart_compare(parsed, fields, question),
        "DetectMinimum": lambda: _chart_compare(parsed, fields, question),
        "DetectAnomaly": lambda: _chart_anomaly(parsed, fields, question),
        "DetectCount":   lambda: _chart_count_groupby(parsed, question),
    }

    chart_fn = CHART_DISPATCH.get(op)
    if chart_fn:
        try:
            path = chart_fn()
            if path:
                print(f"  ✅ Chart saved → {path}")
            return path
        except Exception as e:
            print(f"  ⚠️  Visualization failed: {e}")
    return None
