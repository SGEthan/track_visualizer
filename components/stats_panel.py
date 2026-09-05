"""轨迹指标和图表。"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from data_loader import build_trip_summary, to_display_datetime


FONT = "Inter, -apple-system, BlinkMacSystemFont, SF Pro Display, sans-serif"
PAPER = "rgba(0,0,0,0)"
GRID = "rgba(43,83,97,.11)"
TEXT = "#657c87"
INK = "#19333e"
CYAN = "#0a829f"
MINT = "#15956d"
AMBER = "#bd790e"


def _metric(label: str, value: str, detail: str, accent: str) -> str:
    return f"""
    <div class="tl-metric" style="--metric-accent:{accent}">
      <div class="tl-metric-label">{label}</div>
      <div class="tl-metric-value">{value}</div>
      <div class="tl-metric-detail">{detail}</div>
    </div>
    """


def _layout(height: int, title: str) -> dict:
    return {
        "height": height,
        "paper_bgcolor": PAPER,
        "plot_bgcolor": PAPER,
        "font": {"family": FONT, "color": TEXT, "size": 10},
        "margin": {"l": 8, "r": 8, "t": 52, "b": 12},
        "title": {"text": title, "x": 0.015, "font": {"family": FONT, "color": INK, "size": 12}},
        "hoverlabel": {"bgcolor": "#ffffff", "bordercolor": "rgba(10,130,159,.2)", "font": {"family": FONT, "color": INK}},
    }


def render_stats(df: pd.DataFrame, trip_summary: pd.DataFrame | None = None) -> None:
    if df.empty:
        return
    summary = trip_summary if trip_summary is not None else build_trip_summary(df)
    dates = to_display_datetime(df["ts"])
    local_dates = dates.dt.date if isinstance(dates, pd.Series) else dates.date
    total_km = float(summary["distance_km"].sum()) if not summary.empty else 0.0
    total_hours = float((summary["end_ts"] - summary["start_ts"]).clip(lower=0).sum()) / 3600
    journey_speed = total_km / total_hours if total_hours > 0 else 0.0
    median_accuracy = float(df["accuracy"].median())

    cards = st.columns(5)
    values = [
        ("RECORDED POINTS", f"{len(df):,}", "filtered signal", CYAN),
        ("JOURNEYS", f"{len(summary):,}", "distinct segments", "#8a9cff"),
        ("ACTIVE DAYS", f"{pd.Series(local_dates).nunique():,}", "days with signal", MINT),
        ("DISTANCE", f"{total_km:,.0f}", "kilometres traced", AMBER),
        ("JOURNEY PACE", f"{journey_speed:.1f}", f"km/h · ±{median_accuracy:.0f}m median", "#ff8a76"),
    ]
    for column, item in zip(cards, values):
        with column:
            st.markdown(_metric(*item), unsafe_allow_html=True)

    left, right = st.columns([1.65, 1])
    daily = pd.Series(local_dates).value_counts().sort_index()
    with left:
        figure = go.Figure()
        figure.add_trace(go.Scatter(
            x=daily.index.astype(str), y=daily.values,
            mode="lines", line={"color": CYAN, "width": 2},
            fill="tozeroy", fillcolor="rgba(10,130,159,.09)",
            hovertemplate="<b>%{x}</b><br>%{y:,} points<extra></extra>",
        ))
        figure.add_trace(go.Scatter(
            x=daily.index.astype(str), y=daily.values,
            mode="markers", marker={"size": 4, "color": "#ffffff", "line": {"width": 3, "color": "rgba(10,130,159,.18)"}},
            hoverinfo="skip",
        ))
        figure.update_layout(**_layout(250, "SIGNAL RHYTHM · DAILY POINTS"), showlegend=False)
        figure.update_xaxes(showgrid=False, tickfont={"size": 9}, tickangle=-25, fixedrange=True)
        figure.update_yaxes(gridcolor=GRID, zeroline=False, tickfont={"size": 9}, fixedrange=True)
        st.plotly_chart(figure, width="stretch", config={"displayModeBar": False})

    with right:
        speed = df["speed"].to_numpy(dtype=float)
        labels = ["未知", "步行 <5", "慢速 5–30", "驾驶 30–80", "高速 80+"]
        counts = np.array([
            (speed < 0).sum(),
            ((speed >= 0) & (speed < 5)).sum(),
            ((speed >= 5) & (speed < 30)).sum(),
            ((speed >= 30) & (speed < 80)).sum(),
            (speed >= 80).sum(),
        ])
        keep = counts > 0
        palette = ["#536674", MINT, CYAN, "#8a9cff", "#ff776d"]
        donut = go.Figure(go.Pie(
            labels=np.array(labels)[keep], values=counts[keep], hole=0.7,
            marker={"colors": np.array(palette)[keep], "line": {"color": "#ffffff", "width": 2}},
            textinfo="none", sort=False,
            hovertemplate="<b>%{label}</b><br>%{value:,} points · %{percent}<extra></extra>",
        ))
        donut.update_layout(
            **_layout(250, "PACE PROFILE"),
            showlegend=True,
            legend={"orientation": "h", "y": -0.04, "x": 0.5, "xanchor": "center", "font": {"size": 9}},
            annotations=[{"text": f"{journey_speed:.1f}<br><span style='font-size:9px;color:#718792'>KM/H</span>", "x": .5, "y": .5, "showarrow": False, "font": {"size": 18, "color": INK, "family": FONT}}],
        )
        st.plotly_chart(donut, width="stretch", config={"displayModeBar": False})
