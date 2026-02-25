"""统计面板：指标卡片 + Plotly 图表（深色宇宙主题）。"""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from data_loader import build_trip_summary

_FONT = ("-apple-system, BlinkMacSystemFont, 'SF Pro Text',"
         " 'Helvetica Neue', Arial, sans-serif")
_PAPER  = "rgba(0,0,0,0)"
_PLOT   = "rgba(0,0,0,0)"
_GRID   = "rgba(0,0,0,0.07)"
_T1     = "#1a1f2e"
_T2     = "#6b7280"
_T3     = "#374151"
_BLUE   = "#3b82f6"
_CYAN   = "#06b6d4"
_GREEN  = "#10b981"
_AMBER  = "#f59e0b"
_RED    = "#ef4444"
_SLATE  = "#334155"


def _base_layout(**kw) -> dict:
    return dict(
        paper_bgcolor=_PAPER,
        plot_bgcolor=_PLOT,
        font=dict(color=_T2, size=11, family=_FONT),
        margin=dict(l=4, r=4, t=36, b=4),
        **kw,
    )


def render_stats(df: pd.DataFrame) -> None:
    if df.empty:
        st.info("当前筛选条件下无数据。")
        return

    n_points = len(df)
    n_trips  = df["trip_id"].nunique()
    n_days   = pd.to_datetime(df["ts"], unit="s").dt.date.nunique()
    valid_spd = df.loc[df["speed"] >= 0, "speed"]
    avg_speed = float(valid_spd.mean()) if len(valid_spd) else 0.0

    _trip_sum = build_trip_summary(df)
    total_km  = float(_trip_sum["distance_km"].sum()) if not _trip_sum.empty else 0.0

    # ── 5 指标卡 ───────────────────────────────────────────────────────────────
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("GPS 点数",  f"{n_points:,}")
    c2.metric("行程数",    f"{n_trips:,}")
    c3.metric("活跃天数",  f"{n_days:,}")
    c4.metric("平均速度",  f"{avg_speed:.1f} km/h")
    c5.metric("总里程",    f"{total_km:,.0f} km")

    # ── 图表行 ─────────────────────────────────────────────────────────────────
    col_bar, col_pie = st.columns([3, 1])

    # 每日点数柱状图
    with col_bar:
        dates = pd.to_datetime(df["ts"], unit="s").dt.date
        daily = dates.value_counts().sort_index()

        fig = go.Figure(
            go.Bar(
                x=daily.index.astype(str),
                y=daily.values,
                marker=dict(
                    color=daily.values,
                    colorscale=[
                        [0,   "#1e3a8a"],
                        [0.4, _BLUE],
                        [1,   _CYAN],
                    ],
                    showscale=False,
                    line=dict(width=0),
                ),
                hovertemplate=(
                    "<b>%{x}</b><br>%{y:,} 个点<extra></extra>"
                ),
                hoverlabel=dict(
                    bgcolor="#1a1f2e",
                    bordercolor="rgba(0,0,0,0)",
                    font=dict(color="#f1f5f9", size=11, family=_FONT),
                ),
            )
        )
        fig.update_layout(
            **_base_layout(height=168),
            title=dict(
                text="每日记录点数",
                font=dict(size=12, color=_T3, family=_FONT, weight="bold"),
                x=0, pad=dict(l=0),
            ),
            xaxis=dict(
                showgrid=False,
                tickfont=dict(size=9, color=_T3),
                tickangle=-30,
                showline=False,
                zeroline=False,
            ),
            yaxis=dict(
                gridcolor=_GRID,
                tickfont=dict(size=9, color=_T3),
                showline=False,
                zeroline=False,
            ),
            bargap=0.25,
        )
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

    # 速度分布甜甜圈图
    with col_pie:
        bins = {
            "静止":        int((df["speed"] < 0).sum()),
            "步行\n(<5)":  int(((df["speed"] >= 0) & (df["speed"] < 5)).sum()),
            "慢速\n(5-30)":int(((df["speed"] >= 5) & (df["speed"] < 30)).sum()),
            "驾驶\n(30-80)":int(((df["speed"] >= 30) & (df["speed"] < 80)).sum()),
            "高速\n(80+)": int((df["speed"] >= 80).sum()),
        }
        labels = [k for k, v in bins.items() if v > 0]
        values = [v for v in bins.values() if v > 0]
        palette = [_SLATE, _GREEN, _BLUE, "#6366f1", _RED]
        colors  = palette[:len(labels)]

        fig2 = go.Figure(
            go.Pie(
                labels=labels,
                values=values,
                hole=0.62,
                marker=dict(
                    colors=colors,
                    line=dict(color="rgba(240,244,248,0.9)", width=2),
                ),
                textinfo="percent",
                textfont=dict(size=9.5, family=_FONT),
                insidetextorientation="radial",
                hovertemplate=(
                    "<b>%{label}</b><br>%{value:,} 个点  %{percent}<extra></extra>"
                ),
                hoverlabel=dict(
                    bgcolor="#1a1f2e",
                    bordercolor="rgba(0,0,0,0)",
                    font=dict(color="#f1f5f9", size=11, family=_FONT),
                ),
                pull=[0.03] * len(labels),
            )
        )
        fig2.update_layout(
            **_base_layout(height=168),
            title=dict(
                text="速度分布",
                font=dict(size=12, color=_T3, family=_FONT, weight="bold"),
                x=0, pad=dict(l=0),
            ),
            showlegend=False,
            annotations=[dict(
                text="速度",
                x=0.5, y=0.5,
                font=dict(size=11, color=_T2, family=_FONT, weight="bold"),
                showarrow=False,
            )],
        )
        st.plotly_chart(fig2, width="stretch", config={"displayModeBar": False})
