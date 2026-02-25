"""侧边栏：日期过滤、可视化模式、颜色依据等控件。"""
from __future__ import annotations

from datetime import date, timedelta

import streamlit as st

from data_loader import get_date_range, load_daily_stats
import config


def _init_state(key: str, default):
    """首次加载时才设置默认值，之后保留用户选择。"""
    if key not in st.session_state:
        st.session_state[key] = default


def _parse_qp_date(key: str, fallback: date, lo: date, hi: date) -> date:
    """从 URL query params 解析日期字符串，失败时返回 fallback。"""
    try:
        val = st.query_params.get(key)
        if val:
            d = date.fromisoformat(str(val))
            return max(lo, min(d, hi))
    except (ValueError, TypeError, AttributeError):
        pass
    return fallback


def _render_color_legend(color_by: str) -> None:
    """在侧边栏渲染当前颜色方案对应的图例。"""
    base = "font-size:9px;color:#9ca3af;font-weight:600;letter-spacing:.3px"
    swatch = (
        "display:inline-block;width:11px;height:11px;"
        "border-radius:3px;margin-right:5px;vertical-align:middle;flex-shrink:0"
    )

    if color_by == "速度":
        html = f"""
        <div style="padding:6px 8px 6px;background:rgba(59,130,246,0.04);
                    border-radius:8px;border:0.5px solid rgba(59,130,246,0.1);margin-bottom:2px">
          <div style="display:flex;align-items:center;gap:6px;margin-bottom:6px">
            <div style="{swatch};background:rgba(120,120,120,0.65)"></div>
            <span style="{base}">未知 / 静止</span>
          </div>
          <div style="height:9px;border-radius:5px;
                      background:linear-gradient(90deg,
                        rgb(50,230,80) 0%,
                        rgb(0,255,200) 10%,
                        rgb(0,180,255) 30%,
                        rgb(255,220,0) 60%,
                        rgb(255,120,0) 82%,
                        rgb(255,30,30) 100%);
                      box-shadow:0 1px 3px rgba(0,0,0,0.1);margin-bottom:4px"></div>
          <div style="display:flex;justify-content:space-between;{base};font-feature-settings:'tnum';padding:0 1px">
            <span>0</span><span>5</span><span>20</span><span>60</span><span>100</span><span>150+</span>
          </div>
          <div style="{base};text-align:center;margin-top:2px;letter-spacing:1.5px;opacity:.7">KM / H</div>
        </div>"""

    elif color_by == "精度":
        items = [
            ("rgb(0,255,100)",   "≤ 5 m  精确"),
            ("rgb(0,200,255)",   "≤ 15 m  良好"),
            ("rgb(255,200,0)",   "≤ 40 m  一般"),
            ("rgb(255,60,60)",   "> 40 m  差"),
        ]
        rows = "".join(
            f'<div style="display:flex;align-items:center;margin-bottom:3px">'
            f'<div style="{swatch};background:{c}"></div>'
            f'<span style="{base}">{label}</span></div>'
            for c, label in items
        )
        html = f"""
        <div style="padding:6px 8px;background:rgba(59,130,246,0.04);
                    border-radius:8px;border:0.5px solid rgba(59,130,246,0.1);margin-bottom:2px">
          {rows}
        </div>"""

    elif color_by == "时段":
        items = [
            ("rgb(40,40,200)",   "0 – 6 h  深夜"),
            ("rgb(0,200,255)",   "6 – 12 h  上午"),
            ("rgb(255,220,0)",   "12 – 18 h  下午"),
            ("rgb(160,0,255)",   "18 – 24 h  夜晚"),
        ]
        rows = "".join(
            f'<div style="display:flex;align-items:center;margin-bottom:3px">'
            f'<div style="{swatch};background:{c}"></div>'
            f'<span style="{base}">{label}</span></div>'
            for c, label in items
        )
        html = f"""
        <div style="padding:6px 8px;background:rgba(59,130,246,0.04);
                    border-radius:8px;border:0.5px solid rgba(59,130,246,0.1);margin-bottom:2px">
          {rows}
        </div>"""

    elif color_by == "活动类型":
        items = [
            ("rgb(0,255,120)",  "步行"),
            ("rgb(0,160,255)",  "驾驶"),
        ]
        rows = "".join(
            f'<div style="display:flex;align-items:center;margin-bottom:3px">'
            f'<div style="{swatch};background:{c}"></div>'
            f'<span style="{base}">{label}</span></div>'
            for c, label in items
        )
        html = f"""
        <div style="padding:6px 8px;background:rgba(59,130,246,0.04);
                    border-radius:8px;border:0.5px solid rgba(59,130,246,0.1);margin-bottom:2px">
          {rows}
        </div>"""
    else:
        return

    st.sidebar.markdown(html, unsafe_allow_html=True)


def render_sidebar(df) -> dict:
    """渲染侧边栏并返回过滤参数字典。"""
    st.sidebar.markdown(
        """
        <div class="tl-sidebar-logo">
            <div class="tl-sidebar-icon">🛰</div>
            <div class="tl-sidebar-text">
                <div class="tl-sidebar-wordmark">Track<em>Lens</em></div>
                <div class="tl-sidebar-sub">GPS 轨迹可视化</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.sidebar.divider()

    data_min, data_max = get_date_range(df)

    # ── URL query params → session state（F5 刷新后从 URL 恢复）────────
    _valid_modes    = ["自定义范围", "最近 7 天", "最近 30 天", "全部数据"]
    _valid_vizmodes = ["热力图", "轨迹线", "散点", "热力 + 轨迹"]

    qp_mode    = st.query_params.get("date_mode", "最近 30 天")
    if qp_mode not in _valid_modes:
        qp_mode = "最近 30 天"

    qp_vizmode = st.query_params.get("view_mode", "热力图")
    if qp_vizmode not in _valid_vizmodes:
        qp_vizmode = "热力图"

    _default_start = max(data_min, data_max - timedelta(days=30))
    qp_start = _parse_qp_date("start", _default_start, data_min, data_max)
    qp_end   = _parse_qp_date("end",   data_max,        data_min, data_max)

    def _qp_bool(key: str, default: bool) -> bool:
        v = st.query_params.get(key)
        if v is None:
            return default
        return str(v).lower() not in ("0", "false", "no")

    qp_flights = _qp_bool("show_flights", True)
    qp_photos  = _qp_bool("show_photos",  True)

    # ── session_state 默认值（session 内优先，F5 后从 URL 初始化）──────
    _init_state("date_mode",     qp_mode)
    _init_state("_saved_start",  qp_start)
    _init_state("_saved_end",    qp_end)
    _init_state("view_mode",     qp_vizmode)
    _init_state("globe_mode",    False)
    _init_state("color_by",      "速度")
    _init_state("max_accuracy",  100)
    _init_state("speed_range",   (0, 200))
    _init_state("activity_opts", ["步行 (stepType=1)", "驾驶 (stepType=0)"])
    _init_state("show_flights",  qp_flights)
    _init_state("show_photos",   qp_photos)

    # ── 日期模式 ─────────────────────────────────────────────────────────
    st.sidebar.markdown("#### 时间范围", unsafe_allow_html=False)
    date_mode = st.sidebar.radio(
        "模式",
        _valid_modes,
        label_visibility="collapsed",
        key="date_mode",
    )

    if date_mode == "自定义范围":
        # widget key 被 Streamlit 清除后从持久 key 恢复
        if "custom_start" not in st.session_state:
            st.session_state["custom_start"] = st.session_state["_saved_start"]
        if "custom_end" not in st.session_state:
            st.session_state["custom_end"] = st.session_state["_saved_end"]

        col1, col2 = st.sidebar.columns(2)
        start = col1.date_input(
            "起始",
            min_value=data_min,
            max_value=data_max,
            key="custom_start",
        )
        end = col2.date_input(
            "截止",
            min_value=data_min,
            max_value=data_max,
            key="custom_end",
        )
        if start > end:
            st.sidebar.error("起始日期不能晚于截止日期")
            start = end

        # 同步到持久 key 和 URL（F5 后可恢复）
        st.session_state["_saved_start"] = start
        st.session_state["_saved_end"]   = end
        st.query_params["date_mode"] = "自定义范围"
        st.query_params["start"]     = start.isoformat()
        st.query_params["end"]       = end.isoformat()

    elif date_mode == "最近 7 天":
        end   = data_max
        start = max(data_min, data_max - timedelta(days=6))
        st.query_params["date_mode"] = date_mode
        for k in ("start", "end"):
            if k in st.query_params:
                del st.query_params[k]

    elif date_mode == "最近 30 天":
        end   = data_max
        start = max(data_min, data_max - timedelta(days=29))
        st.query_params["date_mode"] = date_mode
        for k in ("start", "end"):
            if k in st.query_params:
                del st.query_params[k]

    else:  # 全部数据
        start = data_min
        end   = data_max
        st.query_params["date_mode"] = date_mode
        for k in ("start", "end"):
            if k in st.query_params:
                del st.query_params[k]

    st.sidebar.markdown(
        f"<div class='tl-date-caption'>{start} → {end} &nbsp;·&nbsp; {(end - start).days + 1} 天</div>",
        unsafe_allow_html=True,
    )
    st.sidebar.divider()

    # ── 可视化模式 ────────────────────────────────────────────────────────
    st.sidebar.markdown("#### 可视化模式", unsafe_allow_html=False)
    view_mode = st.sidebar.radio(
        "模式",
        ["热力图", "轨迹线", "散点", "热力 + 轨迹"],
        label_visibility="collapsed",
        key="view_mode",
    )
    # 写入 URL，F5 后恢复
    st.query_params["view_mode"] = view_mode

    globe_mode = st.sidebar.toggle("🌍 地球视图", key="globe_mode")
    st.sidebar.divider()

    # ── 颜色依据 ──────────────────────────────────────────────────────────
    st.sidebar.markdown("#### 颜色依据", unsafe_allow_html=False)
    color_by = st.sidebar.selectbox(
        "颜色依据",
        ["速度", "精度", "时段", "活动类型"],
        label_visibility="collapsed",
        key="color_by",
    )
    _render_color_legend(color_by)
    st.sidebar.divider()

    # ── 过滤器 ────────────────────────────────────────────────────────────
    st.sidebar.markdown("#### 数据过滤", unsafe_allow_html=False)
    max_accuracy = st.sidebar.slider(
        "GPS 精度上限（误差 ≤ N 米）", 5, 300, step=5,
        key="max_accuracy",
    )
    speed_range = st.sidebar.slider(
        "速度范围（km/h）", 0, 250, step=5,
        key="speed_range",
    )

    activity_opts = st.sidebar.multiselect(
        "活动类型",
        options=["步行 (stepType=1)", "驾驶 (stepType=0)"],
        key="activity_opts",
    )
    step_types: list[int] = []
    if "步行 (stepType=1)" in activity_opts:
        step_types.append(1)
    if "驾驶 (stepType=0)" in activity_opts:
        step_types.append(0)
    if not step_types:
        step_types = [0, 1]

    st.sidebar.divider()

    # ── 图层叠加开关 ──────────────────────────────────────────────────────
    st.sidebar.markdown("#### 叠加图层", unsafe_allow_html=False)

    show_photos = st.sidebar.toggle("📷 照片位置", key="show_photos")
    st.query_params["show_photos"] = "1" if show_photos else "0"

    import os
    flights_ready = os.path.exists("data/flight_tracks.json")
    if flights_ready:
        show_flights = st.sidebar.toggle("✈️ 飞行轨迹", key="show_flights")
        st.query_params["show_flights"] = "1" if show_flights else "0"
    else:
        st.sidebar.markdown(
            "<small style='color:#4a5568;'>✈️ 飞行轨迹：先运行<br>"
            "<code>python preprocess_flights.py</code></small>",
            unsafe_allow_html=True,
        )
        show_flights = False

    st.sidebar.divider()

    # ── 视口提示（过滤已在 JS 端实时完成）────────────────────────────────
    import streamlit as _st
    _zoom = float(_st.query_params.get("map_zoom", "4"))
    if _zoom >= 11:
        st.sidebar.markdown(
            "<div style='font-size:10px;color:#3b82f6;text-align:center;"
            "padding:4px 0;font-weight:600'>🔍 视口全量模式 · 实时更新</div>",
            unsafe_allow_html=True,
        )
    else:
        st.sidebar.markdown(
            "<div style='font-size:10px;color:#9ca3af;text-align:center;"
            "padding:4px 0'>放大到街道级可查看视口全量散点</div>",
            unsafe_allow_html=True,
        )

    # ── 关于 ──────────────────────────────────────────────────────────────
    st.sidebar.markdown(
        "<div class='tl-sidebar-footer'>Powered by Streamlit · deck.gl</div>",
        unsafe_allow_html=True,
    )

    return {
        "start": start,
        "end": end,
        "view_mode": view_mode,
        "globe_mode": globe_mode,
        "color_by": color_by,
        "max_accuracy": max_accuracy,
        "speed_min": float(speed_range[0]),
        "speed_max": float(speed_range[1]),
        "step_types": step_types,
        "show_flights": show_flights,
        "show_photos": show_photos,
    }
