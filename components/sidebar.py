"""Track Lens 控制面板。"""
from __future__ import annotations

from datetime import date, timedelta

import streamlit as st

import config
from data_loader import get_date_range


def _init(key: str, value) -> None:
    if key not in st.session_state:
        st.session_state[key] = value


def _query_bool(key: str, default: bool) -> bool:
    value = st.query_params.get(key)
    return default if value is None else str(value).lower() not in {"0", "false", "no"}


def _query_date(key: str, fallback: date, low: date, high: date) -> date:
    try:
        value = date.fromisoformat(str(st.query_params.get(key, "")))
        return max(low, min(value, high))
    except (TypeError, ValueError):
        return fallback


def _legend(color_by: str) -> None:
    if color_by == "速度":
        st.sidebar.markdown(
            """
            <div class="tl-legend">
              <div class="tl-gradient speed"></div>
              <div class="tl-legend-axis"><span>0</span><span>5</span><span>20</span><span>60</span><span>100</span><span>150+</span></div>
              <small>KM / H</small>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return
    entries = {
        "精度": [("#39e991", "≤ 5m"), ("#35d8ff", "≤ 15m"), ("#ffd15c", "≤ 40m"), ("#ff6678", "> 40m")],
        "时段": [("#5965e8", "深夜"), ("#36d7ff", "上午"), ("#ffd85a", "下午"), ("#bd5dff", "夜晚")],
        "活动类型": [("#42ef9b", "步行"), ("#3ca8ff", "非步行")],
    }[color_by]
    items = "".join(
        f"<span><i style='background:{color}'></i>{label}</span>" for color, label in entries
    )
    st.sidebar.markdown(f"<div class='tl-dot-legend'>{items}</div>", unsafe_allow_html=True)


def render_sidebar(
    df,
    flights: list[dict] | None = None,
    airborne_candidates=None,
) -> dict:
    st.sidebar.markdown(
        """
        <div class="tl-side-brand">
          <div class="tl-orbit"><i></i></div>
          <div><strong>TRACK LENS</strong><small>PERSONAL MOBILITY ATLAS</small></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    data_min, data_max = get_date_range(df, flights)
    date_modes = ["最近 7 天", "最近 30 天", "自定义范围", "全部数据"]
    view_modes = ["热力图", "轨迹线", "散点", "热力 + 轨迹"]
    recent_end = min(data_max, max(data_min, date.today()))
    default_start = max(data_min, recent_end - timedelta(days=29))

    query_mode = st.query_params.get("date_mode", "最近 30 天")
    query_view = st.query_params.get("view_mode", "热力图")
    if query_mode not in date_modes:
        query_mode = "最近 30 天"
    if query_view not in view_modes:
        query_view = "热力图"

    _init("date_mode", query_mode)
    _init("view_mode", query_view)
    _init("custom_start", _query_date("start", default_start, data_min, data_max))
    _init("custom_end", _query_date("end", recent_end, data_min, data_max))
    _init("globe_mode", _query_bool("globe", False))
    _init("color_by", "速度")
    _init("max_accuracy", 100)
    _init("speed_range", (0, 200))
    _init("activity_opts", ["步行", "非步行"])
    _init("show_photos", _query_bool("show_photos", True))
    _init("show_flights", _query_bool("show_flights", True))
    _init("review_airborne", _query_bool("review_airborne", False))

    st.sidebar.markdown("<div class='tl-section-kicker'>TIME WINDOW</div>", unsafe_allow_html=True)
    date_mode = st.sidebar.radio("时间范围", date_modes, key="date_mode", label_visibility="collapsed")
    if date_mode == "最近 7 天":
        start, end = max(data_min, recent_end - timedelta(days=6)), recent_end
    elif date_mode == "最近 30 天":
        start, end = default_start, recent_end
    elif date_mode == "全部数据":
        start, end = data_min, data_max
    else:
        left, right = st.sidebar.columns(2)
        start = left.date_input("开始", min_value=data_min, max_value=data_max, key="custom_start")
        end = right.date_input("结束", min_value=data_min, max_value=data_max, key="custom_end")
        if start > end:
            start = end
            st.sidebar.warning("开始日期已调整为结束日期。")

    st.query_params["date_mode"] = date_mode
    if date_mode == "自定义范围":
        st.query_params["start"], st.query_params["end"] = start.isoformat(), end.isoformat()
    else:
        for key in ("start", "end"):
            if key in st.query_params:
                del st.query_params[key]
    st.sidebar.markdown(
        f"<div class='tl-date-readout'><span>{start:%Y.%m.%d}</span><i></i><span>{end:%Y.%m.%d}</span><small>{(end-start).days+1} DAYS</small></div>",
        unsafe_allow_html=True,
    )

    st.sidebar.markdown("<div class='tl-section-kicker'>MAP MODE</div>", unsafe_allow_html=True)
    view_mode = st.sidebar.radio("地图模式", view_modes, key="view_mode", label_visibility="collapsed")
    globe_mode = st.sidebar.toggle("地球视图", key="globe_mode")
    st.query_params["view_mode"] = view_mode
    st.query_params["globe"] = "1" if globe_mode else "0"

    st.sidebar.markdown("<div class='tl-section-kicker'>COLOR LANGUAGE</div>", unsafe_allow_html=True)
    color_by = st.sidebar.selectbox(
        "颜色依据", ["速度", "精度", "时段", "活动类型"], key="color_by", label_visibility="collapsed"
    )
    _legend(color_by)

    st.sidebar.markdown("<div class='tl-section-kicker'>SIGNAL FILTER</div>", unsafe_allow_html=True)
    max_accuracy = st.sidebar.slider("定位误差上限", 5, 300, step=5, key="max_accuracy", help="数值越小，保留的位置越精确。")
    speed_range = st.sidebar.slider("速度范围 · km/h", 0, 250, step=5, key="speed_range")
    activity_opts = st.sidebar.multiselect("活动类型", ["步行", "非步行"], key="activity_opts")
    step_types = ([1] if "步行" in activity_opts else []) + ([0] if "非步行" in activity_opts else [])
    if not step_types:
        step_types = [0, 1]

    st.sidebar.markdown("<div class='tl-section-kicker'>OVERLAYS</div>", unsafe_allow_html=True)
    show_photos = st.sidebar.toggle("照片位置", key="show_photos")
    if config.FLIGHTS_PATH.exists():
        show_flights = st.sidebar.toggle("飞行轨迹", key="show_flights")
    else:
        show_flights = False
        st.sidebar.caption("运行 `preprocess_flights.py` 后可显示航班。")
    st.query_params["show_photos"] = "1" if show_photos else "0"
    st.query_params["show_flights"] = "1" if show_flights else "0"

    airborne_flight_id = "__all__"
    if airborne_candidates is not None and not airborne_candidates.empty:
        st.sidebar.markdown("<div class='tl-section-kicker'>AIRBORNE REVIEW</div>", unsafe_allow_html=True)
        review_airborne = st.sidebar.toggle("查看已排除的机上 GPS", key="review_airborne")
        st.query_params["review_airborne"] = "1" if review_airborne else "0"
        if review_airborne:
            grouped = airborne_candidates.groupby("flight_match_id", sort=False)
            labels = {
                str(match_id): (
                    f"{group['flight_match_label'].iloc[0]} · {len(group):,} pts"
                )
                for match_id, group in grouped
            }
            options = ["__all__", *labels]
            airborne_flight_id = st.sidebar.selectbox(
                "候选航班",
                options,
                format_func=lambda value: (
                    f"全部 {len(labels)} 个航班 · {len(airborne_candidates):,} pts"
                    if value == "__all__" else labels[value]
                ),
                key="airborne_flight_id",
            )
            st.sidebar.caption("洋红色虚线是已从常规轨迹和统计中排除的机上 GPS 审计层。")
    else:
        review_airborne = False

    _, provider, _ = config.basemap_config()
    token_state = "Mapbox token active" if provider == "MAPBOX" else "CARTO fallback · no token needed"
    st.sidebar.markdown(
        f"""
        <div class="tl-system-card">
          <div><i></i><span>SYSTEM READY</span></div>
          <p>{token_state}</p><p>{config.TIMEZONE_NAME}</p>
        </div>
        """,
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
        "review_airborne": review_airborne,
        "airborne_flight_id": airborne_flight_id,
    }
