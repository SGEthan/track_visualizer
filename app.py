"""Track Lens — personal mobility atlas."""
from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd
import streamlit as st

import config
from components.color_utils import _PHOTO_COLOR, color_column_vec, path_color
from components.map_layers import _flight_color, make_globe_viewport, make_viewport
from components.map_view import render_flat_map, render_globe_map
from components.sidebar import render_sidebar
from components.stats_panel import render_stats
from data_loader import (
    apply_filters,
    build_path_data,
    build_trip_summary,
    downsample,
    exclude_airborne_points,
    filter_by_dates,
    format_timestamps,
    load_all_data,
    load_flight_data,
    match_flight_windows,
    spread_flight_paths,
)


st.set_page_config(
    page_title="Track Lens · Personal Mobility Atlas",
    page_icon="◉",
    layout="wide",
    initial_sidebar_state="expanded",
)


def _load_styles() -> None:
    path = config.PROJECT_ROOT / "assets" / "style.css"
    if path.exists():
        st.markdown(f"<style>{path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


def _query_float(key: str, default: float) -> float:
    try:
        value = st.query_params.get(key)
        return float(value) if value is not None else default
    except (TypeError, ValueError):
        return default


def _duration(minutes: int) -> str:
    if minutes < 60:
        return f"{minutes} min"
    hours, remain = divmod(minutes, 60)
    return f"{hours}h {remain:02d}m" if remain else f"{hours}h"


def _trip_picker(df: pd.DataFrame) -> list[int]:
    """在侧边栏显示当前筛选范围内的行程选择器。"""
    summary = build_trip_summary(df)
    with st.sidebar:
        st.markdown("<div class='tl-section-kicker'>JOURNEYS</div>", unsafe_allow_html=True)
        with st.expander(f"行程列表 · {len(summary):,}", expanded=False):
            if summary.empty:
                st.caption("当前范围没有可选行程。")
                return []
            if len(summary) > 250:
                st.caption("行程超过 250 条，请先缩小日期范围再精确选择。")
                return []

            labels: dict[int, str] = {}
            starts = format_timestamps(summary["start_ts"].to_numpy(), "%m-%d %H:%M")
            for index, row in summary.iterrows():
                distance = f" · {row['distance_km']:.1f} km" if row["distance_km"] >= 0.1 else ""
                labels[int(row["trip_id"])] = (
                    f"{starts[index]} · {_duration(int(row['duration_min']))}{distance}"
                )
            return st.multiselect(
                "选择行程（留空显示全部）",
                options=summary["trip_id"].astype(int).tolist(),
                format_func=lambda trip_id: labels[int(trip_id)],
                key="trip_selector",
            )


def _flight_layers(flights: list[dict]) -> tuple[list[dict], list[dict]]:
    paths = [
        {
            "path": flight["path"],
            "color": _flight_color(flight),
            "label": (
                f"{flight.get('airline', '')}{flight.get('flight', '')} · "
                f"{flight.get('from_city', flight.get('from_iata', ''))} → "
                f"{flight.get('to_city', flight.get('to_iata', ''))} · "
                f"{flight.get('date', '')} · {flight.get('distance_km', 0):,} km"
            ),
        }
        for flight in flights
    ]
    airports: dict[str, dict] = {}
    for flight in flights:
        for iata, coords, city in (
            (flight.get("from_iata", ""), flight.get("from_coords"), flight.get("from_city", "")),
            (flight.get("to_iata", ""), flight.get("to_coords"), flight.get("to_city", "")),
        ):
            if iata and coords and iata not in airports:
                airports[iata] = {
                    "longitude": coords[0],
                    "latitude": coords[1],
                    "iata": iata,
                    "city": city or iata,
                }
    return paths, list(airports.values())


def _photo_layer(df: pd.DataFrame, enabled: bool) -> list[dict]:
    if not enabled or "source" not in df.columns:
        return []
    photos = df.loc[df["source"].eq("photo")]
    if photos.empty:
        return []
    timestamps = format_timestamps(photos["ts"].to_numpy())
    accuracy = photos["accuracy"].fillna(0).to_numpy()
    return [
        {
            "longitude": float(lon),
            "latitude": float(lat),
            "color": _PHOTO_COLOR,
            "ts_fmt": timestamps[index],
            "accuracy": float(accuracy[index]),
        }
        for index, (lon, lat) in enumerate(photos[["lon", "lat"]].to_numpy())
    ]


def _scatter_payload(df: pd.DataFrame, color_by: str) -> dict | None:
    if df.empty:
        return None
    colors = color_column_vec(df, color_by)
    accuracy = df["accuracy"].fillna(0).to_numpy(dtype=float)
    speed = df["speed"].to_numpy(dtype=float)
    return {
        "n": int(len(df)),
        "lons": df["lon"].to_numpy(),
        "lats": df["lat"].to_numpy(),
        "colors": colors.reshape(-1),
        "radii": np.clip(accuracy * 0.4, 2.0, 25.0),
        "ts_fmts": format_timestamps(df["ts"].to_numpy()),
        "speed_fmts": [f"{value:.1f} km/h" if value >= 0 else "未知" for value in speed],
        "accuracies": accuracy,
    }


def _airborne_payload(df: pd.DataFrame) -> dict:
    """Build the temporary review overlay for time-matched Footprint points."""
    if df.empty:
        return {"paths": [], "points": []}

    sampled = downsample(df, config.MAX_PATH_POINTS)
    paths: list[dict] = []
    for _, group in sampled.groupby(["flight_match_id", "trip_id"], sort=False):
        ordered = group.sort_values("ts")
        if len(ordered) < 2:
            continue
        paths.append({
            "path": ordered[["lon", "lat"]].to_numpy().tolist(),
            "label": str(ordered["flight_match_label"].iloc[0]),
            "timeBasis": str(ordered["flight_time_basis"].iloc[0]),
        })

    point_source = downsample(df, config.MAX_HEATMAP_POINTS)
    timestamps = format_timestamps(point_source["ts"].to_numpy())
    points = [
        {
            "coordinates": [float(row.lon), float(row.lat)],
            "label": str(row.flight_match_label),
            "time": timestamps[index],
            "speed": f"{row.speed:.1f} km/h" if row.speed >= 0 else "未知",
            "altitude": f"{row.altitude:.0f} m" if pd.notna(row.altitude) else "未知",
            "timeBasis": str(row.flight_time_basis),
        }
        for index, row in enumerate(point_source.itertuples(index=False))
    ]
    return {"paths": paths, "points": points}


def _render_hero(start, end, point_count: int, trip_count: int, flight_count: int) -> None:
    st.markdown(
        f"""
        <section class="tl-hero">
          <div class="tl-hero-copy">
            <div class="tl-eyebrow"><span></span> PERSONAL MOBILITY ATLAS</div>
            <h1>Track <em>Lens</em></h1>
            <p>把散落在时间里的位置记录，重新拼成一张属于你的世界地图。</p>
          </div>
          <div class="tl-hero-meta">
            <div><small>WINDOW</small><strong>{start:%Y.%m.%d} — {end:%Y.%m.%d}</strong></div>
            <i></i>
            <div><small>VISIBLE DATA</small><strong>{point_count:,} pts · {trip_count:,} journeys · {flight_count:,} flights</strong></div>
          </div>
        </section>
        """,
        unsafe_allow_html=True,
    )


_load_styles()

with st.spinner("正在整理你的移动档案…"):
    all_tracks = load_all_data()
    all_flights = load_flight_data()
    all_airborne_candidates = match_flight_windows(all_tracks, all_flights)
    ground_tracks = exclude_airborne_points(all_tracks, all_airborne_candidates)

filters = render_sidebar(ground_tracks, all_flights, all_airborne_candidates)
date_tracks = filter_by_dates(ground_tracks, filters["start"], filters["end"])
filtered = apply_filters(
    date_tracks,
    max_accuracy=filters["max_accuracy"],
    speed_min=filters["speed_min"],
    speed_max=filters["speed_max"],
    step_types=filters["step_types"],
)

selected_trips = _trip_picker(filtered)
if selected_trips:
    filtered = filtered.loc[filtered["trip_id"].isin(selected_trips)]

trip_summary = build_trip_summary(filtered)
point_count = len(filtered)
trip_count = len(trip_summary)
start_text, end_text = filters["start"].isoformat(), filters["end"].isoformat()
date_flights = [
    flight for flight in all_flights
    if start_text <= flight.get("date", "") <= end_text
]
display_flights = spread_flight_paths(date_flights) if filters["show_flights"] else []
airborne_candidates = all_airborne_candidates.iloc[0:0]
if filters["review_airborne"]:
    airborne_candidates = all_airborne_candidates
    if filters["airborne_flight_id"] != "__all__":
        airborne_candidates = airborne_candidates.loc[
            airborne_candidates["flight_match_id"].eq(filters["airborne_flight_id"])
        ]
_render_hero(
    filters["start"], filters["end"], point_count, trip_count, len(display_flights)
)
if not airborne_candidates.empty:
    candidate_flights = airborne_candidates["flight_match_id"].nunique()
    st.caption(
        f"AIRBORNE REVIEW · {len(airborne_candidates):,} 个候选点 · "
        f"{candidate_flights:,} 个航班时间窗 · 洋红色虚线 · 常规轨迹已排除"
    )

if filtered.empty and not display_flights and airborne_candidates.empty:
    st.markdown(
        "<div class='tl-empty'><b>这段时间还没有留下轨迹</b><span>调整日期、精度或速度条件后再看看。</span></div>",
        unsafe_allow_html=True,
    )
    st.stop()
if filtered.empty and display_flights:
    st.info("这个时间窗口只有航班记录；地图会继续显示 Flighty 航线。")

view_mode = filters["view_mode"]
globe_mode = filters["globe_mode"]
effective_mode = "轨迹线" if globe_mode and view_mode in ("热力图", "热力 + 轨迹") else view_mode

if globe_mode and effective_mode != view_mode:
    st.caption("地球视图使用轨迹线呈现；热力图仅在平面地图中可用。")

viewport_source = filtered
if not airborne_candidates.empty:
    viewport_source = pd.concat([
        filtered[["lon", "lat"]],
        airborne_candidates[["lon", "lat"]],
    ], ignore_index=True)
elif filtered.empty and display_flights:
    airport_coords = [
        coords
        for flight in display_flights
        for coords in (flight.get("from_coords"), flight.get("to_coords"))
        if coords
    ]
    viewport_source = pd.DataFrame(airport_coords, columns=["lon", "lat"])
viewport = make_globe_viewport(viewport_source) if globe_mode else make_viewport(viewport_source)
data_viewport = {
    "longitude": float(viewport.longitude),
    "latitude": float(viewport.latitude),
    "zoom": float(viewport.zoom),
}
viewport_payload = dict(data_viewport)
if not globe_mode:
    viewport_payload = {
        "longitude": _query_float("map_lon", data_viewport["longitude"]),
        "latitude": _query_float("map_lat", data_viewport["latitude"]),
        "zoom": _query_float("map_zoom", data_viewport["zoom"]),
    }

filter_signature = "|".join((
    filters["start"].isoformat(), filters["end"].isoformat(),
    view_mode, str(globe_mode),
    str(filters["max_accuracy"]), str(filters["speed_min"]), str(filters["speed_max"]),
    ",".join(map(str, sorted(filters["step_types"]))),
    ",".join(map(str, sorted(selected_trips))),
    str(filters["review_airborne"]), filters["airborne_flight_id"],
))
filter_key = hashlib.sha1(filter_signature.encode()).hexdigest()[:12]

heatmap_data: list = []
if effective_mode in ("热力图", "热力 + 轨迹"):
    heat_source = downsample(filtered, config.MAX_HEATMAP_POINTS)
    heatmap_data = heat_source[["lon", "lat"]].to_numpy().tolist()

scatter_data = _scatter_payload(filtered, filters["color_by"]) if effective_mode == "散点" else None

path_data: list[dict] = []
if effective_mode in ("轨迹线", "热力 + 轨迹") or globe_mode:
    path_source = downsample(filtered, config.MAX_PATH_POINTS)
    path_data = [
        {"path": item["path"], "color": path_color(item["avg_speed"])}
        for item in build_path_data(path_source)
    ]

photo_data = _photo_layer(filtered, filters["show_photos"])
airborne_data = _airborne_payload(airborne_candidates)

flight_paths, airport_data = _flight_layers(display_flights)

tile_url, provider, attribution = config.basemap_config()
payload = {
    "tileUrl": tile_url,
    "heatmapData": heatmap_data,
    "scatterData": scatter_data,
    "scatterCap": config.MAX_SCATTER_POINTS,
    "pathData": path_data,
    "photoScatter": photo_data,
    "flightPathData": flight_paths,
    "airportData": airport_data,
    "airborneData": airborne_data,
    "viewport": viewport_payload,
    "dataViewport": data_viewport,
    "filterKey": filter_key,
    "meta": {
        "mode": "ORBITAL VIEW" if globe_mode else effective_mode,
        "summary": (
            f"{point_count:,} points · {trip_count:,} journeys"
            + (f" · {len(airborne_candidates):,} airborne candidates"
               if not airborne_candidates.empty else "")
        ),
        "detail": f"{filters['start']} → {filters['end']} · {config.TIMEZONE_NAME}",
        "provider": provider,
        "attribution": attribution,
    },
}

st.markdown("<div class='tl-map-label'><span>01</span> MOVEMENT CANVAS</div>", unsafe_allow_html=True)
if globe_mode:
    render_globe_map(payload, height=720)
else:
    render_flat_map(payload, height=720)

st.markdown("<div class='tl-section-rule'><span>02</span><b>THE NUMBERS BEHIND THE MAP</b></div>", unsafe_allow_html=True)
render_stats(filtered, trip_summary=trip_summary)

st.markdown(
    f"<div class='tl-footer'>TRACK LENS · {config.TIMEZONE_NAME} · DATA STAYS ON THIS MACHINE</div>",
    unsafe_allow_html=True,
)
