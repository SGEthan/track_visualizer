"""数据加载、过滤与轨迹聚合工具。"""
from __future__ import annotations

import json
from datetime import date, datetime, time
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

import config


REQUIRED_TRACK_COLUMNS = {
    "ts", "lon", "lat", "speed", "accuracy", "stepType", "trip_id"
}


def _file_signature(path: Path) -> tuple[str, int, int]:
    stat = path.stat()
    return str(path), stat.st_mtime_ns, stat.st_size


def _validate_track_data(df: pd.DataFrame) -> pd.DataFrame:
    missing = sorted(REQUIRED_TRACK_COLUMNS.difference(df.columns))
    if missing:
        raise ValueError(
            "tracks.parquet 缺少必要列：" + ", ".join(missing)
            + "。请使用当前版本的 preprocess.py 重新生成。"
        )
    if df.empty:
        raise ValueError("tracks.parquet 中没有可用记录。")
    if not np.isfinite(df[["lon", "lat"]].to_numpy(dtype=float)).all():
        raise ValueError("tracks.parquet 包含非有限经纬度。")
    return df


@st.cache_data(show_spinner=False)
def _read_parquet(signature: tuple[str, int, int]) -> pd.DataFrame:
    path, _, _ = signature
    return _validate_track_data(pd.read_parquet(path))


def load_all_data() -> pd.DataFrame:
    """加载轨迹数据；文件发生变化时自动刷新 Streamlit 缓存。"""
    if not config.PARQUET_PATH.exists():
        st.error(
            "找不到 `data/tracks.parquet`，请先运行 `python preprocess.py`。",
            icon="⚠️",
        )
        st.stop()
    try:
        return _read_parquet(_file_signature(config.PARQUET_PATH))
    except (ValueError, OSError) as exc:
        st.error(f"轨迹数据无法加载：{exc}", icon="⚠️")
        st.stop()


@st.cache_data(show_spinner=False)
def _read_json(signature: tuple[str, int, int]):
    path, _, _ = signature
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def load_flight_data() -> list[dict]:
    if not config.FLIGHTS_PATH.exists():
        return []
    data = _read_json(_file_signature(config.FLIGHTS_PATH))
    return data if isinstance(data, list) else []


def load_daily_stats() -> dict:
    if not config.STATS_PATH.exists():
        return {}
    data = _read_json(_file_signature(config.STATS_PATH))
    return data if isinstance(data, dict) else {}


def to_display_datetime(values) -> pd.Series | pd.DatetimeIndex:
    """Unix 秒 → 应用配置时区，避免筛选、提示和统计使用不同日期。"""
    parsed = pd.to_datetime(values, unit="s", utc=True)
    if isinstance(parsed, pd.Series):
        return parsed.dt.tz_convert(config.DISPLAY_TIMEZONE)
    return parsed.tz_convert(config.DISPLAY_TIMEZONE)


def format_timestamps(values, fmt: str = "%Y-%m-%d %H:%M:%S") -> list[str]:
    return to_display_datetime(values).strftime(fmt).tolist()


def get_date_range(
    df: pd.DataFrame,
    flights: list[dict] | None = None,
) -> tuple[date, date]:
    """Return bounds covering both location history and imported flights."""
    bounds = to_display_datetime([int(df["ts"].min()), int(df["ts"].max())])
    track_min, track_max = bounds[0].date(), bounds[1].date()
    flight_dates: list[date] = []
    for flight in flights or []:
        try:
            flight_dates.append(date.fromisoformat(str(flight.get("date", ""))))
        except ValueError:
            continue
    if not flight_dates:
        return track_min, track_max
    return min(track_min, min(flight_dates)), max(track_max, max(flight_dates))


def filter_by_dates(df: pd.DataFrame, start: date, end: date) -> pd.DataFrame:
    start_dt = datetime.combine(start, time.min, tzinfo=config.DISPLAY_TIMEZONE)
    end_dt = datetime.combine(end, time.max, tzinfo=config.DISPLAY_TIMEZONE)
    start_ts = int(start_dt.timestamp())
    end_ts = int(end_dt.timestamp())
    ts = df["ts"].to_numpy()
    return df.loc[(ts >= start_ts) & (ts <= end_ts)]


def match_flight_windows(
    df: pd.DataFrame,
    flights: list[dict],
) -> pd.DataFrame:
    """Mark Footprint points recorded between each Flighty takeoff and landing.

    This is intentionally a review-only, time-based match: it does not mutate the
    track dataset and it does not assume every matching point should be deleted.
    """
    extra_columns = {
        "flight_match_id": pd.Series(dtype="string"),
        "flight_match_label": pd.Series(dtype="string"),
        "flight_match_route": pd.Series(dtype="string"),
        "flight_match_date": pd.Series(dtype="string"),
        "flight_dep_ts": pd.Series(dtype="int64"),
        "flight_arr_ts": pd.Series(dtype="int64"),
        "flight_time_basis": pd.Series(dtype="string"),
    }
    if df.empty or not flights:
        return df.iloc[0:0].assign(**extra_columns)

    ordered = df.sort_values("ts")
    timestamps = ordered["ts"].to_numpy(dtype=np.int64)
    matches: list[pd.DataFrame] = []

    for index, flight in enumerate(flights):
        if flight.get("canceled"):
            continue
        try:
            departure = int(flight.get("dep_ts"))
            arrival = int(flight.get("arr_ts"))
        except (TypeError, ValueError):
            continue
        if arrival <= departure or arrival - departure > 30 * 3600:
            continue

        left = int(np.searchsorted(timestamps, departure, side="left"))
        right = int(np.searchsorted(timestamps, arrival, side="right"))
        if right <= left:
            continue

        chunk = ordered.iloc[left:right].copy()
        match_id = str(flight.get("id") or f"flight-{index}")
        route = f"{flight.get('from_iata', '?')}→{flight.get('to_iata', '?')}"
        flight_number = f"{flight.get('airline', '')}{flight.get('flight', '')}".strip()
        date_text = str(flight.get("date", ""))
        label_parts = [part for part in (date_text, route, flight_number) if part]
        dep_basis = str(flight.get("dep_time_source") or "departure")
        arr_basis = str(flight.get("arr_time_source") or "arrival")

        chunk["flight_match_id"] = match_id
        chunk["flight_match_label"] = " · ".join(label_parts)
        chunk["flight_match_route"] = route
        chunk["flight_match_date"] = date_text
        chunk["flight_dep_ts"] = departure
        chunk["flight_arr_ts"] = arrival
        chunk["flight_time_basis"] = f"{dep_basis} → {arr_basis}"
        matches.append(chunk)

    if not matches:
        return df.iloc[0:0].assign(**extra_columns)

    result = pd.concat(matches).sort_values("ts")
    return result.loc[~result.index.duplicated(keep="first")]


def apply_filters(
    df: pd.DataFrame,
    max_accuracy: float,
    speed_min: float,
    speed_max: float,
    step_types: list[int],
) -> pd.DataFrame:
    accuracy = df["accuracy"].to_numpy()
    speed = df["speed"].to_numpy()
    mask = (accuracy <= max_accuracy) & (
        (speed < 0) | ((speed >= speed_min) & (speed <= speed_max))
    )
    if step_types:
        mask &= df["stepType"].isin(step_types).to_numpy()
    return df.loc[mask]


def _proportional_allocate(capacity: np.ndarray, slots: int) -> np.ndarray:
    """按容量比例分配整数配额，保证不超容量且尽量用满 slots。"""
    capacity = np.asarray(capacity, dtype=np.int64)
    result = np.zeros_like(capacity)
    slots = min(int(slots), int(capacity.sum()))
    if slots <= 0 or capacity.sum() <= 0:
        return result

    raw = capacity / capacity.sum() * slots
    result = np.minimum(np.floor(raw).astype(np.int64), capacity)
    remaining = slots - int(result.sum())
    if remaining:
        fractional = raw - np.floor(raw)
        order = np.lexsort((-capacity, -fractional))
        for idx in order:
            if remaining == 0:
                break
            if result[idx] < capacity[idx]:
                result[idx] += 1
                remaining -= 1
    if remaining:
        for idx in np.flatnonzero(result < capacity):
            take = min(remaining, int(capacity[idx] - result[idx]))
            result[idx] += take
            remaining -= take
            if remaining == 0:
                break
    return result


def downsample(df: pd.DataFrame, max_points: int) -> pd.DataFrame:
    """按行程公平抽样，并严格保证结果不超过 ``max_points``。"""
    if max_points <= 0:
        return df.iloc[0:0].copy()
    if len(df) <= max_points:
        return df
    ordered = df.sort_values("ts")
    if "trip_id" not in ordered.columns:
        indices = np.linspace(0, len(ordered) - 1, max_points, dtype=np.int64)
        return ordered.iloc[indices].copy()

    groups = list(ordered.groupby("trip_id", sort=False))
    sizes = np.array([len(group) for _, group in groups], dtype=np.int64)
    n_groups = len(groups)

    if n_groups > max_points:
        # 配额不足以覆盖所有行程时，优先保留点数最多的行程。
        quotas = np.zeros(n_groups, dtype=np.int64)
        quotas[np.argsort(-sizes, kind="stable")[:max_points]] = 1
    else:
        quotas = np.ones(n_groups, dtype=np.int64)
        remaining = max_points - n_groups

        preferred = np.maximum(np.minimum(sizes, 20) - 1, 0)
        first = _proportional_allocate(preferred, remaining)
        quotas += first
        remaining -= int(first.sum())

        if remaining:
            residual = np.maximum(sizes - quotas, 0)
            quotas += _proportional_allocate(residual, remaining)

    frames: list[pd.DataFrame] = []
    for (_, group), quota in zip(groups, quotas):
        if quota <= 0:
            continue
        if quota >= len(group):
            frames.append(group)
        else:
            indices = np.linspace(0, len(group) - 1, int(quota), dtype=np.int64)
            frames.append(group.iloc[indices])
    return pd.concat(frames).sort_values("ts").copy()


def _segment_distances_km(lons: np.ndarray, lats: np.ndarray) -> np.ndarray:
    if len(lons) < 2:
        return np.zeros(len(lons), dtype=float)
    lon1, lon2 = np.radians(lons[:-1]), np.radians(lons[1:])
    lat1, lat2 = np.radians(lats[:-1]), np.radians(lats[1:])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    segments = 6371.0 * 2 * np.arcsin(np.sqrt(a).clip(0, 1))
    return np.concatenate(([0.0], segments))


def exclude_airborne_points(
    df: pd.DataFrame,
    airborne_candidates: pd.DataFrame,
) -> pd.DataFrame:
    """Return regular ground history with confirmed in-flight points removed.

    The source frame is left untouched. Remaining points are segmented again so
    the positions before takeoff and after landing cannot become one long path.
    """
    if df.empty or airborne_candidates.empty:
        return df

    excluded_index = df.index.intersection(airborne_candidates.index)
    ground = df.drop(index=excluded_index).sort_values("ts").copy()
    if ground.empty:
        return ground

    timestamps = ground["ts"].to_numpy(dtype=np.int64)
    original_trips = ground["trip_id"].to_numpy()
    distances_m = _segment_distances_km(
        ground["lon"].to_numpy(dtype=float),
        ground["lat"].to_numpy(dtype=float),
    ) * 1000.0
    gaps = np.diff(timestamps, prepend=timestamps[0])
    new_segment = (
        (original_trips != np.roll(original_trips, 1))
        | (gaps > config.TRIP_GAP_SECONDS)
        | (distances_m > config.TRIP_MAX_JUMP_M)
    )
    new_segment[0] = True
    ground["trip_id"] = (np.cumsum(new_segment) - 1).astype("int32")
    with np.errstate(divide="ignore", invalid="ignore"):
        speed = np.where(
            (~new_segment) & (gaps > 0),
            distances_m / gaps * 3.6,
            -1.0,
        )
    speed = np.where(np.isfinite(speed) & (speed <= 300), speed, -1.0)
    ground["speed"] = speed.astype("float32")
    return ground


def build_trip_summary(df: pd.DataFrame) -> pd.DataFrame:
    """向量化计算行程摘要，避免逐行程执行 Python Haversine 循环。"""
    if df.empty:
        return pd.DataFrame(columns=[
            "trip_id", "start_ts", "end_ts", "duration_min",
            "point_count", "avg_speed", "distance_km",
        ])

    ordered = df.sort_values(["trip_id", "ts"]).copy()
    trips = ordered["trip_id"].to_numpy()
    segment_km = _segment_distances_km(
        ordered["lon"].to_numpy(dtype=float),
        ordered["lat"].to_numpy(dtype=float),
    )
    segment_km[1:][trips[1:] != trips[:-1]] = 0.0
    ordered["_segment_km"] = segment_km

    summary = (
        ordered.groupby("trip_id", sort=False)
        .agg(
            start_ts=("ts", "min"),
            end_ts=("ts", "max"),
            point_count=("ts", "size"),
            distance_km=("_segment_km", "sum"),
        )
        .reset_index()
    )
    duration_s = (summary["end_ts"] - summary["start_ts"]).clip(lower=0)
    summary["duration_min"] = (duration_s // 60).astype(int)
    duration_h = duration_s / 3600.0
    summary["avg_speed"] = np.where(
        duration_h > 0, summary["distance_km"] / duration_h, -1.0
    )
    return summary.sort_values("start_ts").reset_index(drop=True)


def build_path_data(df: pd.DataFrame) -> list[dict]:
    paths: list[dict] = []
    for trip_id, group in df.sort_values("ts").groupby("trip_id", sort=False):
        if len(group) < 2:
            continue
        valid_speed = group.loc[group["speed"] >= 0, "speed"]
        paths.append({
            "path": group[["lon", "lat"]].to_numpy().tolist(),
            "avg_speed": float(valid_speed.mean()) if len(valid_speed) else -1.0,
            "trip_id": int(trip_id),
            "point_count": len(group),
        })
    return paths


def _perp_offset_path(path: list, delta: float) -> list:
    """端点固定，以正弦曲线将重复航线侧向展开。"""
    if not path or delta == 0.0:
        return path
    lon0, lat0 = path[0]
    lon1, lat1 = path[-1]
    dx, dy = lon1 - lon0, lat1 - lat0
    length = float(np.hypot(dx, dy))
    if length < 1e-9:
        return path
    px, py = -dy / length * delta, dx / length * delta
    if len(path) < 3:
        return [path[0], [(lon0 + lon1) / 2 + px, (lat0 + lat1) / 2 + py], path[-1]]
    return [
        [point[0] + px * np.sin(i / (len(path) - 1) * np.pi),
         point[1] + py * np.sin(i / (len(path) - 1) * np.pi)]
        for i, point in enumerate(path)
    ]


def spread_flight_paths(flights: list[dict]) -> list[dict]:
    groups: dict[tuple[str, str], list[int]] = {}
    for index, flight in enumerate(flights):
        key = tuple(sorted((flight.get("from_iata", ""), flight.get("to_iata", ""))))
        groups.setdefault(key, []).append(index)

    result = list(flights)
    for indices in groups.values():
        count = len(indices)
        if count <= 1:
            continue
        distance = flights[indices[0]].get("distance_km", 2000)
        step = min(max(0.15, min(1.2, distance / 2500)), 2.5 / (count - 1))
        for rank, index in enumerate(indices):
            delta = (rank - (count - 1) / 2.0) * step
            result[index] = {
                **flights[index],
                "path": _perp_offset_path(flights[index]["path"], delta),
            }
    return result
