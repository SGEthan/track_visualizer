"""数据加载与过滤工具（带 Streamlit 缓存）。"""
from __future__ import annotations

import json
import os
from datetime import date, datetime

import numpy as np
import pandas as pd
import streamlit as st

PARQUET_PATH  = "data/tracks.parquet"
STATS_PATH    = "data/daily_stats.json"
FLIGHTS_PATH  = "data/flight_tracks.json"


@st.cache_data(show_spinner=False)
def load_all_data() -> pd.DataFrame:
    """加载全量 Parquet 数据，会话内常驻内存。"""
    if not os.path.exists(PARQUET_PATH):
        st.error(
            "找不到 `data/tracks.parquet`，请先运行 `python preprocess.py`。",
            icon="⚠️",
        )
        st.stop()
    return pd.read_parquet(PARQUET_PATH)


@st.cache_data(show_spinner=False)
def load_flight_data() -> list[dict]:
    """加载航班轨迹数据，未运行 preprocess_flights.py 时返回空列表。"""
    if not os.path.exists(FLIGHTS_PATH):
        return []
    with open(FLIGHTS_PATH, encoding="utf-8") as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def load_daily_stats() -> dict:
    if not os.path.exists(STATS_PATH):
        return {}
    with open(STATS_PATH) as f:
        return json.load(f)


def get_date_range(df: pd.DataFrame) -> tuple[date, date]:
    """返回数据集的最早/最晚日期。"""
    min_ts = int(df["ts"].min())
    max_ts = int(df["ts"].max())
    return (
        datetime.fromtimestamp(min_ts).date(),
        datetime.fromtimestamp(max_ts).date(),
    )


def filter_by_dates(
    df: pd.DataFrame,
    start: date,
    end: date,
) -> pd.DataFrame:
    start_ts = int(datetime.combine(start, datetime.min.time()).timestamp())
    end_ts   = int(datetime.combine(end,   datetime.max.time()).timestamp())
    mask = (df["ts"] >= start_ts) & (df["ts"] <= end_ts)
    return df[mask]


def apply_filters(
    df: pd.DataFrame,
    max_accuracy: float,
    speed_min: float,
    speed_max: float,
    step_types: list[int],
) -> pd.DataFrame:
    out = df[df["accuracy"] <= max_accuracy]
    speed_mask = (out["speed"] < 0) | (
        (out["speed"] >= speed_min) & (out["speed"] <= speed_max)
    )
    out = out[speed_mask]
    if step_types:
        out = out[out["stepType"].isin(step_types)]
    return out


def downsample(df: pd.DataFrame, max_points: int) -> pd.DataFrame:
    """
    按行程比例分配点数配额，避免小行程被全局步长抽样至消失。
    每条行程至少保留 MIN_PER_TRIP 个点，剩余配额按行程大小比例分配。
    """
    if len(df) <= max_points:
        return df

    MIN_PER_TRIP = 20  # 每条行程保底点数

    if "trip_id" not in df.columns:
        step = max(1, len(df) // max_points)
        return df.sort_values("ts").iloc[::step].copy()

    df_sorted = df.sort_values("ts")
    trip_sizes = df_sorted.groupby("trip_id", sort=False).size()
    n_trips = len(trip_sizes)

    # 保底分配：每条行程 MIN_PER_TRIP 个点
    base_total = min(MIN_PER_TRIP * n_trips, max_points)
    remaining  = max(0, max_points - base_total)
    total_pts  = len(df)

    frames = []
    for trip_id, grp in df_sorted.groupby("trip_id", sort=False):
        n = len(grp)
        # 按行程大小占总点数的比例分配剩余配额
        quota = MIN_PER_TRIP + int(remaining * n / total_pts)
        quota = min(n, quota)  # 不超过实际点数
        if n <= quota:
            frames.append(grp)
        else:
            step = max(1, n // quota)
            frames.append(grp.iloc[::step])

    return pd.concat(frames).sort_values("ts")


def build_photo_gap_paths(df: pd.DataFrame) -> list[dict]:
    """
    找出 GPS 空白段（相邻 GPS 点间距 >300s），将落在其中的照片点按时序连成路径。
    返回与 build_path_data 相同格式（只包含 ≥2 个照片点的段）。
    """
    if "source" not in df.columns:
        return []

    df_sorted = df.sort_values("ts")
    gps_ts  = df_sorted.loc[df_sorted["source"] != "photo", "ts"].values.astype(np.int64)
    photo_df = df_sorted[df_sorted["source"] == "photo"].reset_index(drop=True)

    if photo_df.empty or len(gps_ts) == 0:
        # 全部都是照片点时，当作一整段处理
        coords = photo_df[["lon", "lat"]].values.tolist()
        if len(coords) >= 2:
            return [{"path": coords, "avg_speed": -1.0, "trip_id": -1, "point_count": len(coords)}]
        return []

    photo_ts = photo_df["ts"].values.astype(np.int64)

    # 对每个照片点，用二分查找定位它在 GPS 时间轴中的插入位置
    ins = np.searchsorted(gps_ts, photo_ts)

    # 左侧最近 GPS 点距离
    left_dist = np.where(
        ins == 0,
        np.inf,
        photo_ts - gps_ts[np.clip(ins - 1, 0, len(gps_ts) - 1)],
    )
    # 右侧最近 GPS 点距离
    right_dist = np.where(
        ins >= len(gps_ts),
        np.inf,
        gps_ts[np.clip(ins, 0, len(gps_ts) - 1)] - photo_ts,
    )

    GAP = 300  # 秒

    # 只保留两侧 GPS 点距离都 > GAP 的照片点（确实在 GPS 空白段内）
    in_gap = (left_dist > GAP) & (right_dist > GAP)

    # 每个照片点所属 gap 的 key：(左侧GPS时间戳, 右侧GPS时间戳)
    left_anchor  = np.where(ins == 0,           -1, gps_ts[np.clip(ins - 1, 0, len(gps_ts) - 1)])
    right_anchor = np.where(ins >= len(gps_ts), -1, gps_ts[np.clip(ins,     0, len(gps_ts) - 1)])

    gap_groups: dict[tuple, list[int]] = {}
    for i in range(len(photo_df)):
        if not in_gap[i]:
            continue
        key = (int(left_anchor[i]), int(right_anchor[i]))
        gap_groups.setdefault(key, []).append(i)

    paths = []
    for indices in gap_groups.values():
        if len(indices) < 2:
            continue
        grp = photo_df.iloc[sorted(indices)]
        coords = grp[["lon", "lat"]].values.tolist()
        paths.append({
            "path": coords,
            "avg_speed": -1.0,
            "trip_id": -1,
            "point_count": len(coords),
        })

    return paths


def _haversine_km(lons: np.ndarray, lats: np.ndarray) -> float:
    """相邻点 Haversine 距离之和（km）。仅在无速度数据时作为后备。"""
    if len(lons) < 2:
        return 0.0
    R = 6371.0
    lon1, lon2 = np.radians(lons[:-1]), np.radians(lons[1:])
    lat1, lat2 = np.radians(lats[:-1]), np.radians(lats[1:])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return float(R * 2 * np.arcsin(np.sqrt(a).clip(0, 1)).sum())


def build_trip_summary(df: pd.DataFrame) -> pd.DataFrame:
    """按 trip_id 聚合，返回行程摘要 DataFrame，按开始时间排序。"""
    if df.empty:
        return pd.DataFrame()
    rows = []
    for trip_id, grp in df.groupby("trip_id"):
        grp_sorted = grp.sort_values("ts")
        ts_arr  = grp_sorted["ts"].values
        spd_arr = grp_sorted["speed"].values   # km/h，-1 表示未知
        ts_min = int(ts_arr[0])
        ts_max = int(ts_arr[-1])
        duration_min = max(0, (ts_max - ts_min) // 60)

        # Haversine 累加实际位移距离
        dist_km = _haversine_km(grp_sorted["lon"].values, grp_sorted["lat"].values)
        # 平均速度 = 距离 ÷ 时长（比 GPS 速度列均值更准，不受信号抖动影响）
        duration_h = (ts_max - ts_min) / 3600
        avg_spd = dist_km / duration_h if duration_h > 0 else -1.0

        rows.append({
            "trip_id":      trip_id,
            "start_ts":     ts_min,
            "duration_min": duration_min,
            "point_count":  len(grp),
            "avg_speed":    avg_spd,
            "distance_km":  dist_km,
        })
    return pd.DataFrame(rows).sort_values("start_ts").reset_index(drop=True)


def build_path_data(df: pd.DataFrame) -> list[dict]:
    """将 DataFrame 按 trip_id 分组，构建 PathLayer 所需的列表。"""
    paths = []
    for trip_id, grp in df.sort_values("ts").groupby("trip_id", sort=False):
        coords = grp[["lon", "lat"]].values.tolist()
        if len(coords) < 2:
            continue
        valid_speed = grp.loc[grp["speed"] >= 0, "speed"]
        avg_speed = float(valid_speed.mean()) if len(valid_speed) else -1.0
        paths.append({
            "path": coords,
            "avg_speed": avg_speed,
            "trip_id": int(trip_id),
            "point_count": len(coords),
        })
    return paths


def build_path_data_pacific_copy(df: pd.DataFrame) -> list[dict]:
    """
    为西半球（avg_lon < -60°）的行程生成 +360° 经度副本。

    当用户将地图视角移至太平洋（中心 ~180°，可视范围约 90°-270°），
    美国轨迹的原始坐标（~-120°）位于视口之外。
    通过将这些行程的经度整体 +360°（变为 ~240°），
    它们就落入太平洋视口的可视范围内，从而实现两侧同时可见。

    注意：返回的路径层应使用 wrap_longitude=False，
    防止 deck.gl 将 240° 标准化回 -120°。
    """
    paths = []
    for trip_id, grp in df.sort_values("ts").groupby("trip_id", sort=False):
        coords = grp[["lon", "lat"]].values.tolist()
        if len(coords) < 2:
            continue
        avg_lon = sum(c[0] for c in coords) / len(coords)
        if avg_lon > -60:  # 不是西半球行程，跳过
            continue
        valid_speed = grp.loc[grp["speed"] >= 0, "speed"]
        avg_speed = float(valid_speed.mean()) if len(valid_speed) else -1.0
        paths.append({
            "path": [[c[0] + 360, c[1]] for c in coords],
            "avg_speed": avg_speed,
            "trip_id": int(trip_id),
            "point_count": len(coords),
        })
    return paths


# ── 航班路径平行偏移（同一航线多次飞行时展开显示）─────────────────────────────
import math as _math
from collections import defaultdict as _defaultdict


def _perp_offset_path(path: list, delta: float) -> list:
    """
    将路径向垂直于航线方向弧形偏移 delta 度。
    首尾端点（机场）保持原位，中间段用 sin 曲线混合偏移量，
    使各条平行航线从同一机场出发后自然散开再收拢。
    对只有 1-2 个点的路径，自动插入弧顶中间点。
    """
    if not path or delta == 0.0:
        return path
    lon0, lat0 = path[0]
    lon1, lat1 = path[-1]
    dx = lon1 - lon0
    dy = lat1 - lat0
    length = _math.hypot(dx, dy)
    if length < 1e-9:
        return path
    # 垂直向量（旋转 90°，归一化后乘以偏移量）
    px = -dy / length * delta
    py =  dx / length * delta

    n = len(path)
    if n < 3:
        # 只有端点时插入弧顶中间点
        mid = [(lon0 + lon1) / 2 + px, (lat0 + lat1) / 2 + py]
        return [path[0], mid, path[-1]]

    # 多点路径：sin 曲线混合，端点 blend=0，中点 blend=1
    result = []
    for i, pt in enumerate(path):
        blend = _math.sin(i / (n - 1) * _math.pi)
        result.append([pt[0] + px * blend, pt[1] + py * blend])
    return result


def spread_flight_paths(flights: list[dict]) -> list[dict]:
    """
    对同一航线（from_iata ↔ to_iata）的多次飞行施加侧向偏移，
    使它们在地图上平行排列而非完全重叠。
    单次飞行的航线不受影响。
    """
    groups: dict[tuple, list[int]] = _defaultdict(list)
    for idx, f in enumerate(flights):
        key = tuple(sorted([f.get("from_iata", ""), f.get("to_iata", "")]))
        groups[key].append(idx)

    result = list(flights)
    for indices in groups.values():
        n = len(indices)
        if n <= 1:
            continue
        dist_km = flights[indices[0]].get("distance_km", 2000)
        _MAX_TOTAL_WIDTH = 2.5   # 整条航线最大总展宽（度），超过此值改为压缩间距
        step = max(0.15, min(1.2, dist_km / 2500))
        if n > 1:
            step = min(step, _MAX_TOTAL_WIDTH / (n - 1))
        for rank, idx in enumerate(indices):
            delta = (rank - (n - 1) / 2.0) * step
            f = flights[idx]
            result[idx] = {**f, "path": _perp_offset_path(f["path"], delta)}

    return result
