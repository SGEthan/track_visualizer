#!/usr/bin/env python3
"""将原始位置 CSV 清洗为 Track Lens 使用的 Parquet 数据。"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

import config


DEFAULT_INPUT = config.PROJECT_ROOT / "all_data.csv"

ALIASES = {
    "timestamp": "ts",
    "dataTime": "ts",
    "longitude": "lon",
    "latitude": "lat",
    "isBackForeground": "bg",
}
REQUIRED_INPUT = {"ts", "lon", "lat", "accuracy", "stepType"}
DEFAULT_COLUMNS = {
    "locType": 0,
    "heading": 0.0,
    "distance": 0.0,
    "bg": 0,
    "altitude": 0.0,
    "source": "gps",
}


def _haversine_segments_m(lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    """返回每个点到前一个点的距离，首项固定为 0。"""
    if len(lon) == 0:
        return np.array([], dtype=float)
    lon_rad = np.radians(lon.astype(float))
    lat_rad = np.radians(lat.astype(float))
    dlon = np.diff(lon_rad, prepend=lon_rad[0])
    dlat = np.diff(lat_rad, prepend=lat_rad[0])
    lat_prev = np.roll(lat_rad, 1)
    lat_prev[0] = lat_rad[0]
    a = np.sin(dlat / 2) ** 2 + np.cos(lat_prev) * np.cos(lat_rad) * np.sin(dlon / 2) ** 2
    return 6_371_000 * 2 * np.arctan2(np.sqrt(a), np.sqrt(np.maximum(1 - a, 0)))


def prepare_tracks(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """纯数据转换函数，便于测试和在其他脚本中复用。"""
    df = raw.rename(columns={key: value for key, value in ALIASES.items() if key in raw.columns}).copy()
    missing = sorted(REQUIRED_INPUT.difference(df.columns))
    if missing:
        raise ValueError(
            "输入 CSV 缺少必要列：" + ", ".join(missing)
            + "。时间列可使用 dataTime、timestamp 或 ts。"
        )

    for column, default in DEFAULT_COLUMNS.items():
        if column not in df.columns:
            df[column] = default

    numeric = ["ts", "lon", "lat", "accuracy", "stepType", "bg", "altitude"]
    for column in numeric:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df = df.dropna(subset=["ts", "lon", "lat", "accuracy", "stepType"])
    df = df[
        df["lon"].between(-180, 180)
        & df["lat"].between(-90, 90)
        & np.isfinite(df["lon"])
        & np.isfinite(df["lat"])
    ]

    is_photo = df["source"].fillna("gps").eq("photo")
    df = df[is_photo | (df["accuracy"] > 0)]
    df = (
        df.drop_duplicates(subset=["ts", "lon", "lat"])
        .sort_values("ts")
        .reset_index(drop=True)
    )
    if df.empty:
        raise ValueError("清理后没有有效轨迹点。")

    df["ts"] = df["ts"].astype("int64")
    distances = _haversine_segments_m(df["lon"].to_numpy(), df["lat"].to_numpy())
    gaps = df["ts"].diff().fillna(0).to_numpy()
    new_trip = (gaps > config.TRIP_GAP_SECONDS) | (distances > config.TRIP_MAX_JUMP_M)
    new_trip[0] = False
    df["trip_id"] = np.cumsum(new_trip).astype("int32")

    delta_seconds = df["ts"].diff().to_numpy(dtype=float)
    same_trip = df["trip_id"].to_numpy() == df["trip_id"].shift(1).to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        speed = np.where(
            same_trip & (delta_seconds > 0),
            distances / delta_seconds * 3.6,
            -1.0,
        )
    speed = np.where(np.isfinite(speed) & (speed <= 300), speed, -1.0)
    df["speed"] = speed.astype("float32")

    keep = (~same_trip) | (distances >= config.MIN_POINT_DIST_M)
    before_dedupe = len(df)
    df = df.loc[keep].reset_index(drop=True)

    # 统一稳定的数据类型。时间保留 int64，避免 2038 年溢出。
    df["lon"] = df["lon"].astype("float32")
    df["lat"] = df["lat"].astype("float32")
    df["accuracy"] = df["accuracy"].astype("float32")
    df["stepType"] = df["stepType"].astype("int8")
    df["bg"] = df["bg"].fillna(0).astype("int8")
    df["altitude"] = df["altitude"].fillna(0).astype("float32")
    df["source"] = df["source"].fillna("gps").astype("string")

    local_dates = (
        pd.to_datetime(df["ts"], unit="s", utc=True)
        .dt.tz_convert(config.DISPLAY_TIMEZONE)
        .dt.strftime("%Y-%m-%d")
    )
    daily = (
        df.assign(_date=local_dates)
        .groupby("_date")
        .agg(count=("ts", "size"), min_ts=("ts", "min"), max_ts=("ts", "max"))
    )
    daily_stats = daily.to_dict(orient="index")
    metadata = {
        "input_points": int(len(raw)),
        "output_points": int(len(df)),
        "removed_nearby": int(before_dedupe - len(df)),
        "trips": int(df["trip_id"].nunique()),
        "daily_stats": daily_stats,
    }
    return df, metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input",
        nargs="?",
        type=Path,
        default=DEFAULT_INPUT,
        help="输入 CSV，默认 all_data.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=config.DATA_DIR,
        help="输出目录，默认项目内 data/",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = args.input.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not input_path.exists():
        raise SystemExit(f"[ERROR] 找不到文件：{input_path}")

    started = time.perf_counter()
    print(f"[1/3] 读取 {input_path.name} ...")
    raw = pd.read_csv(input_path, low_memory=False)
    print(f"      {len(raw):,} 行")

    print("[2/3] 清理、切分行程并计算速度 ...")
    try:
        tracks, metadata = prepare_tracks(raw)
    except ValueError as exc:
        raise SystemExit(f"[ERROR] {exc}") from exc

    output_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = output_dir / "tracks.parquet"
    stats_path = output_dir / "daily_stats.json"

    print("[3/3] 写入 Parquet 和每日统计 ...")
    tracks.to_parquet(parquet_path, compression="zstd", index=False)
    with open(stats_path, "w", encoding="utf-8") as handle:
        json.dump(metadata["daily_stats"], handle, ensure_ascii=False)

    elapsed = time.perf_counter() - started
    size_mb = parquet_path.stat().st_size / 1_000_000
    print(
        f"完成：{metadata['output_points']:,} 点 · {metadata['trips']:,} 行程 · "
        f"{size_mb:.1f} MB · {elapsed:.1f}s"
    )
    print(f"时区：{config.TIMEZONE_NAME}")
    print(f"输出：{parquet_path}")


if __name__ == "__main__":
    main()
