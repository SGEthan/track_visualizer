#!/usr/bin/env python3
"""
一次性预处理脚本：CSV → Parquet + JSON stats
运行方式：python preprocess.py
耗时约 10-30 秒（取决于硬件）
"""
import os
import sys
import json
import time
from collections import defaultdict
from datetime import datetime, timezone

import pandas as pd
import numpy as np

import config

CSV_PATH     = "all_data.csv"
DATA_DIR     = "data"
PARQUET_PATH = os.path.join(DATA_DIR, "tracks.parquet")
STATS_PATH   = os.path.join(DATA_DIR, "daily_stats.json")

TRIP_GAP = config.TRIP_GAP_SECONDS  # 从 config.py 统一读取


def assign_trip_ids(df: pd.DataFrame) -> pd.Series:
    """按时间戳排序后，以 >30 分钟间隔划分行程 ID。"""
    df_sorted = df.sort_values("ts")
    gaps = df_sorted["ts"].diff().fillna(0)
    new_trip = gaps > TRIP_GAP
    trip_id = new_trip.cumsum().astype("int32")
    # 恢复原始索引顺序
    return trip_id.reindex(df.index)


def main():
    t0 = time.time()

    if not os.path.exists(CSV_PATH):
        print(f"[ERROR] 找不到文件: {CSV_PATH}")
        sys.exit(1)

    os.makedirs(DATA_DIR, exist_ok=True)

    # ── 1. 读取 CSV ───────────────────────────────────────────────────────
    print(f"[1/4] 读取 {CSV_PATH} ...")
    df = pd.read_csv(
        CSV_PATH,
        dtype={
            "dataTime":         "int64",
            "locType":          "int8",
            "longitude":        "float64",
            "latitude":         "float64",
            "heading":          "float32",
            "accuracy":         "float32",
            "speed":            "float32",
            "distance":         "float32",
            "isBackForeground": "int8",
            "stepType":         "int8",
            "altitude":         "float32",
            "source":           "str",    # all_data.csv 特有列，原始CSV无此列
        },
        low_memory=False,
    )
    print(f"    共 {len(df):,} 行，用时 {time.time()-t0:.1f}s")

    # ── 2. 清理 & 重命名 ─────────────────────────────────────────────────
    print("[2/4] 数据清理 ...")
    df = df.rename(columns={
        "dataTime":         "ts",
        "longitude":        "lon",
        "latitude":         "lat",
        "isBackForeground": "bg",
    })

    # 过滤掉明显无效坐标
    df = df[(df["lon"].between(-180, 180)) & (df["lat"].between(-90, 90))]
    # accuracy=0 对 GPS 点是无效记录，但照片补充点本身就是 accuracy=0，予以保留
    is_photo = df.get("source", pd.Series("gps", index=df.index)) == "photo"
    df = df[is_photo | (df["accuracy"] > 0)]
    df = df.drop_duplicates(subset=["ts", "lon", "lat"])

    # ── 3. 分配行程 ID（时间间隔 + 距离跳跃双重检测）────────────────────
    print("[3/4] 分配行程 ID ...")
    df = df.sort_values("ts").reset_index(drop=True)

    # 先用 Haversine 计算相邻点距离，用于检测 GPS 坐标跳跃
    _lon1 = np.radians(df["lon"].shift(1).values)
    _lat1 = np.radians(df["lat"].shift(1).values)
    _lon2 = np.radians(df["lon"].values)
    _lat2 = np.radians(df["lat"].values)
    _dlat = _lat2 - _lat1; _dlon = _lon2 - _lon1
    _a = np.sin(_dlat/2)**2 + np.cos(_lat1)*np.cos(_lat2)*np.sin(_dlon/2)**2
    _dist_m_raw = 6_371_000 * 2 * np.arctan2(np.sqrt(_a), np.sqrt(1 - _a))
    _dist_m_raw[0] = 0  # 第一行无前驱

    gaps = df["ts"].diff().fillna(0)
    MAX_JUMP_M = config.TRIP_MAX_JUMP_M   # 超过此距离视为坐标跳跃，强制新行程
    new_trip = (gaps > TRIP_GAP) | (_dist_m_raw > MAX_JUMP_M)
    df["trip_id"] = new_trip.cumsum().astype("int32")

    n_trips = df["trip_id"].nunique()
    print(f"    共 {n_trips:,} 条行程")

    # ── 3b. 用坐标差分重新计算速度 ──────────────────────────────────────
    print("[3b] 用坐标插值重新计算速度 ...")

    # Haversine 向量化：返回相邻行之间的距离（米）
    lon1 = np.radians(df["lon"].shift(1).values)
    lat1 = np.radians(df["lat"].shift(1).values)
    lon2 = np.radians(df["lon"].values)
    lat2 = np.radians(df["lat"].values)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    dist_m = 6_371_000 * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    dt_s = df["ts"].diff().values.astype("float64")           # 时间差（秒）
    same_trip = (df["trip_id"].values == df["trip_id"].shift(1).values)  # 同行程

    # 速度 = 位移 / 时间，跨行程边界或 dt=0 时置 -1（未知）
    with np.errstate(divide="ignore", invalid="ignore"):
        spd = np.where(same_trip & (dt_s > 0), (dist_m / dt_s) * 3.6, -1.0)

    # 限制合理上限（300 km/h），超过的视为 GPS 噪声，置 -1
    spd = np.where(spd > 300, -1.0, spd)

    df["speed"] = spd.astype("float32")

    # ── 3c. 距离去重：丢弃相邻位移 < MIN_POINT_DIST_M 的冗余静止点 ────
    MIN_D = config.MIN_POINT_DIST_M
    print(f"[3c] 距离去重（阈值 {MIN_D} m）...")
    before = len(df)
    # 行程内位移 < 阈值的点视为冗余；行程边界第一个点始终保留（same_trip=False）
    keep_mask = (~same_trip) | (dist_m >= MIN_D)
    df = df[keep_mask].reset_index(drop=True)
    removed = before - len(df)
    print(f"    去重后：{len(df):,} 行（减少 {removed:,}，{100*removed/before:.1f}%）")

    # ── 4. 计算每日统计 ──────────────────────────────────────────────────
    print("[4/4] 计算每日统计并写入磁盘 ...")
    df["date"] = pd.to_datetime(df["ts"], unit="s").dt.strftime("%Y-%m-%d")

    daily = (
        df.groupby("date")
        .agg(
            count=("ts", "count"),
            min_ts=("ts", "min"),
            max_ts=("ts", "max"),
        )
        .reset_index()
    )
    daily_dict = daily.set_index("date").to_dict(orient="index")
    with open(STATS_PATH, "w") as f:
        json.dump(daily_dict, f)
    print(f"    每日统计写入 {STATS_PATH}")

    # 删除辅助列，降低内存占用
    df = df.drop(columns=["date"])

    # 压缩数据类型
    df["ts"]       = df["ts"].astype("int32")
    df["lon"]      = df["lon"].astype("float32")
    df["lat"]      = df["lat"].astype("float32")
    df["speed"]    = df["speed"].astype("float32")
    df["accuracy"] = df["accuracy"].astype("float32")
    df["stepType"] = df["stepType"].astype("int8")
    df["bg"]       = df["bg"].astype("int8")
    df["altitude"] = df["altitude"].astype("float32")

    # 写 Parquet
    df.to_parquet(PARQUET_PATH, compression="zstd", index=False)
    size_mb = os.path.getsize(PARQUET_PATH) / 1e6
    print(f"    Parquet 写入 {PARQUET_PATH}（{size_mb:.1f} MB）")

    elapsed = time.time() - t0
    print(f"\n完成！共用时 {elapsed:.1f}s")
    print(f"  行数：{len(df):,}")
    print(f"  行程：{n_trips:,}")
    print(f"  日期：{df['ts'].apply(lambda x: pd.Timestamp(x, unit='s').strftime('%Y-%m-%d')).min()} ~ "
          f"{df['ts'].apply(lambda x: pd.Timestamp(x, unit='s').strftime('%Y-%m-%d')).max()}")


if __name__ == "__main__":
    main()
