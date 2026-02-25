#!/usr/bin/env python3
"""
将照片轨迹合并进主 GPS 轨迹，仅填补 gap（>300s 无 GPS 点的时段）。
输出：all_data.csv
"""
import numpy as np
import pandas as pd

GAP_THRESHOLD = 300   # 秒，超过此间隔视为轨迹空白
ORIG_CSV   = "backUpData-all.csv"
PHOTO_CSV  = "backUpPhotoData.csv"
OUTPUT_CSV = "all_data.csv"


def main():
    # ── 1. 读取数据 ──────────────────────────────────────────────────────
    print("读取主轨迹...")
    orig = pd.read_csv(ORIG_CSV, low_memory=False)
    print(f"  {len(orig):,} 行")

    print("读取照片轨迹...")
    photo = pd.read_csv(PHOTO_CSV, low_memory=False)
    print(f"  {len(photo):,} 行")

    # 统一时间戳列名
    orig  = orig.rename(columns={"dataTime": "ts"})
    photo = photo.rename(columns={"dataTime": "ts"})

    # ── 2. 找出照片点中落在 GPS gap 里的 ────────────────────────────────
    orig_ts = np.sort(orig["ts"].values.astype(np.int64))

    photo_ts = photo["ts"].values.astype(np.int64)

    # 对每个照片时间戳，找主轨迹中最近邻的时间点
    # searchsorted 返回照片点在主轨迹时间轴中的插入位置
    idx = np.searchsorted(orig_ts, photo_ts)

    # 左邻距离（到前一个 GPS 点）
    left_idx  = np.clip(idx - 1, 0, len(orig_ts) - 1)
    left_dist = np.abs(photo_ts - orig_ts[left_idx])

    # 右邻距离（到后一个 GPS 点）
    right_idx  = np.clip(idx, 0, len(orig_ts) - 1)
    right_dist = np.abs(orig_ts[right_idx] - photo_ts)

    # 最近邻距离
    nearest = np.minimum(left_dist, right_dist)

    # 只保留最近 GPS 点距离 > GAP_THRESHOLD 的照片点
    in_gap = nearest > GAP_THRESHOLD
    photo_fill = photo[in_gap].copy()
    photo_fill["source"] = "photo"

    print(f"\n照片点中落在 GPS gap 的: {in_gap.sum():,} / {len(photo):,}")

    # ── 3. 合并 & 去重 ────────────────────────────────────────────────────
    orig["source"] = "gps"

    merged = pd.concat([orig, photo_fill], ignore_index=True)
    merged = merged.sort_values("ts").reset_index(drop=True)

    # 去除完全重复行（相同时间戳+坐标）
    before = len(merged)
    merged = merged.drop_duplicates(subset=["ts", "longitude", "latitude"])
    print(f"去重: {before - len(merged)} 行 → 剩余 {len(merged):,} 行")

    # 恢复原始列名
    merged = merged.rename(columns={"ts": "dataTime"})

    # 输出列顺序与原始 CSV 一致，source 列放最后
    orig_cols = ["dataTime", "locType", "longitude", "latitude",
                 "heading", "accuracy", "speed", "distance",
                 "isBackForeground", "stepType", "altitude", "source"]
    merged = merged[orig_cols]

    # ── 4. 写入 CSV ───────────────────────────────────────────────────────
    merged.to_csv(OUTPUT_CSV, index=False)
    print(f"\n✓ 输出: {OUTPUT_CSV}")
    print(f"  总行数: {len(merged):,}")
    print(f"  GPS 点:  {(merged['source'] == 'gps').sum():,}")
    print(f"  照片补充: {(merged['source'] == 'photo').sum():,}")
    start = pd.to_datetime(merged["dataTime"].min(), unit="s").date()
    end   = pd.to_datetime(merged["dataTime"].max(), unit="s").date()
    print(f"  时间范围: {start} ~ {end}")


if __name__ == "__main__":
    main()
