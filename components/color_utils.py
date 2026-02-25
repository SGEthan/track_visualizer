"""颜色工具：将速度/精度/时段映射为 RGBA 列表。"""
from __future__ import annotations

import numpy as np
import pandas as pd

# ── 向量化颜色计算（numpy，比 .apply() 快 50-100x）────────────────────────────
_VEC_STOP_SPEEDS = np.array([0.0, 5.0, 20.0, 60.0, 100.0, 150.0], dtype=np.float32)
_VEC_STOP_COLORS = np.array([
    [50,  230,  80, 210],
    [0,   255, 200, 220],
    [0,   180, 255, 230],
    [255, 220,   0, 240],
    [255, 120,   0, 250],
    [255,  30,  30, 255],
], dtype=np.float32)


def _speed_rgba_vec(speeds: np.ndarray) -> np.ndarray:
    out = np.empty((len(speeds), 4), dtype=np.int32)
    out[:] = [120, 120, 120, 100]
    valid = speeds >= 0
    if not valid.any():
        return out
    sv = np.clip(speeds[valid], 0.0, 150.0)
    seg = np.clip(
        np.searchsorted(_VEC_STOP_SPEEDS[1:], sv, side="right"),
        0, len(_VEC_STOP_SPEEDS) - 2,
    )
    t = np.clip(
        (sv - _VEC_STOP_SPEEDS[seg]) / np.maximum(_VEC_STOP_SPEEDS[seg + 1] - _VEC_STOP_SPEEDS[seg], 1e-6),
        0.0, 1.0,
    )[:, None]
    out[valid] = np.round(_VEC_STOP_COLORS[seg] + t * (_VEC_STOP_COLORS[seg + 1] - _VEC_STOP_COLORS[seg])).astype(np.int32)
    return out


def color_column_vec(df: pd.DataFrame, mode: str) -> np.ndarray:
    """向量化版本，返回 (n, 4) int32 numpy 数组，比 color_column() 快 50-100x。"""
    n = len(df)
    if mode == "速度":
        out = _speed_rgba_vec(df["speed"].values)
    elif mode == "精度":
        accs = df["accuracy"].values
        out = np.empty((n, 4), dtype=np.int32)
        out[accs <= 5]                  = [0,   255, 100, 230]
        out[(accs > 5) & (accs <= 15)] = [0,   200, 255, 210]
        out[(accs > 15) & (accs <= 40)]= [255, 200,   0, 180]
        out[accs > 40]                 = [255,  60,  60, 150]
    elif mode == "时段":
        hours = pd.to_datetime(df["ts"], unit="s").dt.hour.values
        out = np.empty((n, 4), dtype=np.int32)
        out[hours < 6]                   = [40,  40,  200, 200]
        out[(hours >= 6) & (hours < 12)] = [0,   200, 255, 210]
        out[(hours >= 12) & (hours < 18)]= [255, 220,   0, 210]
        out[hours >= 18]                 = [160,   0, 255, 210]
    elif mode == "活动类型":
        st = df["stepType"].values
        out = np.empty((n, 4), dtype=np.int32)
        out[st == 1] = [0, 255, 120, 220]
        out[st != 1] = [0, 160, 255, 220]
    else:
        out = _speed_rgba_vec(df["speed"].values)
    if "source" in df.columns:
        photo_mask = (df["source"] == "photo").values
        if photo_mask.any():
            out[photo_mask] = [180, 80, 255, 230]
    return out

# ── 速度色阶（km/h → RGBA）──────────────────────────────────────────────────
# 赛博朋克配色：静止灰 → 步行绿 → 低速青 → 驾驶蓝 → 高速黄 → 橙 → 红
_SPEED_STOPS: list[tuple[float, list[int]]] = [
    (-1,  [120, 120, 120, 100]),   # 未知/静止
    (0,   [50,  230,  80, 210]),   # 0 km/h
    (5,   [0,   255, 200, 220]),   # 步行
    (20,  [0,   180, 255, 230]),   # 慢速
    (60,  [255, 220,   0, 240]),   # 驾车
    (100, [255, 120,   0, 250]),   # 快速
    (150, [255,  30,  30, 255]),   # 超速
]


def _lerp_color(c0: list[int], c1: list[int], t: float) -> list[int]:
    return [int(c0[i] + t * (c1[i] - c0[i])) for i in range(4)]


def speed_to_rgba(speed: float) -> list[int]:
    """单个速度值 → RGBA 列表。"""
    if speed < 0:
        return _SPEED_STOPS[0][1]
    for i in range(len(_SPEED_STOPS) - 1):
        v0, c0 = _SPEED_STOPS[i]
        v1, c1 = _SPEED_STOPS[i + 1]
        if v0 <= speed <= v1:
            t = (speed - v0) / max(v1 - v0, 1e-6)
            return _lerp_color(c0, c1, t)
    return _SPEED_STOPS[-1][1]


def accuracy_to_rgba(acc: float) -> list[int]:
    if acc <= 5:
        return [0, 255, 100, 230]
    if acc <= 15:
        return [0, 200, 255, 210]
    if acc <= 40:
        return [255, 200,   0, 180]
    return [255,  60,  60, 150]


def hour_to_rgba(hour: int) -> list[int]:
    if hour < 6:
        return [40,  40, 200, 200]   # 深夜：深蓝
    if hour < 12:
        return [0,  200, 255, 210]   # 上午：青
    if hour < 18:
        return [255, 220,  0, 210]   # 下午：黄
    return [160,  0, 255, 210]       # 夜晚：紫


def activity_to_rgba(step_type: int) -> list[int]:
    return [0, 255, 120, 220] if step_type == 1 else [0, 160, 255, 220]


# 照片补充点专属颜色：紫色，与机场金黄色明显区分
_PHOTO_COLOR = [180, 80, 255, 230]   # 亮紫


def _is_photo(df: pd.DataFrame) -> pd.Series:
    """返回布尔 Series，标记照片来源点。"""
    if "source" in df.columns:
        return df["source"] == "photo"
    return pd.Series(False, index=df.index)


# ── 向量化版本（用于 DataFrame 列）──────────────────────────────────────────

def color_column(df: pd.DataFrame, mode: str) -> list[list[int]]:
    """返回每行 RGBA 的列表，供 PyDeck 使用。照片点始终显示为琥珀色。"""
    photo_mask = _is_photo(df)

    if mode == "速度":
        colors = df["speed"].apply(speed_to_rgba).tolist()
    elif mode == "精度":
        colors = df["accuracy"].apply(accuracy_to_rgba).tolist()
    elif mode == "时段":
        hours = pd.to_datetime(df["ts"], unit="s").dt.hour
        colors = hours.apply(hour_to_rgba).tolist()
    elif mode == "活动类型":
        colors = df["stepType"].apply(activity_to_rgba).tolist()
    else:
        colors = df["speed"].apply(speed_to_rgba).tolist()

    # 照片点覆盖为琥珀色
    for i, is_p in enumerate(photo_mask):
        if is_p:
            colors[i] = _PHOTO_COLOR

    return colors


def path_color(avg_speed: float) -> list[int]:
    """行程平均速度 → PathLayer 颜色。"""
    return speed_to_rgba(avg_speed)
