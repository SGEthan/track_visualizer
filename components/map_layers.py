"""PyDeck layer 工厂函数。"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pydeck as pdk

from .color_utils import color_column, path_color, _PHOTO_COLOR


# ── Heatmap ──────────────────────────────────────────────────────────────────
def make_heatmap_layer(df: pd.DataFrame) -> pdk.Layer:
    data = df[["lon", "lat"]].rename(columns={"lon": "longitude", "lat": "latitude"})
    return pdk.Layer(
        "HeatmapLayer",
        data=data,
        get_position=["longitude", "latitude"],
        get_weight=1,
        radius_pixels=30,
        intensity=1.2,
        threshold=0.05,
        color_range=[
            [80,   0, 180,  80],
            [0,   80, 255, 120],
            [0,  220, 200, 160],
            [255, 220,   0, 200],
            [255, 100,   0, 230],
            [255, 255, 255, 255],
        ],
        aggregation="SUM",
    )


# ── Scatter ───────────────────────────────────────────────────────────────────
def make_scatter_layer(df: pd.DataFrame, color_by: str) -> pdk.Layer:
    df = df.copy()
    df["color"]  = color_column(df, color_by)
    df["radius"] = np.clip(df["accuracy"] * 0.4, 2.0, 25.0)
    df["ts_fmt"] = pd.to_datetime(df["ts"], unit="s").dt.strftime("%Y-%m-%d %H:%M:%S")
    df["speed_fmt"] = df["speed"].apply(
        lambda s: f"{s:.1f} km/h" if s >= 0 else "未知"
    )

    return pdk.Layer(
        "ScatterplotLayer",
        data=df[["lon", "lat", "color", "radius", "ts_fmt", "speed_fmt", "accuracy"]].rename(
            columns={"lon": "longitude", "lat": "latitude"}
        ),
        get_position=["longitude", "latitude"],
        get_fill_color="color",
        get_radius="radius",
        radius_min_pixels=2,
        radius_max_pixels=14,
        pickable=True,
        auto_highlight=True,
        highlight_color=[255, 255, 100, 180],
        wrap_longitude=True,
    )


# ── Path (trip lines) ─────────────────────────────────────────────────────────
def make_path_layer(paths: list[dict], wrap_longitude: bool = True) -> pdk.Layer:
    """接收 data_loader.build_path_data() 的输出。

    wrap_longitude=False 时用于太平洋副本层（坐标已 +360°），
    防止 deck.gl 将 240° 标准化回 -120°。
    """
    data = [
        {
            "path":  p["path"],
            "color": path_color(p["avg_speed"]),
            "avg_speed": p["avg_speed"],
            "trip_id": p["trip_id"],
        }
        for p in paths
    ]
    return pdk.Layer(
        "PathLayer",
        data=data,
        get_path="path",
        get_color="color",
        get_width=4,
        width_min_pixels=1,
        width_max_pixels=8,
        width_scale=1,
        pickable=True,
        auto_highlight=True,
        highlight_color=[255, 255, 100, 200],
        joint_rounded=True,
        cap_rounded=True,
        wrap_longitude=wrap_longitude,
    )


# ── Photo gap paths（照片补充轨迹线，紫色）──────────────────────────────────
def make_photo_path_layer(paths: list[dict]) -> pdk.Layer | None:
    """将 build_photo_gap_paths 的结果渲染为紫色细线。"""
    if not paths:
        return None
    data = [{"path": p["path"], "color": _PHOTO_COLOR} for p in paths]
    return pdk.Layer(
        "PathLayer",
        data=data,
        get_path="path",
        get_color="color",
        get_width=3,
        width_min_pixels=1,
        width_max_pixels=6,
        pickable=False,
        joint_rounded=True,
        cap_rounded=True,
        opacity=0.75,
        wrap_longitude=True,
    )


# ── Photo points scatter（照片点专用，用于轨迹线模式叠加）────────────────────
def make_photo_scatter_layer(df: pd.DataFrame) -> pdk.Layer | None:
    """从 df 中提取 source=='photo' 的点，渲染为琥珀色散点。"""
    if "source" not in df.columns:
        return None
    photo_df = df[df["source"] == "photo"].copy()
    if photo_df.empty:
        return None

    photo_df["color"]    = [_PHOTO_COLOR] * len(photo_df)
    photo_df["ts_fmt"]   = pd.to_datetime(photo_df["ts"], unit="s").dt.strftime("%Y-%m-%d %H:%M:%S")
    photo_df["speed_fmt"] = "照片位置"
    photo_df["accuracy"] = photo_df["accuracy"].fillna(0)

    return pdk.Layer(
        "ScatterplotLayer",
        data=photo_df[["lon", "lat", "color", "ts_fmt", "speed_fmt", "accuracy"]].rename(
            columns={"lon": "longitude", "lat": "latitude"}
        ),
        get_position=["longitude", "latitude"],
        get_fill_color="color",
        get_radius=30,
        radius_min_pixels=2,
        radius_max_pixels=7,
        pickable=True,
        auto_highlight=True,
        highlight_color=[255, 255, 100, 200],
        wrap_longitude=True,
    )


# ── Viewport ──────────────────────────────────────────────────────────────────
def make_viewport(df: pd.DataFrame) -> pdk.ViewState:
    if df.empty:
        return pdk.ViewState(longitude=116.4, latitude=39.9, zoom=5, pitch=30)

    lon_min = float(df["lon"].min())
    lon_max = float(df["lon"].max())
    lat_min = float(df["lat"].min())
    lat_max = float(df["lat"].max())
    lon_range = lon_max - lon_min
    lat_range = lat_max - lat_min

    if lon_range > 180:
        # 数据横跨两个半球（如中国 + 美国）。
        # 以本初子午线（0°）为中心、世界级缩放来展示，这样中美两侧都在
        # [-180°, 180°] 主世界副本范围内，无需依赖 repeat。
        center_lon = 0.0
        center_lat = (lat_max + lat_min) / 2
        zoom = 1.5
    else:
        center_lon = float(df["lon"].median())
        center_lat = float(df["lat"].median())
        spread = max(lon_range, lat_range, 0.001)
        # 经验公式：spread 0.01° → zoom 13，spread 10° → zoom 6
        zoom = float(np.clip(np.log2(360 / spread) - 1.5, 3, 15))

    return pdk.ViewState(
        longitude=center_lon,
        latitude=center_lat,
        zoom=zoom,
        pitch=0,
        bearing=0,
    )


# ── Flight path layer ─────────────────────────────────────────────────────────
# 国际航班：深橙  国内航班：深蓝  已取消：低透明灰
_INTL_COLOR   = [255, 110,  10, 245]   # 深橙，白底高对比
_DOM_COLOR    = [ 30,  95, 215, 235]   # 深皇家蓝，与 GPS 速度色阶明显区分
_CANCEL_COLOR = [ 80,  80,  80,  80]

# 机场 IATA → 国家代码（用于判断是否跨国航班）
_AIRPORT_COUNTRY: dict[str, str] = {
    # 中国大陆
    "PEK": "CN", "PKX": "CN", "PVG": "CN", "SHA": "CN",
    "CAN": "CN", "CTU": "CN", "CKG": "CN", "XIY": "CN",
    "WUH": "CN", "CSX": "CN", "KMG": "CN", "HGH": "CN",
    "NKG": "CN", "XMN": "CN", "FOC": "CN", "TAO": "CN",
    "TNA": "CN", "SZX": "CN", "URC": "CN", "LHW": "CN",
    "TSN": "CN", "SHE": "CN", "CGO": "CN", "HET": "CN",
    "HAK": "CN", "SYX": "CN", "KWE": "CN", "NNG": "CN",
    "TXN": "CN", "YNT": "CN", "WNZ": "CN", "NTG": "CN",
    "KHN": "CN", "TYN": "CN", "HRB": "CN", "CGQ": "CN",
    # 香港
    "HKG": "HK",
    # 澳门
    "MFM": "MO",
    # 台湾
    "TPE": "TW", "TSA": "TW", "KHH": "TW",
    # 美国
    "LAX": "US", "SFO": "US", "SJC": "US", "SEA": "US",
    "ORD": "US", "JFK": "US", "LGA": "US", "EWR": "US",
    "IAD": "US", "DFW": "US", "DEN": "US", "DTW": "US",
    "ROC": "US", "PHL": "US", "LAS": "US", "ATL": "US",
    "MIA": "US", "BOS": "US", "MSP": "US", "PHX": "US",
    "CLT": "US", "SAN": "US", "PDX": "US", "SLC": "US",
    # 加拿大
    "YYZ": "CA", "YVR": "CA", "YUL": "CA", "YYC": "CA",
    # 韩国
    "ICN": "KR", "GMP": "KR", "PUS": "KR",
    # 日本
    "NRT": "JP", "HND": "JP", "KIX": "JP", "NGO": "JP",
    "FUK": "JP", "CTS": "JP",
    # 新加坡
    "SIN": "SG",
    # 泰国
    "BKK": "TH", "DMK": "TH", "HKT": "TH",
    # 印度尼西亚
    "DPS": "ID", "CGK": "ID",
    # 马来西亚
    "KUL": "MY", "PEN": "MY",
    # 英国
    "LHR": "GB", "LGW": "GB", "MAN": "GB",
    # 法国
    "CDG": "FR", "ORY": "FR",
    # 德国
    "FRA": "DE", "MUC": "DE",
    # 荷兰
    "AMS": "NL",
    # 阿联酋
    "DXB": "AE", "AUH": "AE",
    # 澳大利亚
    "SYD": "AU", "MEL": "AU", "BNE": "AU",
}


def _flight_color(f: dict) -> list[int]:
    if f.get("canceled"):
        return _CANCEL_COLOR
    from_iata = f.get("from_iata", "")
    to_iata   = f.get("to_iata",   "")
    from_country = _AIRPORT_COUNTRY.get(from_iata)
    to_country   = _AIRPORT_COUNTRY.get(to_iata)
    # 两端国家代码不同 → 国际航班（金色）；相同或未知 → 国内/同区域（蓝色）
    if from_country and to_country and from_country != to_country:
        return _INTL_COLOR
    return _DOM_COLOR


def make_flight_path_layer(flights: list[dict]) -> pdk.Layer | None:
    """航班大圆弧路径层（PathLayer）。"""
    if not flights:
        return None
    data = [
        {
            "path":  f["path"],
            "color": _flight_color(f),
            "label": (
                f"{f.get('airline','')}{f.get('flight','')}  "
                f"{f.get('from_city', f.get('from_iata',''))} → "
                f"{f.get('to_city',   f.get('to_iata',''))}  "
                f"({f.get('date','')})  {f.get('distance_km',0):,} km"
            ),
        }
        for f in flights
    ]
    return pdk.Layer(
        "PathLayer",
        data=data,
        get_path="path",
        get_color="color",
        get_width=2,
        width_min_pixels=1,
        width_max_pixels=4,
        pickable=True,
        auto_highlight=True,
        highlight_color=[255, 255, 200, 220],
        joint_rounded=True,
        cap_rounded=True,
        opacity=0.85,
        wrap_longitude=True,
    )


def make_flight_airport_layer(flights: list[dict]) -> pdk.Layer | None:
    """出发 / 到达机场散点层。"""
    if not flights:
        return None
    seen: dict[str, dict] = {}
    for f in flights:
        for iata, coords, city in [
            (f.get("from_iata", ""), f.get("from_coords"), f.get("from_city", "")),
            (f.get("to_iata",   ""), f.get("to_coords"),   f.get("to_city",   "")),
        ]:
            if iata and coords and iata not in seen:
                seen[iata] = {
                    "longitude": coords[0],
                    "latitude":  coords[1],
                    "iata":      iata,
                    "city":      city or iata,
                }
    if not seen:
        return None
    return pdk.Layer(
        "ScatterplotLayer",
        data=list(seen.values()),
        get_position=["longitude", "latitude"],
        get_fill_color=[255, 210, 100, 200],
        get_radius=8000,
        radius_min_pixels=3,
        radius_max_pixels=10,
        pickable=True,
        auto_highlight=True,
        highlight_color=[255, 255, 200, 240],
        wrap_longitude=True,
    )


def make_globe_viewport(df: pd.DataFrame) -> pdk.ViewState:
    """地球视图的初始视角：缩放到能看到整个数据分布。"""
    if df.empty:
        return pdk.ViewState(longitude=116.4, latitude=30.0, zoom=1)
    center_lon = float(df["lon"].median())
    center_lat = float(df["lat"].median())
    return pdk.ViewState(
        longitude=center_lon,
        latitude=center_lat,
        zoom=1.5,
        pitch=0,
        bearing=0,
    )
