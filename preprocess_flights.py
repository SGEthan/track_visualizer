"""
航班轨迹预处理
==============
运行一次，生成 data/flight_tracks.json：
    python preprocess_flights.py

可选：设置环境变量启用 OpenSky 真实航迹查询（仅对近 30 天内的航班有效）：
    export OPENSKY_USER=your_username
    export OPENSKY_PASS=your_password
    python preprocess_flights.py

说明：
- 对所有航班生成大圆弧路径（飞机实际飞行的航线就是大圆弧）
- 如果设置了 OpenSky 凭据，对近期航班额外尝试获取 ADS-B 真实轨迹
- 中国国内早期航班（2013-2019）OpenSky 覆盖率很低，大圆弧足够准确
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import numpy as np
import requests
import airportsdata

import config

# ── 配置 ────────────────────────────────────────────────────────────────────
OPENSKY_USER = os.environ.get("OPENSKY_USER", "")
OPENSKY_PASS = os.environ.get("OPENSKY_PASS", "")

# 只对近 N 秒内的航班尝试 OpenSky（免费账号历史数据约 30 天）
OPENSKY_LOOKBACK_SECS = 30 * 86400

# 大圆弧采样点数（短途 50，长途国际 100）
N_POINTS_SHORT = 50   # < 3000 km
N_POINTS_LONG  = 100  # >= 3000 km

AIRPORTS = airportsdata.load("IATA")


# ── 地理工具 ─────────────────────────────────────────────────────────────────
def airport_coords(iata: str) -> tuple[float, float] | None:
    ap = AIRPORTS.get(iata.strip().upper())
    return (ap["lon"], ap["lat"]) if ap else None


def haversine_km(lon1, lat1, lon2, lat2) -> float:
    R = 6371.0
    lat1r, lon1r, lat2r, lon2r = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2r - lat1r
    dlon = lon2r - lon1r
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1r) * math.cos(lat2r) * math.sin(dlon / 2) ** 2
    return R * 2 * math.asin(math.sqrt(a))


def great_circle_path(
    lon1: float, lat1: float,
    lon2: float, lat2: float,
    n: int = 80,
) -> list[list[float]]:
    """沿大圆弧生成 n 个 [lon, lat] 航路点。"""
    lat1r, lon1r, lat2r, lon2r = map(math.radians, [lat1, lon1, lat2, lon2])

    d = 2 * math.asin(math.sqrt(
        math.sin((lat2r - lat1r) / 2) ** 2
        + math.cos(lat1r) * math.cos(lat2r) * math.sin((lon2r - lon1r) / 2) ** 2
    ))

    if d < 1e-8:
        return [[lon1, lat1], [lon2, lat2]]

    t = np.linspace(0.0, 1.0, n)
    A = np.sin((1 - t) * d) / math.sin(d)
    B = np.sin(t * d) / math.sin(d)

    x = A * math.cos(lat1r) * math.cos(lon1r) + B * math.cos(lat2r) * math.cos(lon2r)
    y = A * math.cos(lat1r) * math.sin(lon1r) + B * math.cos(lat2r) * math.sin(lon2r)
    z = A * math.sin(lat1r) + B * math.sin(lat2r)

    lats = np.degrees(np.arctan2(z, np.sqrt(x ** 2 + y ** 2)))
    lons = np.degrees(np.arctan2(y, x))

    # 展开经度，消除跨日界线的 ±360° 跳变，使路径坐标连续。
    # deck.gl PathLayer 会把不连续的跳变渲染成断裂，np.unwrap 解决这个问题。
    lons = np.unwrap(lons, period=360)

    return [[float(lons[i]), float(lats[i])] for i in range(n)]


# ── OpenSky 查询 ─────────────────────────────────────────────────────────────
def _opensky_get(url: str, params: dict) -> dict | list | None:
    try:
        r = requests.get(
            url, params=params,
            auth=(OPENSKY_USER, OPENSKY_PASS),
            timeout=20,
        )
        if r.status_code == 200:
            return r.json()
        if r.status_code == 404:
            return None
        print(f"    OpenSky HTTP {r.status_code}")
        return None
    except Exception as e:
        print(f"    OpenSky error: {e}")
        return None


def query_opensky_track(callsign: str, dep_ts: int) -> list[list[float]] | None:
    """尝试从 OpenSky 获取真实 ADS-B 轨迹，返回 [[lon, lat], ...] 或 None。"""
    if not OPENSKY_USER:
        return None

    # 1. 用 callsign + 时间窗搜索航班 ICAO24
    begin = dep_ts - 1800   # 出发前 30 min
    end   = dep_ts + 7200   # 出发后 2 h
    flights = _opensky_get(
        "https://opensky-network.org/api/flights/all",
        {"begin": begin, "end": end, "callsign": callsign.ljust(8)},
    )
    if not flights:
        return None

    icao24 = flights[0].get("icao24")
    if not icao24:
        return None

    # 2. 取轨迹
    track_data = _opensky_get(
        "https://opensky-network.org/api/tracks/all",
        {"icao24": icao24, "time": dep_ts},
    )
    if not track_data:
        return None

    # path 格式：[time, lat, lon, baro_alt, true_alt, on_ground]
    raw_path = track_data.get("path", [])
    path = [
        [float(p[2]), float(p[1])]
        for p in raw_path
        if p[1] is not None and p[2] is not None
    ]
    return path if len(path) >= 5 else None


# ── 时间解析 ─────────────────────────────────────────────────────────────────
def _parse_ts(s: str, timezone_name: str = "UTC") -> int | None:
    if not s or not s.strip():
        return None
    s = s.strip()
    try:
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            try:
                local_timezone = ZoneInfo(timezone_name)
            except ZoneInfoNotFoundError:
                local_timezone = timezone.utc
            dt = dt.replace(tzinfo=local_timezone)
        return int(dt.timestamp())
    except (TypeError, ValueError, OverflowError):
        return None


def _best_ts(
    row: dict,
    timezone_name: str,
    *keys: str,
) -> tuple[int | None, str | None]:
    for key in keys:
        value = _parse_ts(row.get(key, ""), timezone_name)
        if value is not None:
            return value, key
    return None, None


# ── 主流程 ───────────────────────────────────────────────────────────────────
def _default_flight_csv() -> Path | None:
    """自动选择项目目录下最新修改的 Flighty 导出文件。"""
    candidates = list(config.PROJECT_ROOT.glob("FlightyExport-*.csv"))
    return max(candidates, key=lambda path: path.stat().st_mtime_ns) if candidates else None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="将 Flighty CSV 转换为地图航线数据")
    parser.add_argument("input", nargs="?", type=Path, help="Flighty 导出的 CSV")
    parser.add_argument(
        "--output",
        type=Path,
        default=config.FLIGHTS_PATH,
        help="输出 JSON 路径",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = args.input.expanduser().resolve() if args.input else _default_flight_csv()
    if input_path is None or not input_path.exists():
        raise SystemExit(
            "[ERROR] 找不到 FlightyExport-*.csv。请传入文件路径："
            "python preprocess_flights.py /path/to/FlightyExport.csv"
        )
    output_path = args.output.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(input_path, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

    print(f"读取 {input_path.name}：{len(rows)} 条航班记录")
    if OPENSKY_USER:
        print(f"OpenSky 已启用（用户: {OPENSKY_USER}），将尝试查询近 30 天航班的真实轨迹")
    else:
        print("OpenSky 未启用，使用大圆弧路径（设置 OPENSKY_USER/OPENSKY_PASS 可启用）")

    now_ts = datetime.now(timezone.utc).timestamp()
    results: list[dict] = []

    for i, row in enumerate(rows):
        frm = row.get("From", "").strip().upper()
        to  = row.get("To", "").strip().upper()

        from_c = airport_coords(frm)
        to_c   = airport_coords(to)

        if not from_c or not to_c:
            print(f"  [{i+1:3d}] 跳过 {frm}→{to}：机场坐标未知")
            continue

        dep_timezone = AIRPORTS.get(frm, {}).get("tz", "UTC")
        arr_timezone = AIRPORTS.get(to, {}).get("tz", "UTC")
        dep_ts, dep_time_source = _best_ts(
            row,
            dep_timezone,
            "Take off (Actual)", "Take off (Scheduled)",
            "Gate Departure (Actual)", "Gate Departure (Scheduled)",
        )
        arr_ts, arr_time_source = _best_ts(
            row,
            arr_timezone,
            "Landing (Actual)", "Landing (Scheduled)",
            "Gate Arrival (Actual)", "Gate Arrival (Scheduled)",
        )

        airline    = row.get("Airline", "").strip()
        flight_num = row.get("Flight", "").strip()
        callsign   = f"{airline}{flight_num}"
        dist_km    = haversine_km(from_c[0], from_c[1], to_c[0], to_c[1])
        n_pts      = N_POINTS_LONG if dist_km >= 3000 else N_POINTS_SHORT

        # 尝试 OpenSky（仅限近 30 天）
        actual_path = None
        if OPENSKY_USER and dep_ts and (now_ts - dep_ts) < OPENSKY_LOOKBACK_SECS:
            print(f"  [{i+1:3d}] OpenSky: {callsign}  {frm}→{to}")
            actual_path = query_opensky_track(callsign, dep_ts)
            if actual_path:
                print(f"    ✓ 真实轨迹 {len(actual_path)} 点")
            else:
                print("    ✗ 未找到，使用大圆弧")
            time.sleep(0.6)  # 避免超速

        path = actual_path or great_circle_path(
            from_c[0], from_c[1], to_c[0], to_c[1], n=n_pts
        )

        frm_ap = AIRPORTS.get(frm, {})
        to_ap  = AIRPORTS.get(to, {})
        canceled = row.get("Canceled", "false").strip().lower() == "true"

        results.append({
            "id":            f"{callsign}_{row.get('Date', '').strip()}",
            "airline":       airline,
            "flight":        flight_num,
            "date":          row.get("Date", "").strip(),
            "from_iata":     frm,
            "to_iata":       to,
            "from_name":     frm_ap.get("name", frm),
            "to_name":       to_ap.get("name", to),
            "from_city":     frm_ap.get("city", frm),
            "to_city":       to_ap.get("city", to),
            "from_country":  frm_ap.get("country", ""),
            "to_country":    to_ap.get("country", ""),
            "from_coords":   list(from_c),
            "to_coords":     list(to_c),
            "aircraft":      row.get("Aircraft Type Name", "").strip(),
            "tail":          row.get("Tail Number", "").strip(),
            "seat":          row.get("Seat", "").strip(),
            "cabin":         row.get("Cabin Class", "").strip(),
            "dep_ts":        dep_ts,
            "arr_ts":        arr_ts,
            "dep_time_source": dep_time_source,
            "arr_time_source": arr_time_source,
            "distance_km":   round(dist_km),
            "canceled":      canceled,
            "is_actual_track": bool(actual_path),
            "path":          path,
        })

        status = "✓ 真实" if actual_path else "○ 大圆"
        print(f"  [{i+1:3d}] {status}  {frm}→{to}  {dist_km:.0f} km  {row.get('Date','')}")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False)

    total_km    = sum(r["distance_km"] for r in results)
    actual_cnt  = sum(1 for r in results if r["is_actual_track"])
    print(f"\n完成！{len(results)} 条航班 → {output_path}")
    print(f"总飞行距离：{total_km:,} km")
    if OPENSKY_USER:
        print(f"真实轨迹：{actual_cnt}/{len(results)}")


if __name__ == "__main__":
    main()
