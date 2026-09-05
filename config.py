"""Track Lens 的集中配置。

所有文件路径都相对项目目录解析，因此可以从任意工作目录启动应用。
本地 ``.env`` 会在安装 python-dotenv 后自动加载；生产环境仍建议使用环境变量。
"""
from __future__ import annotations

import os
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError


PROJECT_ROOT = Path(__file__).resolve().parent

try:
    from dotenv import load_dotenv

    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:
    # 不让可选的便利依赖阻止应用启动；系统环境变量始终可用。
    pass


def _detect_timezone() -> str:
    """尽量从系统配置取得 IANA 时区名，失败时使用 UTC。"""
    configured = os.environ.get("TRACK_TIMEZONE", "").strip()
    if configured:
        return configured

    localtime = Path("/etc/localtime")
    try:
        target = localtime.resolve().as_posix()
        marker = "/zoneinfo/"
        if marker in target:
            return target.split(marker, 1)[1]
    except OSError:
        pass
    return "UTC"


TIMEZONE_NAME = _detect_timezone()
try:
    DISPLAY_TIMEZONE = ZoneInfo(TIMEZONE_NAME)
except ZoneInfoNotFoundError:
    TIMEZONE_NAME = "UTC"
    DISPLAY_TIMEZONE = ZoneInfo("UTC")


# ── Data paths ───────────────────────────────────────────────────────────────
DATA_DIR = PROJECT_ROOT / "data"
PARQUET_PATH = DATA_DIR / "tracks.parquet"
STATS_PATH = DATA_DIR / "daily_stats.json"
FLIGHTS_PATH = DATA_DIR / "flight_tracks.json"


# ── Basemap ──────────────────────────────────────────────────────────────────
# Mapbox 是可选的。没有 Token 时自动使用无需注册的 CARTO Positron。
MAPBOX_TOKEN = os.environ.get("MAPBOX_TOKEN", "").strip()
MAPBOX_STYLE = os.environ.get("MAPBOX_STYLE", "mapbox/light-v11").strip()
CARTO_TILE_URL = (
    "https://a.basemaps.cartocdn.com/light_all/{z}/{x}/{y}@2x.png"
)


def has_mapbox_token() -> bool:
    return MAPBOX_TOKEN.startswith("pk.") and "your_" not in MAPBOX_TOKEN.lower()


def basemap_config() -> tuple[str, str, str]:
    """返回 ``(tile_url, provider_label, attribution)``。"""
    if has_mapbox_token():
        tile_url = (
            f"https://api.mapbox.com/styles/v1/{MAPBOX_STYLE}/tiles/256/"
            "{z}/{x}/{y}@2x?access_token=" + MAPBOX_TOKEN
        )
        return tile_url, "MAPBOX", "© Mapbox © OpenStreetMap"
    return CARTO_TILE_URL, "CARTO", "© OpenStreetMap © CARTO"


# ── Performance ──────────────────────────────────────────────────────────────
MAX_SCATTER_POINTS = 150_000
MAX_PATH_POINTS = 150_000
MAX_HEATMAP_POINTS = 80_000
TRIP_GAP_SECONDS = 900
MIN_POINT_DIST_M = 10
TRIP_MAX_JUMP_M = 50_000
PHOTO_GAP_SECONDS = 300


# ── Speed thresholds (km/h) ──────────────────────────────────────────────────
SPEED_UNKNOWN = -1
SPEED_STATIONARY = 0
SPEED_WALKING = 5
SPEED_SLOW = 15
SPEED_DRIVING = 60
SPEED_FAST = 120


# ── UI palette ───────────────────────────────────────────────────────────────
ACCENT_COLOR = "#0a829f"
BG_COLOR = "#f4f7f8"
CARD_BG = "rgba(255, 255, 255, 0.88)"
TEXT_COLOR = "#162c36"
BORDER_COLOR = "rgba(38, 76, 89, 0.13)"
