import os

# ── Mapbox ──────────────────────────────────────────────────────────────────
# 在 mapbox.com 免费注册并获取 Public Token，通过环境变量传入：
#   export MAPBOX_TOKEN="pk.xxxx..."
# 或在项目根目录创建 .env 文件（已加入 .gitignore）：
#   MAPBOX_TOKEN=pk.xxxx...
MAPBOX_TOKEN = os.environ.get("MAPBOX_TOKEN", "")

MAP_STYLE = "mapbox://styles/mapbox/light-v11"

# ── Performance ──────────────────────────────────────────────────────────────
MAX_SCATTER_POINTS = 150_000
MAX_PATH_POINTS    = 150_000
TRIP_GAP_SECONDS   = 900    # 间隔超过此秒数 = 新行程
MIN_POINT_DIST_M   = 10     # 预处理距离去重阈值（米）：相邻点位移 < 此值则丢弃
TRIP_MAX_JUMP_M    = 50_000 # 相邻点距离超过此值（米）强制新行程，防止坐标跳跃画出异常长线

# ── Speed thresholds (km/h) ──────────────────────────────────────────────────
SPEED_UNKNOWN    = -1
SPEED_STATIONARY =  0
SPEED_WALKING    =  5
SPEED_SLOW       = 15
SPEED_DRIVING    = 60
SPEED_FAST       = 120

# ── Light color palette ──────────────────────────────────────────────────────
ACCENT_COLOR  = "#0099cc"
BG_COLOR      = "#f5f7fa"
CARD_BG       = "rgba(255, 255, 255, 0.95)"
TEXT_COLOR    = "#1e2030"
BORDER_COLOR  = "rgba(0, 153, 204, 0.2)"
