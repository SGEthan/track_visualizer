# Track Lens 🛰

个人 GPS 轨迹可视化工具，基于 Streamlit + deck.gl 构建。

![Python](https://img.shields.io/badge/Python-3.10+-blue) ![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-red)

> 100% vibe coded with [Claude Code](https://claude.ai/claude-code)

## 功能

- **散点模式** — 全量 GPS 点，按速度 / 精度 / 时段 / 活动类型着色，JS 端实时视口过滤（最多显示 15 万点）
- **热力图模式** — 密度热力图，直观展示常驻区域
- **轨迹线模式** — 按行程分段着色，自动识别时间断点与坐标跳跃
- **地球视图** — deck.gl GlobeView，俯瞰全球轨迹分布
- **航班记录** — 大圆弧路径，国际 / 国内航班色彩区分，重复航线自动平行偏移

## 快速开始

**1. 克隆仓库**
```bash
git clone https://github.com/SGEthan/track_visualizer.git
cd track_visualizer
```

**2. 安装依赖**
```bash
pip install -r requirements.txt
```

**3. 配置 Mapbox Token**

在 [mapbox.com](https://mapbox.com) 注册并获取 Public Token，然后：
```bash
cp .env.example .env
# 编辑 .env，填入你的 token
```
或直接 export：
```bash
export MAPBOX_TOKEN="pk.your_token_here"
```

**4. 准备数据**

将 GPS 数据放入 `data/` 目录（格式见下方），然后运行预处理：
```bash
python preprocess.py        # 生成 data/tracks.parquet
python preprocess_flights.py  # 生成 data/flight_tracks.json（可选）
```

**5. 启动**
```bash
streamlit run app.py
```

## 数据格式

`preprocess.py` 读取 `all_data.csv`，需包含以下列：

| 列名 | 说明 |
|------|------|
| `timestamp` | Unix 时间戳（秒） |
| `latitude` | 纬度 |
| `longitude` | 经度 |
| `speed` | 速度（km/h），未知填 -1 |
| `accuracy` | GPS 精度（米） |
| `stepType` | 活动类型（1 = 步行，其他 = 非步行） |

## 技术架构

```
app.py                  # 主入口，Streamlit 页面
├── _FLAT_MAP_HTML      # 稳定 iframe（React 不重载，地图瓦片不消失）
├── messenger_html      # 0 高度 iframe，通过 postMessage 传递数据
components/
├── color_utils.py      # numpy 向量化颜色计算（~100x vs .apply()）
├── map_layers.py       # deck.gl 图层工厂，航班颜色逻辑
├── sidebar.py          # 侧边栏控件
└── stats_panel.py      # 统计面板（每日点数、速度分布等）
data_loader.py          # 数据加载、过滤、行程分割、航班大圆弧
preprocess.py           # 原始 CSV → Parquet，距离去重 + 行程分割
config.py               # 性能参数、阈值配置
```

**性能优化：**
- `orjson`（Rust）序列化，比 stdlib json 快 5–10x
- Columnar 散点格式（flat 数组 vs list-of-dicts），JSON 体积减少 3–4x
- JS 端视口过滤，平移缩放无需触发 Python rerun
- 行程分割同时检测时间断点（900s）和距离跳跃（50km），避免 GPS 漂移画出异常长线
