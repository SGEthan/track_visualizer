# Track Lens

一个本地优先的个人移动轨迹地图。它把 GPS、照片位置和航班记录整理成可筛选的个人移动档案，并用 Streamlit + MapLibre GL JS 在浏览器中渲染。

## 功能

- 热力图、轨迹线、散点和组合视图
- 平面地图缩小到世界级时自动过渡为可旋转地球，重新放大后恢复平面
- 按日期、定位精度、速度、活动类型与具体行程筛选
- 散点按速度、精度、时段或活动类型着色
- 地球视图、照片位置和 Flighty 航班轨迹
- 地图视口保持、视口内实时过滤和大数据公平抽样
- 每日记录趋势、速度构成、里程与行程统计
- 原始数据只保留在本机，默认不会被 Git 跟踪

## 快速开始

建议使用 Python 3.10 或更高版本：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
pip install -r requirements.txt
```

准备数据后运行：

```bash
python preprocess.py /path/to/your-location-data.csv
streamlit run app.py
```

应用默认打开 `http://localhost:8501`。

## Mapbox 是可选的

不配置 Token 时，Track Lens 会自动使用 CARTO Positron 浅色底图，完整功能可以直接运行。

如果希望使用 Mapbox 底图：

1. 注册或登录 [Mapbox](https://console.mapbox.com/)。
2. 打开 Developer Console 的 **Access Tokens** 页面。
3. 本地试用可以复制账户自带的 `Default public token`；正式部署建议新建独立的 Public Token。
4. 新建 Token 时只授予地图显示所需的公开读取权限，不要使用 `sk.` 开头的 Secret Token。
5. 创建本地配置：

```bash
cp .env.example .env
```

编辑 `.env`：

```dotenv
MAPBOX_TOKEN=pk.your_public_token
TRACK_TIMEZONE=America/Los_Angeles
MAPBOX_STYLE=mapbox/light-v11
```

`.env` 已被 Git 忽略。Token 必须以 `pk.` 开头；浏览器地图不需要、也不应该使用 `sk.` Secret Token。

## GPS 数据格式

预处理器接受 `dataTime`、`timestamp` 或 `ts` 作为 Unix 秒时间戳，并接受以下坐标别名：

| 标准字段 | 可接受别名 | 必需 | 说明 |
| --- | --- | --- | --- |
| `ts` | `dataTime`, `timestamp` | 是 | Unix 时间戳，单位为秒 |
| `lon` | `longitude` | 是 | 经度，范围 -180 到 180 |
| `lat` | `latitude` | 是 | 纬度，范围 -90 到 90 |
| `accuracy` | — | 是 | 定位误差，单位为米 |
| `stepType` | — | 是 | `1` 表示步行，`0` 表示非步行 |
| `source` | — | 否 | `gps` 或 `photo`，默认 `gps` |
| `altitude` | — | 否 | 海拔，默认 0 |
| `isBackForeground` | `bg` | 否 | 后台状态，默认 0 |

预处理会：

- 丢弃非法经纬度和无效 GPS 精度；
- 使用 15 分钟时间间隔或 50 公里坐标跳跃切分行程；
- 根据相邻坐标重新计算速度；
- 删除相距小于 10 米的冗余点；
- 输出 `data/tracks.parquet` 和 `data/daily_stats.json`。

可以指定输出目录：

```bash
python preprocess.py input.csv --output-dir ./data
```

### 合并照片位置

旧版备份格式可以先执行：

```bash
python merge_photo_data.py
python preprocess.py all_data.csv
```

`merge_photo_data.py` 默认读取 `backUpData-all.csv` 和 `backUpPhotoData.csv`。

## 航班数据

把 Flighty 导出的 CSV 放在项目目录，脚本会自动选择最新的 `FlightyExport-*.csv`：

```bash
python preprocess_flights.py
```

也可以显式传入文件：

```bash
python preprocess_flights.py /path/to/FlightyExport.csv
```

输出为 `data/flight_tracks.json`。默认使用机场间大圆航线；配置 `OPENSKY_USER` 和 `OPENSKY_PASS` 后，脚本会对近期航班尝试获取 OpenSky 轨迹。

## 项目结构

```text
app.py                       页面编排与地图数据准备
config.py                    路径、时区、底图和性能配置
data_loader.py               加载、校验、过滤、抽样与行程聚合
components/
  map_view.py                MapLibre 自适应球面投影与地图交互
  sidebar.py                 筛选控制面板
  stats_panel.py             指标卡和 Plotly 图表
  color_utils.py             向量化颜色计算
  map_layers.py              视口和航班颜色辅助函数
preprocess.py                GPS CSV → Parquet
preprocess_flights.py        Flighty CSV → 航班 JSON
merge_photo_data.py          GPS 与照片位置合并工具
assets/style.css             Daylight Cartography 浅色视觉系统
```

## 隐私与部署

`data/`、所有 CSV、Parquet、`.env` 和 Streamlit secrets 均不会被 Git 跟踪。部署到远程服务器时，需要单独、安全地传输生成后的数据文件。

## 开发检查

```bash
pip install -r requirements-dev.txt
pytest -q
ruff check .
```
