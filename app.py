"""
Track Lens — GPS 轨迹可视化主入口
运行：streamlit run app.py
"""
from __future__ import annotations

import hashlib
import json
import os

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

try:
    import orjson as _orjson
    def _fast_dumps(obj: object) -> str:
        return _orjson.dumps(obj, option=_orjson.OPT_SERIALIZE_NUMPY).decode()
except ImportError:
    def _fast_dumps(obj: object) -> str:  # type: ignore[misc]
        return json.dumps(obj)

import config
from data_loader import (
    load_all_data,
    load_flight_data,
    filter_by_dates,
    apply_filters,
    downsample,
    build_path_data,
    build_trip_summary,
    spread_flight_paths,
)
from components.color_utils import (
    color_column,
    color_column_vec,
    path_color as _path_color,
    _PHOTO_COLOR,
)
from components.map_layers import (
    make_viewport,
    make_globe_viewport,
    _flight_color,
)
from components.sidebar import render_sidebar
from components.stats_panel import render_stats


# ── 稳定平面地图 HTML（不含任何数据，每次 rerun 字符串完全相同）────────────────
# React 检测到 srcdoc 未变化 → 不重载 iframe → 底图瓦片永远不消失
# 所有运行时数据通过 window.parent._trackLensData 和 postMessage 传入
_FLAT_MAP_HTML = """<!DOCTYPE html>
<html><head><meta charset="UTF-8"><style>
  body{margin:0;background:#f0f4f8;overflow:hidden;}
  #container{width:100%;height:100%;position:absolute;top:0;left:0;}
  #cover{position:absolute;top:0;left:0;width:100%;height:100%;
    background:#f0f4f8;z-index:10;transition:opacity 0.18s ease;pointer-events:none;}
</style></head><body>
<div id="cover"></div>
<div id="container"></div>
<div id="zoom-badge" style="position:absolute;bottom:12px;right:12px;z-index:100;
  background:rgba(255,255,255,0.88);backdrop-filter:blur(8px);
  border:0.5px solid rgba(0,0,0,0.1);border-radius:8px;padding:4px 9px;
  font-family:-apple-system,BlinkMacSystemFont,'SF Pro Text',sans-serif;
  font-size:11px;font-weight:700;color:#374151;letter-spacing:0.3px;
  box-shadow:0 1px 6px rgba(0,0,0,0.08);pointer-events:none;
  user-select:none;white-space:nowrap;">Z 5.0</div>
<script src="https://unpkg.com/deck.gl@8.9.35/dist.min.js"></script>
<script>
/* ── 图层构建（从数据对象 d 重建所有图层）──────────────────────────────── */
function buildLayers(d) {
  var L = [];
  L.push(new deck.TileLayer({
    id:'tiles', data:d.tileUrl, minZoom:0, maxZoom:19, tileSize:256,
    renderSubLayers:function(p){
      var b=p.tile.bbox;
      return new deck.BitmapLayer(p,{data:null,image:p.data,
        bounds:[b.west,b.south,b.east,b.north]});
    }
  }));
  if(d.heatmapData&&d.heatmapData.length>0){
    L.push(new deck.HeatmapLayer({
      id:'heatmap',data:d.heatmapData,
      getPosition:function(p){return p;},getWeight:1,
      radiusPixels:30,intensity:1.2,threshold:0.05,
      colorRange:[[80,0,180,80],[0,80,255,120],[0,220,200,160],
                  [255,220,0,200],[255,100,0,230],[255,255,255,255]],
      aggregation:'SUM'
    }));
  }
  if(d.pathData&&d.pathData.length>0){
    L.push(new deck.PathLayer({
      id:'paths',data:d.pathData,
      getPath:function(p){return p.path;},getColor:function(p){return p.color;},
      getWidth:4,widthMinPixels:1,widthMaxPixels:8,
      pickable:true,autoHighlight:true,highlightColor:[255,255,100,200],
      jointRounded:true,capRounded:true
    }));
  }
  var _col=d.scatterData,_idx=d._scatterIdx;
  if(_col&&_col.n>0&&_idx&&_idx.length>0){
    L.push(new deck.ScatterplotLayer({
      id:'scatter',data:_idx,
      getPosition:function(i){return[_col.lons[i],_col.lats[i]];},
      getFillColor:function(i){var o=i*4;return[_col.colors[o],_col.colors[o+1],_col.colors[o+2],_col.colors[o+3]];},
      getRadius:function(i){return _col.radii[i];},
      radiusMinPixels:2,radiusMaxPixels:14,
      pickable:true,autoHighlight:true,highlightColor:[255,255,100,180]
    }));
  }
  if(d.photoScatter&&d.photoScatter.length>0){
    L.push(new deck.ScatterplotLayer({
      id:'photo-scatter',data:d.photoScatter,
      getPosition:function(p){return[p.longitude,p.latitude];},
      getFillColor:function(p){return p.color;},
      getRadius:30,radiusMinPixels:2,radiusMaxPixels:7,
      pickable:true,autoHighlight:true,highlightColor:[255,255,100,200]
    }));
  }
  if(d.flightPathData&&d.flightPathData.length>0){
    L.push(new deck.PathLayer({
      id:'flights',data:d.flightPathData,
      getPath:function(p){return p.path;},getColor:function(p){return p.color;},
      getWidth:3,widthMinPixels:1,widthMaxPixels:5,
      pickable:true,autoHighlight:true,highlightColor:[255,255,200,220],
      jointRounded:true,capRounded:true,opacity:0.92
    }));
  }
  if(d.airportData&&d.airportData.length>0){
    L.push(new deck.ScatterplotLayer({
      id:'airports',data:d.airportData,
      getPosition:function(p){return[p.longitude,p.latitude];},
      getFillColor:[255,210,100,200],getRadius:8000,
      radiusMinPixels:3,radiusMaxPixels:10,
      pickable:true,autoHighlight:true,highlightColor:[255,255,200,240]
    }));
  }
  return L;
}
/* ── Tooltip ──────────────────────────────────────────────────────────────── */
function getTooltip(info){
  if(info.object==null)return null;
  var obj=info.object,lid=info.layer?info.layer.id:'',html='';
  if(lid==='scatter'){
    var col=window._scatterCol;
    if(!col)return null;
    var i=obj;  /* obj is now an integer index into columnar arrays */
    html='<b style="color:#00ffe0">'+col.ts_fmts[i]+'</b><br>'+
         '&#x1F680; &#x901F;&#x5EA6;&#xFF1A;'+col.speed_fmts[i]+'<br>'+
         '&#x1F4CD; &#x7CBE;&#x5EA6;&#xFF1A;'+col.accuracies[i].toFixed(0)+' m';
  }else if(lid==='photo-scatter'){
    html='<b style="color:#ffc832">'+obj.ts_fmt+'</b><br>'+
         '&#x1F4F7; &#x7167;&#x7247;&#x4F4D;&#x7F6E;<br>'+
         '&#x1F4CD; &#x7CBE;&#x5EA6;&#xFF1A;'+obj.accuracy.toFixed(0)+' m';
  }else if(lid==='flights'){
    html='&#x2708;&#xFE0F; '+obj.label;
  }else if(lid==='airports'){
    html='<b style="color:#ffd264">'+obj.iata+'</b>&ensp;'+obj.city;
  }
  if(!html)return null;
  return{html:html,style:{background:'rgba(8,10,20,0.92)',
    border:'1px solid rgba(0,255,224,0.3)',borderRadius:'8px',padding:'8px 12px',
    fontFamily:'Courier New, monospace',color:'#e0e8ff',fontSize:'12px',
    lineHeight:'1.6',maxWidth:'300px',zIndex:'9999'}};
}
/* ── DeckGL 实例（首次数据到达后才创建，之后只 setProps 更新图层）──────── */
var deckgl=null;
var _lastFilterKey=null;
var _currentData=null;    /* 最新一次 Python 下发的完整数据对象 */
var _lastViewState=null;  /* 最新视口状态，供视口过滤使用 */
var _VP_SCATTER_CAP=150000;/* 视口内散点上限，超出则等间隔采样 */
/* ── 视口过滤：在 columnar 数组中找 bbox 内的索引，超限等间隔采样 ─────── */
function filterViewport(col,vs){
  if(!col||!col.n)return[];
  var cw=(document.getElementById('container')||{}).clientWidth||1400;
  var ch=(document.getElementById('container')||{}).clientHeight||700;
  var hw=cw/256*180/Math.pow(2,vs.zoom);
  var hh=ch/256*180/Math.pow(2,vs.zoom)/Math.max(0.05,Math.cos(vs.latitude*Math.PI/180));
  var lon0=vs.longitude-hw,lon1=vs.longitude+hw;
  var lat0=vs.latitude-hh,lat1=vs.latitude+hh;
  var lons=col.lons,lats=col.lats,n=col.n,idx=[];
  for(var i=0;i<n;i++){
    if(lons[i]>=lon0&&lons[i]<=lon1&&lats[i]>=lat0&&lats[i]<=lat1)idx.push(i);
  }
  if(idx.length>_VP_SCATTER_CAP){
    var step=Math.ceil(idx.length/_VP_SCATTER_CAP);
    return idx.filter(function(_,j){return j%step===0;});
  }
  return idx;
}
/* ── 用当前视口重建散点图层并推送给 deck.gl ─────────────────────────────── */
function applyViewportFilter(){
  if(!deckgl||!_currentData||!_lastViewState)return;
  var col=window._scatterCol;
  var idx=(col&&col.n>0)?filterViewport(col,_lastViewState):[];
  var d=Object.assign({},_currentData,{_scatterIdx:idx});
  deckgl.setProps({layers:buildLayers(d)});
}
function initDeckGL(data){
  var sv=null;
  try{sv=window.parent._trackLensVP||null;}catch(e){}
  var vs=sv
    ?{longitude:sv.longitude,latitude:sv.latitude,zoom:sv.zoom,minZoom:0,maxZoom:20}
    :{longitude:data.viewport.longitude,latitude:data.viewport.latitude,
      zoom:data.viewport.zoom,minZoom:0,maxZoom:20};
  _lastFilterKey=data.filterKey||null;
  deckgl=new deck.DeckGL({
    container:'container',
    views:new deck.MapView({controller:true,repeat:true}),
    initialViewState:vs,
    layers:buildLayers(data),
    getTooltip:getTooltip,
    onViewStateChange:function(params){
      _lastViewState=params.viewState;
      var cover=document.getElementById('cover');
      if(cover&&cover.style.opacity!=='0'){cover.style.opacity='0';}
      try{window.parent._trackLensVP=params.viewState;}catch(e){}
      var zb=document.getElementById('zoom-badge');
      if(zb){var z=params.viewState.zoom;zb.textContent='Z '+z.toFixed(1);
        zb.style.color=z>=11?'#2563eb':'#374151';
        zb.style.borderColor=z>=11?'rgba(59,130,246,0.35)':'rgba(0,0,0,0.1)';}
      /* 视口散点过滤（200ms 防抖，平移停止后更新）*/
      clearTimeout(window._vpScatterTimer);
      window._vpScatterTimer=setTimeout(function(){applyViewportFilter();},200);
      /* 持久化视口位置到 URL（300ms 防抖）*/
      clearTimeout(window._vpTimer);
      window._vpTimer=setTimeout(function(){
        try{
          var vs2=params.viewState;
          var url=new URL(window.parent.location.href);
          url.searchParams.set('map_lon',vs2.longitude.toFixed(5));
          url.searchParams.set('map_lat',vs2.latitude.toFixed(5));
          url.searchParams.set('map_zoom',vs2.zoom.toFixed(2));
          window.parent.history.replaceState(null,'',url.toString());
        }catch(e){}
      },300);
    }
  });
}
function applyData(data){
  _currentData=data;
  window._scatterCol=(data.scatterData&&data.scatterData.n>0)?data.scatterData:null;
  if(!deckgl){
    initDeckGL(data);
    if(_lastViewState)applyViewportFilter();
  }else{
    if(data.filterKey&&data.filterKey!==_lastFilterKey&&data.dataViewport){
      _lastFilterKey=data.filterKey;
      try{window.parent._trackLensVP=null;}catch(e){}
      deckgl.setProps({
        layers:buildLayers(data),
        initialViewState:{
          longitude:data.dataViewport.longitude,
          latitude:data.dataViewport.latitude,
          zoom:data.dataViewport.zoom,
          minZoom:0,maxZoom:20,
          transitionDuration:800,
          transitionInterpolator:new deck.FlyToInterpolator({speed:1.5})
        }
      });
    }else{
      applyViewportFilter();
    }
  }
}
/* ── 监听 messenger iframe 的数据更新通知 ──────────────────────────────── */
window.addEventListener('message',function(e){
  if(e.data&&e.data.type==='tracklens-update'){
    var d=null;
    try{d=window.parent._trackLensData;}catch(e2){}
    if(d)applyData(d);
  }
});
/* ── 首次加载：等待 messenger 写入 parent 后初始化 ────────────────────── */
(function tryInit(){
  var d=null;
  try{d=window.parent._trackLensData;}catch(e){}
  if(d){applyData(d);}else{setTimeout(tryInit,50);}
})();
</script></body></html>"""


# ── 页面配置 ─────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Track Lens",
    page_icon="🛰",
    layout="wide",
    initial_sidebar_state="expanded",
)

css_path = os.path.join(os.path.dirname(__file__), "assets", "style.css")
if os.path.exists(css_path):
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

_SIDEBAR_FAB_HTML = """<!DOCTYPE html><html><body style="margin:0;background:transparent;overflow:hidden;">
<script>
(function(){
  var doc = window.parent.document;

  // 注入样式到父页面 <head>，确保和 style.css 作用域无关
  var oldStyle = doc.getElementById('tl-sb-fab-style');
  if (oldStyle) oldStyle.remove();
  var s = doc.createElement('style');
  s.id = 'tl-sb-fab-style';
  s.textContent = [
    '#tl-sb-fab{',
      'position:fixed;top:50%;left:0;transform:translateY(-50%);z-index:99999;',
      'width:28px;height:48px;',
      'background:#ffffff;',
      'border:0.5px solid rgba(0,0,0,0.12);border-left:none;',
      'border-radius:0 10px 10px 0;',
      'color:#3b82f6;font-size:18px;font-family:sans-serif;line-height:1;',
      'cursor:pointer;user-select:none;',
      'display:none;align-items:center;justify-content:center;',
      'box-shadow:3px 0 12px rgba(0,0,0,0.08);',
      'transition:background .18s,width .18s,box-shadow .18s;',
    '}',
    '#tl-sb-fab:hover{',
      'background:#f0f4f8;width:34px;',
      'box-shadow:3px 0 16px rgba(59,130,246,0.15);',
    '}',
  ].join('');
  doc.head.appendChild(s);

  // 创建/重建 FAB 元素
  var old = doc.getElementById('tl-sb-fab');
  if (old) old.remove();
  if (window.parent._tlFabTimer) clearInterval(window.parent._tlFabTimer);

  var fab = doc.createElement('div');
  fab.id = 'tl-sb-fab';
  fab.innerHTML = '&#9776;';
  fab.title = '控制面板';

  fab.onclick = function(){
    var sels = [
      '[data-testid="collapsedControl"]',
      '[data-testid="stSidebarCollapsedControl"]',
      'button[data-testid*="Sidebar"]',
      'button[aria-label*="sidebar" i]',
      'button[aria-label*="open" i]',
    ];
    for (var i=0; i<sels.length; i++) {
      var b = doc.querySelector(sels[i]); if (b) { b.click(); return; }
    }
    ['keydown','keypress','keyup'].forEach(function(t){
      doc.dispatchEvent(new KeyboardEvent(t,{key:'[',code:'BracketLeft',keyCode:219,bubbles:true}));
    });
  };

  doc.body.appendChild(fab);

  function update(){
    var sb = doc.querySelector('[data-testid="stSidebar"]');
    if (!sb) return;
    var hidden = sb.offsetWidth < 60 || window.parent.getComputedStyle(sb).display === 'none';
    fab.style.display = hidden ? 'flex' : 'none';
  }
  window.parent._tlFabTimer = setInterval(update, 300);
  update();
})();
</script></body></html>"""
components.html(_SIDEBAR_FAB_HTML, height=0)


# ── 加载数据 ──────────────────────────────────────────────────────────────────
with st.spinner("加载数据中..."):
    df_all       = load_all_data()
    flights_data = load_flight_data()


# ── 侧边栏 ────────────────────────────────────────────────────────────────────
filters = render_sidebar(df_all)


# ── 数据过滤 ──────────────────────────────────────────────────────────────────
df_date = filter_by_dates(df_all, filters["start"], filters["end"])
df_filtered = apply_filters(
    df_date,
    max_accuracy=filters["max_accuracy"],
    speed_min=filters["speed_min"],
    speed_max=filters["speed_max"],
    step_types=filters["step_types"],
)


# ── 行程选择器（侧边栏，基于已过滤数据）───────────────────────────────────────
def _fmt_dur(minutes: int) -> str:
    if minutes < 60:
        return f"{minutes}min"
    h, m = divmod(minutes, 60)
    return f"{h}h{m:02d}min" if m else f"{h}h"


with st.sidebar:
    st.markdown("---")
    with st.expander("🗺 行程列表", expanded=False):
        trip_sum = build_trip_summary(df_filtered)
        if trip_sum.empty:
            st.caption("无行程数据。")
            _selected_ids: list = []
        elif len(trip_sum) > 200:
            st.caption(f"共 {len(trip_sum)} 条行程，请缩小日期范围后再筛选。")
            _selected_ids = []
        else:
            def _trip_label(row: pd.Series) -> str:
                dt  = pd.to_datetime(row["start_ts"], unit="s").strftime("%m-%d %H:%M")
                dur = _fmt_dur(int(row["duration_min"]))
                dist = row["distance_km"]
                dist_str = f" · {dist:.1f} km" if dist >= 0.1 else ""
                spd = f" · {row['avg_speed']:.0f} km/h" if row["avg_speed"] >= 0 else ""
                return f"{dt}  {dur}{dist_str}{spd}"

            _id_list   = trip_sum["trip_id"].tolist()
            _label_map = {int(row["trip_id"]): _trip_label(row)
                          for _, row in trip_sum.iterrows()}

            _selected_ids = st.multiselect(
                "选择行程（空 = 显示全部）",
                options=_id_list,
                format_func=lambda tid: _label_map[int(tid)],
                default=[],
                key="trip_selector",
            )

if _selected_ids:
    df_filtered = df_filtered[df_filtered["trip_id"].isin(_selected_ids)]


# ── 状态条 ────────────────────────────────────────────────────────────────────
n_pts   = len(df_filtered)
n_trips = df_filtered["trip_id"].nunique() if not df_filtered.empty else 0


# ── 参数提取 ──────────────────────────────────────────────────────────────────
view_mode  = filters["view_mode"]
color_by   = filters["color_by"]
globe_mode = filters.get("globe_mode", False)

effective_mode = view_mode
if globe_mode and view_mode in ("热力图", "热力 + 轨迹"):
    effective_mode = "轨迹线"
    st.info("🌍 地球视图不支持热力图，已自动切换为轨迹线模式。")

# ── 航班过滤 ──────────────────────────────────────────────────────────────────
show_flights = filters.get("show_flights", False)
_start_str = filters["start"].isoformat()
_end_str   = filters["end"].isoformat()
filtered_flights = [
    f for f in flights_data
    if _start_str <= f.get("date", "") <= _end_str
] if flights_data else []
# 同一航线有多次飞行时，将路径平行展开以便区分
display_flights = spread_flight_paths(filtered_flights) if filtered_flights else []


# ── 渲染 ──────────────────────────────────────────────────────────────────────
if df_filtered.empty:
    st.warning("当前筛选条件下没有数据，请调整时间范围或过滤器。")
else:
    # ── 紧凑状态条 ────────────────────────────────────────────────────────────
    st.markdown(
        f"""
        <div class="tl-status">
            <span class="tl-chip b">🛰 {n_pts:,} 个点</span>
            <span class="tl-chip g">↗ {n_trips:,} 条行程</span>
            <span class="tl-chip">{filters['start']} → {filters['end']}</span>
            <span class="tl-chip a">{effective_mode}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    # ── 视口 ──────────────────────────────────────────────────────────────────
    vp = make_globe_viewport(df_filtered) if globe_mode else make_viewport(df_filtered)
    center_lon = float(vp.longitude)
    center_lat = float(vp.latitude)
    zoom_level = float(vp.zoom)

    # ── 数据拟合视口（不受 URL 覆盖，用于自动缩放）─────────────────────────────
    data_lon  = center_lon
    data_lat  = center_lat
    data_zoom = zoom_level

    # ── 过滤状态哈希（供前端检测 filter 变化，触发飞行动画）──────────────────
    _fk_str = (
        f"{filters['start']}|{filters['end']}|{filters['max_accuracy']}|"
        f"{filters['speed_min']}|{filters['speed_max']}|"
        f"{sorted(filters['step_types'])}|{sorted([int(x) for x in _selected_ids])}"
    )
    filter_key = hashlib.md5(_fk_str.encode()).hexdigest()[:12]

    # ── 从 URL 恢复上次视口（防止每次 rerun 地图归零）─────────────────────────
    def _get_qp_float(key: str, default: float) -> float:
        try:
            v = st.query_params.get(key)
            if v is not None:
                return float(v)
        except (ValueError, TypeError):
            pass
        return default

    if not globe_mode:
        center_lon = _get_qp_float("map_lon",  center_lon)
        center_lat = _get_qp_float("map_lat",  center_lat)
        zoom_level = _get_qp_float("map_zoom", zoom_level)

    # ── 数据准备 ───────────────────────────────────────────────────────────────
    # --- 热力图点（最多 80,000，控制 HTML 大小）---
    heatmap_data: list = []
    _need_heatmap = effective_mode in ("热力图", "热力 + 轨迹")
    if _need_heatmap:
        df_hm = (
            df_filtered
            if len(df_filtered) <= 80_000
            else df_filtered.sample(80_000, random_state=0)
        )
        heatmap_data = df_hm[["lon", "lat"]].values.tolist()

    # --- 散点（columnar 格式：flat 数组 + orjson，比 list-of-dicts 快 5-10x）---
    scatter_data = None  # None → JS 收到 null → 不渲染散点层
    if effective_mode == "散点":
        df_s       = df_filtered
        colors_arr = color_column_vec(df_s, color_by)          # (n,4) int32 numpy
        radii_arr  = np.clip(df_s["accuracy"].values * 0.4, 2.0, 25.0)
        spd_vals   = df_s["speed"].tolist()
        scatter_data = {
            "n":          int(len(df_s)),
            "lons":       df_s["lon"].tolist(),
            "lats":       df_s["lat"].tolist(),
            "colors":     colors_arr.flatten().tolist(),   # flat RGBA: [r,g,b,a, r,g,b,a, ...]
            "radii":      radii_arr.tolist(),
            "ts_fmts":    pd.to_datetime(df_s["ts"], unit="s").dt.strftime("%Y-%m-%d %H:%M:%S").tolist(),
            "speed_fmts": [f"{s:.1f} km/h" if s >= 0 else "未知" for s in spd_vals],
            "accuracies": df_s["accuracy"].tolist(),
        }

    # --- 轨迹线 + 照片散点 ---
    show_photos        = filters.get("show_photos", True)
    path_data:          list = []
    photo_scatter_data: list = []
    if effective_mode in ("轨迹线", "热力 + 轨迹") and not globe_mode:
        df_p  = downsample(df_filtered, config.MAX_PATH_POINTS)
        trips = build_path_data(df_p)
        path_data = [
            {"path": p["path"], "color": _path_color(p["avg_speed"])}
            for p in trips
        ]
        if show_photos and "source" in df_filtered.columns:
            photo_df = df_filtered[df_filtered["source"] == "photo"].copy()
            if not photo_df.empty:
                photo_df["accuracy"] = photo_df["accuracy"].fillna(0)
                p_ts = (
                    pd.to_datetime(photo_df["ts"], unit="s")
                    .dt.strftime("%Y-%m-%d %H:%M:%S")
                    .tolist()
                )
                p_lons = photo_df["lon"].tolist()
                p_lats = photo_df["lat"].tolist()
                p_accs = photo_df["accuracy"].tolist()
                photo_scatter_data = [
                    {
                        "longitude": float(p_lons[i]),
                        "latitude":  float(p_lats[i]),
                        "color":     _PHOTO_COLOR,
                        "ts_fmt":    p_ts[i],
                        "accuracy":  float(p_accs[i]),
                    }
                    for i in range(len(p_lons))
                ]

    # --- 航班 ---
    flight_path_data: list = []
    airport_data:     list = []
    if show_flights and display_flights:
        flight_path_data = [
            {
                "path":  f["path"],
                "color": _flight_color(f),
                "label": (
                    f"{f.get('airline','')}{f.get('flight','')}  "
                    f"{f.get('from_city', f.get('from_iata',''))} → "
                    f"{f.get('to_city',   f.get('to_iata',''))}  "
                    f"({f.get('date','')})  {f.get('distance_km', 0):,} km"
                ),
            }
            for f in display_flights
        ]
        seen: dict[str, dict] = {}
        for ff in filtered_flights:  # 机场坐标用原始数据
            for iata, coords, city in [
                (ff.get("from_iata", ""), ff.get("from_coords"), ff.get("from_city", "")),
                (ff.get("to_iata",   ""), ff.get("to_coords"),   ff.get("to_city",   "")),
            ]:
                if iata and coords and iata not in seen:
                    seen[iata] = {
                        "longitude": coords[0],
                        "latitude":  coords[1],
                        "iata":      iata,
                        "city":      city or iata,
                    }
        airport_data = list(seen.values())

    # ═══════════════════════════════════════════════════════════════════════════
    # 地球模式（_GlobeView，保持原有逻辑）
    # ═══════════════════════════════════════════════════════════════════════════
    if globe_mode:
        df_p_globe = downsample(df_filtered, config.MAX_PATH_POINTS)
        globe_trips = build_path_data(df_p_globe)
        path_json = json.dumps([
            {"path": p["path"], "color": _path_color(p["avg_speed"])}
            for p in globe_trips
        ])
        if show_flights and display_flights:
            flight_path_json = json.dumps([
                {"path": f["path"], "color": _flight_color(f)}
                for f in display_flights
            ])
            seen_g: dict[str, dict] = {}
            for ff in filtered_flights:  # 机场坐标用原始数据
                for iata, coords, city in [
                    (ff.get("from_iata", ""), ff.get("from_coords"), ff.get("from_city", "")),
                    (ff.get("to_iata",   ""), ff.get("to_coords"),   ff.get("to_city",   "")),
                ]:
                    if iata and coords and iata not in seen_g:
                        seen_g[iata] = {
                            "longitude": coords[0],
                            "latitude":  coords[1],
                            "iata":      iata,
                            "city":      city or iata,
                        }
            globe_airport_json = json.dumps(list(seen_g.values()))
        else:
            flight_path_json   = "[]"
            globe_airport_json = "[]"

        # 照片散点
        globe_photo_json = "[]"
        if show_photos and "source" in df_filtered.columns:
            photo_df = df_filtered[df_filtered["source"] == "photo"].copy()
            if not photo_df.empty:
                photo_df["accuracy"] = photo_df["accuracy"].fillna(0)
                p_ts = (
                    pd.to_datetime(photo_df["ts"], unit="s")
                    .dt.strftime("%Y-%m-%d %H:%M:%S")
                    .tolist()
                )
                globe_photo_json = json.dumps([
                    {
                        "longitude": float(photo_df["lon"].iloc[i]),
                        "latitude":  float(photo_df["lat"].iloc[i]),
                        "color":     _PHOTO_COLOR,
                        "ts_fmt":    p_ts[i],
                        "accuracy":  float(photo_df["accuracy"].iloc[i]),
                    }
                    for i in range(len(photo_df))
                ])

        globe_html = """<!DOCTYPE html>
<html><head><meta charset="UTF-8">
<style>
  body { margin:0; background:#0d1b2e; overflow:hidden; }
  #container { width:100%; height:700px; position:relative; }
</style></head><body>
<div id="container"></div>
<script src="https://unpkg.com/deck.gl@8.9.35/dist.min.js"></script>
<script>
var pathData       = __PATH_DATA__;
var flightPathData = __FLIGHT_PATH_DATA__;
var photoData      = __PHOTO_DATA__;
var airportData    = __AIRPORT_DATA__;

new deck.DeckGL({
  container: 'container',
  views: new deck._GlobeView({ controller: true }),
  parameters: { clearColor: [0.05, 0.11, 0.18, 1] },
  initialViewState: {
    longitude: __CENTER_LON__,
    latitude:  __CENTER_LAT__,
    zoom: 1.5,
    minZoom: 0,
    maxZoom: 10,
  },
  layers: [
    /* 海洋面：与陆地合起来铺满整个球面，阻断背面透视 */
    new deck.GeoJsonLayer({
      id: 'ocean',
      data: 'https://d2ad6b4ur7yvpq.cloudfront.net/naturalearth-3.3.0/ne_110m_ocean.geojson',
      filled: true,
      stroked: false,
      getFillColor: [184, 207, 224, 255],
    }),
    /* 国家面：各国独立多边形，自带国界线 */
    new deck.GeoJsonLayer({
      id: 'countries',
      data: 'https://d2ad6b4ur7yvpq.cloudfront.net/naturalearth-3.3.0/ne_110m_admin_0_countries.geojson',
      stroked: true,
      filled: true,
      getFillColor: [220, 232, 242, 255],
      getLineColor: [120, 150, 170, 220],
      lineWidthMinPixels: 0.5,
      lineWidthMaxPixels: 1.5,
    }),
    new deck.PathLayer({
      id: 'gps-paths',
      data: pathData,
      getPath:  function(d) { return d.path; },
      getColor: function(d) { return d.color; },
      getWidth: 4,
      widthMinPixels: 1,
      widthMaxPixels: 8,
      jointRounded: true,
      capRounded: true
    }),
    new deck.PathLayer({
      id: 'flight-paths',
      data: flightPathData,
      getPath:  function(d) { return d.path; },
      getColor: function(d) { return d.color; },
      getWidth: 2,
      widthMinPixels: 1,
      widthMaxPixels: 4,
      jointRounded: true,
      capRounded: true,
      opacity: 0.85
    }),
    new deck.ScatterplotLayer({
      id: 'photo-scatter',
      data: photoData,
      getPosition: function(d) { return [d.longitude, d.latitude]; },
      getFillColor: function(d) { return d.color; },
      getRadius: 30,
      radiusMinPixels: 3,
      radiusMaxPixels: 8,
      pickable: false,
    }),
    new deck.ScatterplotLayer({
      id: 'airports',
      data: airportData,
      getPosition: function(d) { return [d.longitude, d.latitude]; },
      getFillColor: [255, 210, 100, 200],
      getRadius: 8000,
      radiusMinPixels: 3,
      radiusMaxPixels: 10,
      pickable: false,
    })
  ]
});
</script></body></html>"""

        globe_html = (globe_html
            .replace("__PATH_DATA__",        path_json)
            .replace("__FLIGHT_PATH_DATA__", flight_path_json)
            .replace("__PHOTO_DATA__",       globe_photo_json)
            .replace("__AIRPORT_DATA__",     globe_airport_json)
            .replace("__CENTER_LON__",       str(center_lon))
            .replace("__CENTER_LAT__",       str(center_lat)))

        components.html(globe_html, height=730)

    # ═══════════════════════════════════════════════════════════════════════════
    # 平面地图：messenger（数据）+ 稳定 iframe（地图壳，React 不重载）
    # ═══════════════════════════════════════════════════════════════════════════
    else:
        _has_token = bool(
            config.MAPBOX_TOKEN
            and config.MAPBOX_TOKEN != "YOUR_MAPBOX_TOKEN_HERE"
        )
        _tile_url = (
            "https://api.mapbox.com/styles/v1/mapbox/light-v11/tiles/"
            "{z}/{x}/{y}?access_token=" + config.MAPBOX_TOKEN
            if _has_token
            else "https://a.basemaps.cartocdn.com/light_all/{z}/{x}/{y}.png"
        )

        # 0 高度 messenger：写入数据并通知地图 iframe 更新图层
        # 注意：用三引号避免 <\/script> 转义问题
        messenger_html = """<!DOCTYPE html><html><head><meta charset="UTF-8"></head><body>
<script>
window.parent._trackLensData={
  tileUrl:__TILE_URL__,
  heatmapData:__HEATMAP_DATA__,
  scatterData:__SCATTER_DATA__,
  pathData:__PATH_DATA__,
  photoScatter:__PHOTO_SCATTER_DATA__,
  flightPathData:__FLIGHT_PATH_DATA__,
  airportData:__AIRPORT_DATA__,
  viewport:{longitude:__CENTER_LON__,latitude:__CENTER_LAT__,zoom:__ZOOM__},
  dataViewport:{longitude:__DATA_LON__,latitude:__DATA_LAT__,zoom:__DATA_ZOOM__},
  filterKey:__FILTER_KEY__
};
var fr=window.parent.document.querySelectorAll('iframe');
for(var i=0;i<fr.length;i++){
  try{fr[i].contentWindow.postMessage({type:'tracklens-update'},'*');}catch(e){}}
</script></body></html>"""
        messenger_html = (messenger_html
            .replace("__TILE_URL__",           json.dumps(_tile_url))
            .replace("__HEATMAP_DATA__",       json.dumps(heatmap_data))
            .replace("__SCATTER_DATA__",       _fast_dumps(scatter_data))
            .replace("__PATH_DATA__",          json.dumps(path_data))
            .replace("__PHOTO_SCATTER_DATA__", json.dumps(photo_scatter_data))
            .replace("__FLIGHT_PATH_DATA__",   json.dumps(flight_path_data))
            .replace("__AIRPORT_DATA__",       json.dumps(airport_data))
            .replace("__CENTER_LON__",         str(center_lon))
            .replace("__CENTER_LAT__",         str(center_lat))
            .replace("__ZOOM__",               str(zoom_level))
            .replace("__DATA_LON__",           str(data_lon))
            .replace("__DATA_LAT__",           str(data_lat))
            .replace("__DATA_ZOOM__",          str(data_zoom))
            .replace("__FILTER_KEY__",         json.dumps(filter_key)))

        # messenger 先渲染（写入 parent 数据），地图 iframe 后渲染并读取
        components.html(messenger_html, height=0)
        components.html(_FLAT_MAP_HTML, height=730)


# ── 统计面板 ──────────────────────────────────────────────────────────────────
render_stats(df_filtered)
