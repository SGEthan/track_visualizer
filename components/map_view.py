"""MapLibre 地图渲染器。

MapLibre GL JS 5 在同一 WebGL canvas 中插值 globe 与 Mercator 投影，避免在两个
renderer 之间跳切。Streamlit rerun 时只更新 GeoJSON source，不销毁地图和相机。
"""
from __future__ import annotations

import json
from copy import deepcopy
from typing import Any

import streamlit as st

try:
    import orjson
except ImportError:  # pragma: no cover
    orjson = None


def _dumps(value: Any) -> str:
    if orjson is not None:
        text = orjson.dumps(value, option=orjson.OPT_SERIALIZE_NUMPY).decode()
    else:
        text = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return (
        text.replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("&", "\\u0026")
        .replace("\u2028", "\\u2028")
        .replace("\u2029", "\\u2029")
    )


_MAP_HTML = r"""<!doctype html>
<html lang="zh-CN"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<link rel="stylesheet" href="https://unpkg.com/maplibre-gl@5.24.0/dist/maplibre-gl.css">
<style>
:root{--ink:#19333e;--muted:#657c87;--cyan:#0a829f;--line:rgba(38,76,89,.16);--panel:rgba(255,255,255,.88);--shadow:0 18px 60px rgba(39,66,76,.18)}
*{box-sizing:border-box}html,body,#map{width:100%;height:100%;margin:0;overflow:hidden;background:#e6eef1}
body{font-family:Inter,-apple-system,BlinkMacSystemFont,"SF Pro Display","Segoe UI",sans-serif;color:var(--ink)}
#map{position:absolute;inset:0}.maplibregl-canvas{outline:none}.maplibregl-control-container{display:none}
#veil{position:absolute;inset:0;z-index:30;background:radial-gradient(circle at 50% 42%,rgba(88,169,188,.2),transparent 32%),#f3f7f8;display:grid;place-items:center;transition:opacity .55s ease;pointer-events:none}
.loader{text-align:center;letter-spacing:.2em;text-transform:uppercase;font-size:10px;color:#657c87}.loader-mark{width:42px;height:42px;border:1px solid rgba(10,130,159,.2);border-top-color:#0a829f;border-radius:50%;margin:0 auto 13px;animation:spin 1s linear infinite;box-shadow:0 0 28px rgba(10,130,159,.1)}
@keyframes spin{to{transform:rotate(360deg)}}.glass{background:var(--panel);border:1px solid var(--line);box-shadow:var(--shadow);backdrop-filter:blur(18px);-webkit-backdrop-filter:blur(18px)}
#hud{position:absolute;top:18px;left:18px;z-index:20;border-radius:16px;padding:12px 15px;min-width:190px;pointer-events:none;transition:background .35s ease,border-color .35s ease}
.hud-top{display:flex;align-items:center;gap:9px;margin-bottom:7px}.pulse{width:7px;height:7px;border-radius:50%;background:var(--cyan);box-shadow:0 0 0 5px rgba(10,130,159,.1),0 0 16px rgba(10,130,159,.45)}
#mode{font-size:10px;font-weight:800;letter-spacing:.17em;color:#0a728c;text-transform:uppercase;transition:opacity .2s ease}#summary{font-size:12px;font-weight:650;letter-spacing:.015em;color:#19333e}#subsummary{font-size:9px;color:var(--muted);margin-top:4px;letter-spacing:.08em;text-transform:uppercase}
#controls{position:absolute;top:18px;right:18px;z-index:21;display:flex;flex-direction:column;gap:7px}.map-btn{width:36px;height:36px;border-radius:11px;border:1px solid var(--line);color:#19333e;background:var(--panel);box-shadow:0 8px 24px rgba(39,66,76,.16);backdrop-filter:blur(18px);cursor:pointer;font-size:17px;line-height:1;display:grid;place-items:center;transition:transform .15s ease,border-color .15s ease,background .15s ease}.map-btn:hover{transform:translateY(-1px);border-color:rgba(10,130,159,.45);background:rgba(255,255,255,.96)}.map-btn:active{transform:scale(.96)}#fit{font-size:14px}
#footer{position:absolute;left:18px;bottom:16px;z-index:20;border-radius:10px;padding:7px 10px;font-size:8px;letter-spacing:.08em;color:#657c87;text-transform:uppercase;pointer-events:none}#zoom{position:absolute;right:18px;bottom:16px;z-index:20;border-radius:10px;padding:7px 10px;font-size:9px;font-weight:800;letter-spacing:.12em;color:#0a728c;pointer-events:none}
#error{display:none;position:absolute;inset:0;z-index:40;background:#f4f7f8;color:#a43d3d;place-items:center;text-align:center;padding:40px}#error b{display:block;color:#19333e;margin-bottom:8px}
.maplibregl-popup-content{background:rgba(255,255,255,.96);border:1px solid rgba(10,130,159,.22);border-radius:12px;padding:10px 13px;color:#19333e;font:11px/1.7 Inter,-apple-system,sans-serif;box-shadow:0 12px 34px rgba(39,66,76,.2);max-width:320px}.maplibregl-popup-tip{display:none}.tt-time{color:#0a829f;font-weight:750;letter-spacing:.04em}
@media(max-width:700px){#hud{top:10px;left:10px;padding:10px 12px;min-width:150px}#controls{top:10px;right:10px}.map-btn{width:34px;height:34px}#footer{left:10px;bottom:10px}#zoom{right:10px;bottom:10px}}
</style></head><body>
<div id="map"></div><div id="veil"><div class="loader"><div class="loader-mark"></div>assembling atlas</div></div>
<div id="hud" class="glass"><div class="hud-top"><i class="pulse"></i><span id="mode">TRACK LENS</span></div><div id="summary">正在加载轨迹</div><div id="subsummary">Adaptive projection</div></div>
<div id="controls"><button class="map-btn" id="plus" title="放大">+</button><button class="map-btn" id="minus" title="缩小">−</button><button class="map-btn" id="fit" title="适应当前数据">⌖</button></div>
<div id="footer" class="glass">BASEMAP · <span id="provider">—</span> · <span id="attribution"></span></div><div id="zoom" class="glass">Z 4.0</div>
<div id="error"><div><b>地图渲染器未能启动</b><span id="error-message">请检查网络连接。</span></div></div>
<script src="https://unpkg.com/maplibre-gl@5.24.0/dist/maplibre-gl.js"></script>
<script>
(function(){
'use strict';
var map=null,current=null,lastFilterKey=null,urlTimer=null,popup=null,baseTile=null,handlersReady=false;
var EMPTY={type:'FeatureCollection',features:[]};
function text(id,value){document.getElementById(id).textContent=value==null?'':String(value)}
function esc(value){return String(value==null?'':value).replace(/[&<>"']/g,function(c){return{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]})}
function rgba(c){return c&&c.length>=3?'rgba('+c[0]+','+c[1]+','+c[2]+','+((c[3]==null?255:c[3])/255)+')':'rgba(103,232,249,.8)'}
function featureCollection(features){return{type:'FeatureCollection',features:features}}
function pointFeature(coordinates,properties){return{type:'Feature',geometry:{type:'Point',coordinates:coordinates},properties:properties||{}}}
function lineFeature(coordinates,properties){return{type:'Feature',geometry:{type:'LineString',coordinates:coordinates},properties:properties||{}}}
function showError(message){document.getElementById('error').style.display='grid';text('error-message',message||'请检查浏览器控制台。')}
if(typeof maplibregl==='undefined'){showError('MapLibre CDN 加载失败，请确认网络可以访问 unpkg.com。');return}

function styleFor(data){return{
 version:8,projection:{type:'globe'},
 sources:{basemap:{type:'raster',tiles:[data.tileUrl],tileSize:256,attribution:(data.meta||{}).attribution||''}},
 layers:[{id:'space',type:'background',paint:{'background-color':'#dfe9ed'}},{id:'basemap',type:'raster',source:'basemap',paint:{'raster-opacity':.96,'raster-fade-duration':220}}],
 sky:{'atmosphere-blend':['interpolate',['linear'],['zoom'],0,1,5,1,7,0],'horizon-color':'#c8e2e9','horizon-fog-blend':.08,'fog-color':'#eef5f7','sky-color':'#dbe9ee','sky-horizon-blend':.35}
}}
function heatGeo(data){return featureCollection((data.heatmapData||[]).map(function(p){return pointFeature(p,{weight:1})}))}
function pathGeo(items){return featureCollection((items||[]).map(function(p){return lineFeature(p.path,{color:rgba(p.color),label:p.label||''})}))}
function scatterGeo(data){var c=data.scatterData,features=[];if(!c||!c.n)return featureCollection(features);for(var i=0;i<c.n;i++){var o=i*4;features.push(pointFeature([c.lons[i],c.lats[i]],{color:rgba([c.colors[o],c.colors[o+1],c.colors[o+2],c.colors[o+3]]),radius:c.radii[i],time:c.ts_fmts[i],speed:c.speed_fmts[i],accuracy:c.accuracies[i]}))}return featureCollection(features)}
function photoGeo(items){return featureCollection((items||[]).map(function(p){return pointFeature([p.longitude,p.latitude],{color:rgba(p.color),time:p.ts_fmt,accuracy:p.accuracy})}))}
function airportGeo(items){return featureCollection((items||[]).map(function(p){return pointFeature([p.longitude,p.latitude],{iata:p.iata,city:p.city})}))}
function airborneGeo(data){var review=data.airborneData||{},features=[];(review.paths||[]).forEach(function(p){features.push(lineFeature(p.path,{label:p.label||'',timeBasis:p.timeBasis||''}))});(review.points||[]).forEach(function(p){features.push(pointFeature(p.coordinates,{label:p.label||'',time:p.time||'',speed:p.speed||'',altitude:p.altitude||'',timeBasis:p.timeBasis||''}))});return featureCollection(features)}
function setSource(id,data){var source=map.getSource(id);if(source)source.setData(data);else map.addSource(id,{type:'geojson',data:data})}
function addLayer(layer){if(!map.getLayer(layer.id))map.addLayer(layer)}
function installDataLayers(data){
 setSource('heat-data',heatGeo(data));setSource('path-data',pathGeo(data.pathData));setSource('scatter-data',scatterGeo(data));setSource('photo-data',photoGeo(data.photoScatter));setSource('flight-data',pathGeo(data.flightPathData));setSource('airport-data',airportGeo(data.airportData));setSource('airborne-data',airborneGeo(data));
 addLayer({id:'heatmap',type:'heatmap',source:'heat-data',maxzoom:13,paint:{'heatmap-weight':1,'heatmap-intensity':['interpolate',['linear'],['zoom'],0,.45,10,1.3],'heatmap-radius':['interpolate',['linear'],['zoom'],0,4,10,30],'heatmap-opacity':['interpolate',['linear'],['zoom'],0,.72,12,.95],'heatmap-color':['interpolate',['linear'],['heatmap-density'],0,'rgba(7,18,35,0)',.18,'rgba(24,78,119,.55)',.38,'rgba(23,190,187,.72)',.58,'rgba(103,232,249,.82)',.78,'rgba(250,204,21,.9)',1,'rgba(255,248,218,1)']}});
 addLayer({id:'paths',type:'line',source:'path-data',layout:{'line-join':'round','line-cap':'round'},paint:{'line-color':['get','color'],'line-width':['interpolate',['linear'],['zoom'],0,.65,6,1.6,13,4.2],'line-opacity':.9}});
 addLayer({id:'scatter',type:'circle',source:'scatter-data',paint:{'circle-color':['get','color'],'circle-radius':['interpolate',['linear'],['zoom'],0,1.5,8,2.5,14,['min',10,['max',3,['*',['get','radius'],.5]]]],'circle-stroke-color':'rgba(225,255,255,.38)','circle-stroke-width':['interpolate',['linear'],['zoom'],5,0,12,.5]}});
 addLayer({id:'photo-scatter',type:'circle',source:'photo-data',paint:{'circle-color':['get','color'],'circle-radius':['interpolate',['linear'],['zoom'],0,2,12,5.5],'circle-stroke-color':'#f4e9ff','circle-stroke-width':1}});
 addLayer({id:'flights',type:'line',source:'flight-data',layout:{'line-join':'round','line-cap':'round'},paint:{'line-color':['get','color'],'line-width':['interpolate',['linear'],['zoom'],0,1,8,2.4],'line-opacity':.38}});
 addLayer({id:'airborne-lines',type:'line',source:'airborne-data',filter:['==',['geometry-type'],'LineString'],layout:{'line-join':'round','line-cap':'round'},paint:{'line-color':'#d12cab','line-width':['interpolate',['linear'],['zoom'],0,1.2,8,3.2,14,5],'line-opacity':.88,'line-dasharray':[2,1.35]}});
 addLayer({id:'airports',type:'circle',source:'airport-data',paint:{'circle-color':'#ffd778','circle-radius':['interpolate',['linear'],['zoom'],0,2,10,5],'circle-stroke-color':'rgba(255,250,220,.85)','circle-stroke-width':1}});
 addLayer({id:'airborne-points',type:'circle',source:'airborne-data',filter:['==',['geometry-type'],'Point'],paint:{'circle-color':'#d12cab','circle-opacity':.78,'circle-radius':['interpolate',['linear'],['zoom'],0,1.5,8,2.7,14,4.5],'circle-stroke-color':'rgba(255,255,255,.88)','circle-stroke-width':['interpolate',['linear'],['zoom'],5,0,12,1]}});
 if(!handlersReady){installHandlers();handlersReady=true}
}
function tooltipHtml(layer,p){if(layer==='scatter')return'<div class="tt-time">'+esc(p.time)+'</div><div>速度&nbsp; '+esc(p.speed)+'</div><div>精度&nbsp; '+Number(p.accuracy).toFixed(0)+' m</div>';if(layer==='photo-scatter')return'<div class="tt-time">'+esc(p.time)+'</div><div>照片位置</div><div>精度&nbsp; '+Number(p.accuracy).toFixed(0)+' m</div>';if(layer==='flights')return'<div class="tt-time">FLIGHT</div><div>'+esc(p.label)+'</div>';if(layer==='airports')return'<div class="tt-time">'+esc(p.iata)+'</div><div>'+esc(p.city)+'</div>';if(layer==='airborne-lines')return'<div class="tt-time">疑似机上 GPS</div><div>'+esc(p.label)+'</div><div>'+esc(p.timeBasis)+'</div>';if(layer==='airborne-points')return'<div class="tt-time">'+esc(p.time)+'</div><div>'+esc(p.label)+'</div><div>速度&nbsp; '+esc(p.speed)+' · 高度&nbsp; '+esc(p.altitude)+'</div><div>'+esc(p.timeBasis)+'</div>';return''}
function installHandlers(){['scatter','photo-scatter','flights','airports','airborne-lines','airborne-points'].forEach(function(id){map.on('mouseenter',id,function(){map.getCanvas().style.cursor='pointer'});map.on('mouseleave',id,function(){map.getCanvas().style.cursor='';if(popup)popup.remove()});map.on('mousemove',id,function(e){if(!e.features||!e.features.length)return;var html=tooltipHtml(id,e.features[0].properties||{});if(!html)return;if(popup)popup.remove();popup=new maplibregl.Popup({closeButton:false,closeOnClick:false,offset:12}).setLngLat(e.lngLat).setHTML(html).addTo(map)})})}
function orbital(){return map&&map.getZoom()<5.5}
function updateHud(data){var m=data.meta||{};text('mode',orbital()?'ORBITAL VIEW':(m.mode||'TRACK LENS'));text('summary',m.summary||'');text('subsummary',m.detail||'');text('provider',m.provider||'CARTO');text('attribution',m.attribution||'')}
function persist(){try{var center=map.getCenter(),url=new URL(window.parent.location.href);url.searchParams.set('map_lon',center.lng.toFixed(5));url.searchParams.set('map_lat',center.lat.toFixed(5));url.searchParams.set('map_zoom',map.getZoom().toFixed(2));window.parent.history.replaceState(null,'',url.toString());window.parent._trackLensVP={longitude:center.lng,latitude:center.lat,zoom:map.getZoom()}}catch(ignore){}}
function camera(view,duration){map.easeTo({center:[view.longitude,view.latitude],zoom:view.zoom,duration:duration||900,easing:function(t){return 1-Math.pow(1-t,3)},essential:true})}
function onMove(){text('zoom','Z '+map.getZoom().toFixed(1));updateHud(current);clearTimeout(urlTimer);urlTimer=setTimeout(persist,260)}
function init(data){
 var saved=null;try{saved=window.parent._trackLensVP||null}catch(ignore){}var view=saved||data.viewport;baseTile=data.tileUrl;
 map=new maplibregl.Map({container:'map',style:styleFor(data),center:[view.longitude,view.latitude],zoom:view.zoom,minZoom:1.15,maxZoom:20,attributionControl:false,renderWorldCopies:true,fadeDuration:220});
 map.on('move',onMove);map.on('load',function(){installDataLayers(data);updateHud(data);text('zoom','Z '+map.getZoom().toFixed(1));requestAnimationFrame(function(){document.getElementById('veil').style.opacity='0'})});
 map.on('error',function(e){if(e&&e.error&&/style|source/i.test(e.error.message||''))console.warn('Map source:',e.error.message)});
 document.getElementById('plus').onclick=function(){map.easeTo({zoom:Math.min(20,map.getZoom()+1),duration:520,easing:function(t){return 1-Math.pow(1-t,3)}})};
 document.getElementById('minus').onclick=function(){map.easeTo({zoom:Math.max(1.15,map.getZoom()-1),duration:520,easing:function(t){return 1-Math.pow(1-t,3)}})};
 document.getElementById('fit').onclick=function(){camera(current.dataViewport,850)};
}
function apply(data){current=data;if(!map){lastFilterKey=data.filterKey||null;init(data);return}updateHud(data);if(data.tileUrl!==baseTile){baseTile=data.tileUrl;handlersReady=false;map.setStyle(styleFor(data));map.once('style.load',function(){installDataLayers(data)})}else if(map.isStyleLoaded()){installDataLayers(data)}else{map.once('load',function(){installDataLayers(data)})}if(data.filterKey&&data.filterKey!==lastFilterKey){lastFilterKey=data.filterKey;try{window.parent._trackLensVP=null}catch(ignore){}camera(data.dataViewport,1000)}}
window.addEventListener('message',function(e){if(e.data&&e.data.type==='tracklens-update'){var d=null;try{d=window.parent._trackLensData}catch(ignore){}if(d)apply(d)}});try{window.parent._trackLensMapFrame=window}catch(ignore){}(function wait(){var d=null;try{d=window.parent._trackLensData}catch(ignore){}if(d)apply(d);else setTimeout(wait,40)})();
})();
</script></body></html>"""


_MESSENGER_HTML = r"""<script>
window._trackLensData=__PAYLOAD__;
try{var target=window._trackLensMapFrame;if(target){target.postMessage({type:'tracklens-update'},'*');}}catch(ignore){}
</script>"""


def render_flat_map(payload: dict, height: int = 720) -> None:
    messenger = _MESSENGER_HTML.replace("__PAYLOAD__", _dumps(payload))
    st.html(messenger, unsafe_allow_javascript=True)
    st.iframe(_MAP_HTML, height=height)


def render_globe_map(payload: dict, height: int = 720) -> None:
    globe_payload = deepcopy(payload)
    globe_payload["viewport"] = {**payload["viewport"], "zoom": min(payload["viewport"]["zoom"], 1.35)}
    globe_payload.setdefault("meta", {})["forceGlobe"] = True
    render_flat_map(globe_payload, height=height)
