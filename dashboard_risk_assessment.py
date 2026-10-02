"""
RISK ASSESSMENT & VULNERABILITY DASHBOARD
==========================================
Re-purposes the Wildfire CNN model to serve as a 
Regional Risk & Vulnerability Assessment system.
"""

import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
import socketserver
from http.server import SimpleHTTPRequestHandler
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Force UTF-8 output on Windows
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
EXP_DIR    = os.path.join(BASE_DIR, "data",   "experimental_v1")
MODEL_PATH = os.path.join(BASE_DIR, "models", "experimental_v1", "cnn_model.h5")
RESULTS_DIR = os.path.join(BASE_DIR, "results", "experimental_v1", "CNN_1D")
OUT_DIR    = os.path.join(BASE_DIR, "models", "experimental_v1")
os.makedirs(OUT_DIR, exist_ok=True)

HOST, PORT = "localhost", 8080 

# ── Uttarakhand region bounding boxes ────────────────────────────────────────
REGION_BOXES = {
    "garhwal": {"lon": (78.5, 80.5), "lat": (29.5, 31.0)},
    "kumaon":  {"lon": (79.0, 80.5), "lat": (28.5, 30.5)},
    "terai":   {"lon": (78.5, 80.0), "lat": (28.5, 29.5)},
}

# ── HTML Template ─────────────────────────────────────────────────────────────
HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Uttarakhand · Risk Watch</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
  <link rel="stylesheet" href="https://unpkg.com/leaflet.markercluster@1.4.1/dist/MarkerCluster.css"/>
  <link rel="stylesheet" href="https://unpkg.com/leaflet.markercluster@1.4.1/dist/MarkerCluster.Default.css"/>
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
  <style>
    :root {
      --extreme: #ef4444; --moderate: #f59e0b; --stable: #10b981; --cloud: #64748b;
      --indigo: #6366f1; --bg: #020617; --surface: #0f172a; --surface2: #1e293b;
      --border: rgba(255,255,255,0.08); --text: #f1f5f9; --muted: #94a3b8;
    }
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: 'Plus Jakarta Sans', sans-serif; background: var(--bg); color: var(--text); height: 100vh; display: flex; flex-direction: column; overflow: hidden; }
    #hdr { background: rgba(15, 23, 42, 0.8); backdrop-filter: blur(12px); border-bottom: 1px solid var(--border); padding: 0 32px; height: 70px; display: flex; align-items: center; justify-content: space-between; flex-shrink: 0; z-index: 1000; }
    #hdr .brand { display: flex; align-items: center; gap: 14px; }
    #hdr .icon-box { width: 40px; height: 40px; background: linear-gradient(135deg, var(--extreme), var(--moderate)); border-radius: 10px; display: flex; align-items: center; justify-content: center; font-size: 20px; box-shadow: 0 0 20px rgba(239, 68, 68, 0.3); }
    #hdr h1 { font-size: 18px; font-weight: 800; letter-spacing: -0.02em; }
    #hdr h1 span { color: var(--extreme); }
    #hdr p { font-size: 11px; color: var(--muted); margin-top: 1px; text-transform: uppercase; letter-spacing: 0.05em; }
    .badge-row { display: flex; gap: 12px; }
    .badge { background: rgba(255,255,255,0.03); border: 1px solid var(--border); border-radius: 8px; padding: 6px 16px; font-size: 11px; font-weight: 700; color: var(--text); display: flex; align-items: center; gap: 8px; }
    .badge .dot { width: 6px; height: 6px; border-radius: 50%; }
    #layout { display: flex; flex: 1; overflow: hidden; }
    #map { flex: 1; position: relative; }
    #map-topbar { position: absolute; top: 20px; left: 50%; transform: translateX(-50%); z-index: 900; display: flex; gap: 10px; background: rgba(2, 6, 23, 0.7); backdrop-filter: blur(16px); border: 1px solid var(--border); border-radius: 14px; padding: 10px 14px; box-shadow: 0 10px 40px rgba(0,0,0,0.4); }
    .filter-chip { font-size: 11px; font-weight: 700; padding: 6px 14px; border-radius: 8px; border: 1px solid transparent; cursor: pointer; transition: all 0.2s ease; background: rgba(255,255,255,0.04); color: var(--muted); }
    .filter-chip.on { color: #fff; }
    .filter-chip[data-cls="extreme"].on { background: var(--extreme); }
    .filter-chip[data-cls="moderate"].on { background: var(--moderate); }
    .filter-chip[data-cls="cloud"].on { background: var(--cloud); }
    .filter-chip[data-cls="stable"].on { background: var(--stable); }
    #sidebar { width: 360px; background: var(--surface); border-left: 1px solid var(--border); overflow-y: auto; flex-shrink: 0; padding: 24px 20px; display: flex; flex-direction: column; gap: 20px; }
    .card { background: rgba(30, 41, 59, 0.4); border: 1px solid var(--border); border-radius: 18px; padding: 18px; }
    .card-title { font-size: 11px; font-weight: 800; text-transform: uppercase; letter-spacing: 0.1em; color: var(--muted); margin-bottom: 16px; display: flex; align-items: center; gap: 10px; }
    .stats { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
    .stat { background: rgba(255,255,255,0.02); border: 1px solid var(--border); border-radius: 14px; padding: 14px 12px; text-align: center; }
    .stat-val { font-size: 26px; font-weight: 800; line-height: 1; margin-bottom: 4px; }
    .stat-lbl { font-size: 10px; color: var(--muted); font-weight: 700; }
    .c-extreme { color: var(--extreme); } .c-moderate { color: var(--moderate); } .c-stable { color: var(--stable); }
    .metric-row { margin-bottom: 12px; }
    .metric-label { display: flex; justify-content: space-between; font-size: 12px; font-weight: 600; color: #94a3b8; margin-bottom: 6px; }
    .metric-label span { color: var(--text); font-weight: 800; }
    .bar-track { height: 6px; background: rgba(255,255,255,0.05); border-radius: 99px; overflow: hidden; }
    .bar-fill { height: 100%; border-radius: 99px; transition: width 1.2s ease-out; }
    .info-box { background: rgba(99, 102, 241, 0.05); border: 1px dashed rgba(99, 102, 241, 0.2); border-radius: 12px; padding: 12px; font-size: 11px; color: #cbd5e1; line-height: 1.4; }
    .chart-wrap { position: relative; }
    .marker-cluster div { color: #fff; font-weight: 700; font-family: inherit; font-size: 12px; }
    @keyframes pulse-risk { 0% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0.6); } 100% { box-shadow: 0 0 0 12px rgba(239, 68, 68, 0); } }
    .risk-pulse { animation: pulse-risk 2s infinite ease-out; border-radius: 50%; }
  </style>
</head>
<body>
<div id="hdr">
  <div class="brand"><div class="icon-box">🛡️</div><div><h1>Uttarakhand · <span>Risk</span> Watch</h1><p>Environmental Vulnerability Monitoring</p></div></div>
  <div class="badge-row">
    <div class="badge"><div class="dot" style="background:var(--indigo)"></div>CNN Model</div>
    <div class="badge"><div class="dot" style="background:var(--stable)"></div>Acc: __ACCURACY__%</div>
  </div>
</div>
<div id="layout">
  <div id="map"><div id="map-topbar"><button class="filter-chip on" data-cls="extreme">🔴 Extreme Risk</button><button class="filter-chip on" data-cls="moderate">🟠 Moderate Risk</button><button class="filter-chip" data-cls="cloud">⚪ Cloud Mask</button><button class="filter-chip" data-cls="stable">🟢 Stable Zone</button></div></div>
  <div id="sidebar">
    <div class="card"><div class="card-title">🛡️ Risk Susceptibility</div><div class="stats"><div class="stat"><div class="stat-lbl">EXTREME</div><div class="stat-val c-extreme" id="cnt-fire">–</div></div><div class="stat"><div class="stat-lbl">MODERATE</div><div class="stat-val c-moderate" id="cnt-smoke">–</div></div></div><div style="margin-top:12px;" class="info-box"><b>Extreme Risk</b> zones represent thermal anomalies potentially primed for ignition.</div></div>
    <div class="card"><div class="card-title">📊 Assessment Reliability</div>
      <div class="metric-row"><div class="metric-label">Precision <span>__PRECISION__%</span></div><div class="bar-track"><div class="bar-fill" style="width:__PRECISION__%;background:var(--stable)"></div></div></div>
      <div class="metric-row"><div class="metric-label">Recall <span>__RECALL__%</span></div><div class="bar-track"><div class="bar-fill" style="width:__RECALL__%;background:var(--moderate)"></div></div></div>
    </div>
    <div class="card"><div class="card-title">🥧 Territorial Split</div><div class="chart-wrap" style="height:150px"><canvas id="pieChart"></canvas></div></div>
    <div class="card"><div class="card-title">🗺️ Regional Distribution</div><div class="chart-wrap" style="height:130px"><canvas id="regionChart"></canvas></div></div>
  </div>
</div>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<script src="https://unpkg.com/leaflet.markercluster@1.4.1/dist/leaflet.markercluster.js"></script>
<script>
const PREDICTIONS = __PREDICTIONS_JSON__;
const REGION_DATA = __REGION_JSON__;
const map = L.map('map', { zoomControl: false, preferCanvas: true }).setView([29.9, 79.5], 8);
L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', { attribution:'&copy; CARTO' }).addTo(map);
const cluster = L.markerClusterGroup({ maxClusterRadius: 40, showCoverageOnHover: false });
const COLORS = { extreme:'#f43f5e', moderate:'#f59e0b', cloud:'#64748b', stable:'#10b981' };
const CLASS_MAP = { 0:'stable', 1:'moderate', 2:'cloud', 3:'extreme' };
let activeFilters = new Set(['extreme','moderate']);
function refreshMap() {
  cluster.clearLayers();
  const counts = { extreme:0, moderate:0, cloud:0, stable:0 };
  PREDICTIONS.forEach(p => {
    const name = CLASS_MAP[p.cls];
    if (!activeFilters.has(name) || p.prob < 0.45) return;
    counts[name]++;
    const m = L.circleMarker([p.lat, p.lon], { radius: name === 'extreme' ? 7 : 5, fillColor: COLORS[name], weight: 0, fillOpacity: 0.9, className: name === 'extreme' ? 'risk-pulse' : '' });
    m.bindPopup(`<b style="color:${COLORS[name]}">${name.toUpperCase()} RISK</b><br>Confidence: ${(p.prob*100).toFixed(1)}%`);
    cluster.addLayer(m);
  });
  map.addLayer(cluster);
  document.getElementById('cnt-fire').textContent = counts.extreme;
  document.getElementById('cnt-smoke').textContent = counts.moderate;
}
document.querySelectorAll('.filter-chip').forEach(btn => { btn.addEventListener('click', () => { const cls = btn.dataset.cls; if (activeFilters.has(cls)) { activeFilters.delete(cls); btn.classList.remove('on'); } else { activeFilters.add(cls); btn.classList.add('on'); } refreshMap(); }); });
new Chart(document.getElementById('pieChart'), { type:'doughnut', data:{ labels:['Stable','Moderate','Cloud','Extreme'], datasets:[{ data:[PREDICTIONS.filter(p=>p.cls===0).length, PREDICTIONS.filter(p=>p.cls===1).length, PREDICTIONS.filter(p=>p.cls===2).length, PREDICTIONS.filter(p=>p.cls===3).length], backgroundColor:[COLORS.stable, COLORS.moderate, COLORS.cloud, COLORS.extreme], borderWidth:0 }] }, options:{ responsive:true, cutout:'70%', plugins:{ legend:{ position:'right', labels:{ color:'#94a3b8', font:{size:10} } } } } });
new Chart(document.getElementById('regionChart'), { type:'bar', data:{ labels:Object.keys(REGION_DATA).map(r=>r.toUpperCase()), datasets:[{ data:Object.values(REGION_DATA), backgroundColor:['#6366f1','#f59e0b','#10b981'], borderRadius:6 }] }, options:{ responsive:true, scales:{ y:{ beginAtZero:true, grid:{ color:'rgba(255,255,255,0.04)' } }, x:{ grid:{ display:false } } }, plugins:{ legend:{ display:false } } } });
refreshMap();
</script>
</body>
</html>
"""

def assign_region(idx: int, total: int) -> str:
    t = idx / total
    if t < 0.50: return "garhwal"
    elif t < 0.80: return "kumaon"
    else: return "terai"

def synthesise_coords(region: str, rng: np.random.Generator) -> tuple[float, float]:
    box = REGION_BOXES[region]
    lat = rng.uniform(*box["lat"])
    lon = rng.uniform(*box["lon"])
    return round(float(lat), 5), round(float(lon), 5)

def main():
    print("\n[PROCESS] Generating Risk Dashboard...")
    try:
        from tensorflow.keras.models import load_model
        model = load_model(MODEL_PATH, compile=False)
    except Exception as e:
        print(f"[ERROR] Model load failed: {e}"); return

    test_df = pd.read_csv(os.path.join(EXP_DIR, "test_scaled.csv"))
    X_raw = test_df.drop(columns=["class"]).values
    X_cnn = X_raw.reshape(X_raw.shape[0], X_raw.shape[1], 1)

    probs_2d = model.predict(X_cnn, batch_size=254, verbose=0)
    y_pred = np.argmax(probs_2d, axis=1)
    max_prob = np.max(probs_2d, axis=1)

    # Metrics loading with fallback
    metrics_path = os.path.join(RESULTS_DIR, "metrics.json")
    acc, prec, rec = 0, 0, 0
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            m_data = json.load(f)
            acc = m_data.get("accuracy", 0)
            wa = m_data.get("weighted_avg", {})
            prec = wa.get("precision", m_data.get("precision", 0))
            rec = wa.get("recall", m_data.get("recall", 0))

    rng = np.random.default_rng(42)
    total = len(test_df)
    pred_list = []
    region_counts = {}
    for i, (cls, prob) in enumerate(zip(y_pred, max_prob)):
        region = assign_region(i, total)
        lat, lon = synthesise_coords(region, rng)
        pred_list.append({"cls": int(cls), "prob": round(float(prob), 4), "lat": lat, "lon": lon, "region": region})
        region_counts[region] = region_counts.get(region, 0) + 1

    # Render HTML
    rendered = HTML.replace("__PREDICTIONS_JSON__", json.dumps(pred_list[:7000]))
    rendered = rendered.replace("__REGION_JSON__", json.dumps(region_counts))
    rendered = rendered.replace("__ACCURACY__", str(round(acc*100, 1)))
    rendered = rendered.replace("__PRECISION__", str(round(prec*100, 1)))
    rendered = rendered.replace("__RECALL__", str(round(rec*100, 1)))

    out_p = os.path.join(OUT_DIR, "vulnerability_dashboard.html")
    with open(out_p, "w", encoding="utf-8") as f: f.write(rendered)
    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=OUT_DIR, **kwargs)
        def log_message(self, *args): pass

    # Start server with automatic port selection to avoid WinError 10048
    socketserver.TCPServer.allow_reuse_address = True
    current_port = PORT
    max_tries = 10
    httpd = None

    for i in range(max_tries):
        try:
            httpd = socketserver.TCPServer((HOST, current_port), Handler)
            break
        except OSError:
            print(f"[RETRY] Port {current_port} busy, trying {current_port + 1}...")
            current_port += 1

    if httpd:
        print(f"Serving at http://{HOST}:{current_port}/vulnerability_dashboard.html")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nServer stopped.")
    else:
        print(f"[ERROR] Could not find an available port after {max_tries} tries.")

if __name__ == "__main__":
    main()
