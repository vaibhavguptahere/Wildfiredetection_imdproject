"""
CNN WILDFIRE DASHBOARD
======================
Loads the trained 1D CNN model from models/experimental_v1/cnn_model.h5,
runs inference on the test split, synthesises per-region geo-coordinates
(since the scaled CSVs dropped lat/lon) and serves a premium HTML dashboard.
"""

import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
import socketserver
from http.server import SimpleHTTPRequestHandler

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

HOST, PORT = "localhost", 8050

# ── Uttarakhand region bounding boxes ────────────────────────────────────────
REGION_BOXES = {
    "garhwal": {"lon": (78.5, 80.5), "lat": (29.5, 31.0)},
    "kumaon":  {"lon": (79.0, 80.5), "lat": (28.5, 30.5)},
    "terai":   {"lon": (78.5, 80.0), "lat": (28.5, 29.5)},
}

CLASS_NAMES = {0: "clear", 1: "smoke", 2: "cloud", 3: "active_fire"}

# ── HTML Template ─────────────────────────────────────────────────────────────
HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Uttarakhand Wildfire · CNN Dashboard</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
  <link rel="stylesheet" href="https://unpkg.com/leaflet.markercluster@1.4.1/dist/MarkerCluster.css"/>
  <link rel="stylesheet" href="https://unpkg.com/leaflet.markercluster@1.4.1/dist/MarkerCluster.Default.css"/>
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
  <style>
    /* ── Design Tokens ─────────────────────────────── */
    :root {
      --fire:      #ff4040;
      --smoke:     #f59e0b;
      --cloud:     #94a3b8;
      --clear:     #22c55e;
      --indigo:    #6366f1;
      --bg:        #080d1a;
      --surface:   #0e1729;
      --surface2:  #162035;
      --border:    rgba(255,255,255,.07);
      --text:      #e2e8f0;
      --muted:     #64748b;
    }
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

    body {
      font-family: 'Inter', sans-serif;
      background: var(--bg);
      color: var(--text);
      height: 100vh;
      display: flex;
      flex-direction: column;
      overflow: hidden;
    }

    /* ── Header ──────────────────────────────────────── */
    #hdr {
      background: linear-gradient(90deg, #0b1120 0%, #111827 50%, #0b1120 100%);
      border-bottom: 1px solid var(--border);
      padding: 0 28px;
      height: 62px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      flex-shrink: 0;
      z-index: 1000;
    }
    #hdr .brand {
      display: flex;
      align-items: center;
      gap: 12px;
    }
    #hdr .flame { font-size: 26px; }
    #hdr h1 { font-size: 16px; font-weight: 800; letter-spacing: -.03em; }
    #hdr h1 span { color: var(--fire); }
    #hdr p { font-size: 11px; color: var(--muted); margin-top: 1px; }

    .badge-row { display: flex; gap: 10px; }
    .badge {
      background: rgba(255,255,255,.05);
      border: 1px solid var(--border);
      border-radius: 999px;
      padding: 5px 14px;
      font-size: 11px;
      font-weight: 600;
      color: var(--text);
      display: flex;
      align-items: center;
      gap: 6px;
    }
    .badge .dot { width: 7px; height: 7px; border-radius: 50%; }

    /* ── Layout ──────────────────────────────────────── */
    #layout { display: flex; flex: 1; overflow: hidden; }

    /* ── Map ─────────────────────────────────────────── */
    #map { flex: 1; position: relative; }

    /* Floating top-bar inside map */
    #map-topbar {
      position: absolute;
      top: 14px;
      left: 50%;
      transform: translateX(-50%);
      z-index: 900;
      display: flex;
      gap: 8px;
      background: rgba(8,13,26,.75);
      backdrop-filter: blur(14px);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 8px 12px;
    }
    .filter-chip {
      font-size: 11px;
      font-weight: 700;
      padding: 5px 13px;
      border-radius: 8px;
      border: 1px solid transparent;
      cursor: pointer;
      transition: .15s ease;
      background: rgba(255,255,255,.06);
      color: var(--muted);
    }
    .filter-chip.on { color: #fff; }
    .filter-chip[data-cls="active_fire"].on  { background: var(--fire);  border-color: var(--fire); }
    .filter-chip[data-cls="smoke"].on        { background: var(--smoke); border-color: var(--smoke); }
    .filter-chip[data-cls="cloud"].on        { background: var(--cloud); border-color: var(--cloud); }
    .filter-chip[data-cls="clear"].on        { background: var(--clear); border-color: var(--clear); }

    /* Floating legend */
    #legend {
      position: absolute;
      bottom: 30px;
      left: 18px;
      z-index: 900;
      background: rgba(8,13,26,.82);
      backdrop-filter: blur(14px);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 12px 16px;
    }
    .leg { display: flex; align-items: center; gap: 9px; font-size: 11px; font-weight: 600; margin-bottom: 7px; color: #cbd5e1; }
    .leg:last-child { margin-bottom: 0; }
    .leg-dot { width: 9px; height: 9px; border-radius: 50%; flex-shrink: 0; }

    /* ── Sidebar ─────────────────────────────────────── */
    #sidebar {
      width: 340px;
      background: var(--surface);
      border-left: 1px solid var(--border);
      overflow-y: auto;
      flex-shrink: 0;
      padding: 16px 14px;
      display: flex;
      flex-direction: column;
      gap: 12px;
    }
    #sidebar::-webkit-scrollbar { width: 4px; }
    #sidebar::-webkit-scrollbar-track { background: transparent; }
    #sidebar::-webkit-scrollbar-thumb { background: var(--border); border-radius: 4px; }

    .card {
      background: var(--surface2);
      border: 1px solid var(--border);
      border-radius: 14px;
      padding: 14px;
    }
    .card-title {
      font-size: 10px;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: .08em;
      color: var(--muted);
      margin-bottom: 12px;
      display: flex;
      align-items: center;
      gap: 7px;
    }

    /* Stat grid */
    .stats { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }
    .stat {
      background: rgba(255,255,255,.03);
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 11px 10px;
      text-align: center;
    }
    .stat-val { font-size: 24px; font-weight: 800; line-height: 1; margin-bottom: 3px; }
    .stat-lbl { font-size: 10px; color: var(--muted); font-weight: 600; }
    .c-fire   { color: var(--fire);   }
    .c-smoke  { color: var(--smoke);  }
    .c-clear  { color: var(--clear);  }
    .c-indigo { color: var(--indigo); }

    /* Metric bars */
    .metric-row { margin-bottom: 9px; }
    .metric-row:last-child { margin-bottom: 0; }
    .metric-label {
      display: flex;
      justify-content: space-between;
      font-size: 11px;
      font-weight: 600;
      color: #94a3b8;
      margin-bottom: 4px;
    }
    .metric-label span { color: var(--text); font-weight: 700; }
    .bar-track {
      height: 5px;
      background: rgba(255,255,255,.06);
      border-radius: 99px;
      overflow: hidden;
    }
    .bar-fill {
      height: 100%;
      border-radius: 99px;
      transition: width 1s cubic-bezier(.4,0,.2,1);
    }

    /* Threshold slider */
    .slider-wrap label {
      font-size: 11px;
      font-weight: 600;
      color: #94a3b8;
      display: flex;
      justify-content: space-between;
      margin-bottom: 8px;
    }
    .slider-wrap label span { color: var(--text); }
    input[type=range] {
      width: 100%;
      height: 4px;
      -webkit-appearance: none;
      background: rgba(255,255,255,.1);
      border-radius: 99px;
      outline: none;
    }
    input[type=range]::-webkit-slider-thumb {
      -webkit-appearance: none;
      width: 16px; height: 16px;
      background: var(--indigo);
      border-radius: 50%;
      cursor: pointer;
      border: 2px solid #fff;
      box-shadow: 0 0 8px rgba(99,102,241,.5);
    }

    /* Chart containers */
    .chart-wrap { position: relative; }

    /* Cluster colors */
    .marker-cluster-small     { background-color: rgba(34,197,94,.35); }
    .marker-cluster-small div { background-color: rgba(34,197,94,.8); }
    .marker-cluster-medium     { background-color: rgba(245,158,11,.35); }
    .marker-cluster-medium div { background-color: rgba(245,158,11,.8); }
    .marker-cluster-large      { background-color: rgba(255,64,64,.35); }
    .marker-cluster-large div  { background-color: rgba(255,64,64,.8); }
    .marker-cluster div { color: #fff; font-weight: 700; font-family: 'Inter', sans-serif; font-size: 12px; }

    @keyframes pulse-fire {
      0%   { box-shadow: 0 0 0 0   rgba(255,64,64,.7); }
      100% { box-shadow: 0 0 0 10px rgba(255,64,64,0); }
    }
    .fire-pulse { animation: pulse-fire 1.6s infinite; border-radius: 50%; }
  </style>
</head>
<body>

<!-- ── Header ─────────────────────────────────────────────────────────── -->
<div id="hdr">
  <div class="brand">
    <div class="flame">🔥</div>
    <div>
      <h1>Uttarakhand <span>Wildfire</span> Watch</h1>
      <p>1D-CNN · AI-powered spatiotemporal fire &amp; smoke detection</p>
    </div>
  </div>
  <div class="badge-row">
    <div class="badge"><div class="dot" style="background:var(--indigo)"></div>Model: 1D CNN</div>
    <div class="badge"><div class="dot" style="background:var(--clear)"></div>Accuracy: __ACCURACY__%</div>
    <div class="badge"><div class="dot" style="background:var(--smoke)"></div>F1: __F1__%</div>
  </div>
</div>

<!-- ── Body ───────────────────────────────────────────────────────────── -->
<div id="layout">

  <!-- MAP -->
  <div id="map">
    <!-- floating filter chips -->
    <div id="map-topbar">
      <button class="filter-chip on" data-cls="active_fire">🔥 Fire</button>
      <button class="filter-chip on" data-cls="smoke">💨 Smoke</button>
      <button class="filter-chip"    data-cls="cloud">☁️ Cloud</button>
      <button class="filter-chip"    data-cls="clear">✅ Clear</button>
    </div>
    <!-- legend -->
    <div id="legend">
      <div class="leg"><div class="leg-dot" style="background:var(--fire)"></div>Active Fire</div>
      <div class="leg"><div class="leg-dot" style="background:var(--smoke)"></div>Smoke / Smog</div>
      <div class="leg"><div class="leg-dot" style="background:var(--cloud)"></div>Cloud Cover</div>
      <div class="leg"><div class="leg-dot" style="background:var(--clear)"></div>Clear Ground</div>
    </div>
  </div>

  <!-- SIDEBAR -->
  <div id="sidebar">

    <!-- Stats -->
    <div class="card">
      <div class="card-title">🚨 Detection Summary</div>
      <div class="stats">
        <div class="stat">
          <div class="stat-lbl">Active Fires</div>
          <div class="stat-val c-fire"   id="cnt-fire">–</div>
        </div>
        <div class="stat">
          <div class="stat-lbl">Smoke / Smog</div>
          <div class="stat-val c-smoke"  id="cnt-smoke">–</div>
        </div>
        <div class="stat">
          <div class="stat-lbl">Cloud Cover</div>
          <div class="stat-val"          id="cnt-cloud">–</div>
        </div>
        <div class="stat">
          <div class="stat-lbl">Clear Pixels</div>
          <div class="stat-val c-clear"  id="cnt-clear">–</div>
        </div>
      </div>
    </div>

    <!-- CNN Performance -->
    <div class="card">
      <div class="card-title">📈 CNN Model Performance</div>

      <div class="metric-row">
        <div class="metric-label">Accuracy <span>__ACCURACY__%</span></div>
        <div class="bar-track"><div class="bar-fill" style="width:__ACCURACY__%;background:var(--indigo)"></div></div>
      </div>
      <div class="metric-row">
        <div class="metric-label">Precision <span>__PRECISION__%</span></div>
        <div class="bar-track"><div class="bar-fill" style="width:__PRECISION__%;background:var(--clear)"></div></div>
      </div>
      <div class="metric-row">
        <div class="metric-label">Recall <span>__RECALL__%</span></div>
        <div class="bar-track"><div class="bar-fill" style="width:__RECALL__%;background:var(--smoke)"></div></div>
      </div>
      <div class="metric-row">
        <div class="metric-label">F1-Score <span>__F1__%</span></div>
        <div class="bar-track"><div class="bar-fill" style="width:__F1__%;background:var(--fire)"></div></div>
      </div>
    </div>

    <!-- Threshold -->
    <div class="card">
      <div class="card-title">🎚️ Probability Threshold</div>
      <div class="slider-wrap">
        <label>Min confidence <span id="thresh-val">50%</span></label>
        <input id="thresh" type="range" min="0" max="100" value="50">
      </div>
    </div>

    <!-- Composition Doughnut -->
    <div class="card">
      <div class="card-title">📊 Class Composition</div>
      <div class="chart-wrap" style="height:160px">
        <canvas id="pieChart"></canvas>
      </div>
    </div>

    <!-- Regional Bar -->
    <div class="card">
      <div class="card-title">🗺️ Regional Distribution</div>
      <div class="chart-wrap" style="height:130px">
        <canvas id="regionChart"></canvas>
      </div>
    </div>

    <!-- Confidence Histogram -->
    <div class="card">
      <div class="card-title">📉 Confidence Distribution</div>
      <div class="chart-wrap" style="height:130px">
        <canvas id="histChart"></canvas>
      </div>
    </div>

  </div><!-- /sidebar -->
</div>

<!-- ── Scripts ────────────────────────────────────────────────────────── -->
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<script src="https://unpkg.com/leaflet.markercluster@1.4.1/dist/leaflet.markercluster.js"></script>
<script>
const PREDICTIONS   = __PREDICTIONS_JSON__;
const REGION_DATA   = __REGION_JSON__;
const CONF_HIST     = __CONF_HIST_JSON__;

// ── Map ────────────────────────────────────────────────────────────────
const map = L.map('map', { zoomControl: false, preferCanvas: true })
              .setView([29.9, 79.5], 8);
L.control.zoom({ position: 'topright' }).addTo(map);

L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
  attribution: '&copy; CARTO &copy; OSM'
}).addTo(map);

const cluster = L.markerClusterGroup({
  chunkedLoading: true,
  maxClusterRadius: 45,
  showCoverageOnHover: false,
  spiderfyOnMaxZoom: true
});

const COLORS = { active_fire:'#ff4040', smoke:'#f59e0b', cloud:'#94a3b8', clear:'#22c55e' };
const CLASS_MAP = { 0:'clear', 1:'smoke', 2:'cloud', 3:'active_fire' };

let activeFilters = new Set(['active_fire','smoke']);
let threshold = 0.5;

function refreshMap() {
  cluster.clearLayers();
  const counts = { active_fire:0, smoke:0, cloud:0, clear:0 };

  PREDICTIONS.forEach(p => {
    const name = CLASS_MAP[p.cls];
    if (!activeFilters.has(name) || p.prob < threshold) return;
    counts[name]++;

    const m = L.circleMarker([p.lat, p.lon], {
      radius: name === 'active_fire' ? 6 : 5,
      fillColor: COLORS[name],
      color: 'rgba(255,255,255,.25)',
      weight: 1,
      fillOpacity: .88,
      className: name === 'active_fire' && p.prob > .75 ? 'fire-pulse' : ''
    });

    const label = name === 'active_fire' ? '🔥 Active Fire'
                : name === 'smoke'       ? '💨 Smoke'
                : name === 'cloud'       ? '☁️ Cloud'
                :                          '✅ Clear';
    m.bindPopup(`
      <div style="font-family:'Inter',sans-serif;padding:4px 2px;min-width:160px;">
        <div style="color:${COLORS[name]};font-weight:800;font-size:13px;margin-bottom:6px;">${label}</div>
        <div style="font-size:11px;color:#94a3b8;line-height:1.8;">
          <b style="color:#e2e8f0">Region:</b> ${p.region}<br>
          <b style="color:#e2e8f0">Confidence:</b> <span style="color:${COLORS[name]};font-weight:700;">${(p.prob*100).toFixed(1)}%</span><br>
          <b style="color:#e2e8f0">Coords:</b> ${p.lat.toFixed(4)}, ${p.lon.toFixed(4)}
        </div>
      </div>
    `);
    cluster.addLayer(m);
  });

  map.addLayer(cluster);
  document.getElementById('cnt-fire').textContent  = counts.active_fire.toLocaleString();
  document.getElementById('cnt-smoke').textContent = counts.smoke.toLocaleString();
  document.getElementById('cnt-cloud').textContent = counts.cloud.toLocaleString();
  document.getElementById('cnt-clear').textContent = counts.clear.toLocaleString();
}

// Filter chips
document.querySelectorAll('.filter-chip').forEach(btn => {
  btn.addEventListener('click', () => {
    const cls = btn.dataset.cls;
    if (activeFilters.has(cls)) { activeFilters.delete(cls); btn.classList.remove('on'); }
    else { activeFilters.add(cls); btn.classList.add('on'); }
    refreshMap();
  });
});

// Slider
document.getElementById('thresh').addEventListener('input', e => {
  threshold = e.target.value / 100;
  document.getElementById('thresh-val').textContent = e.target.value + '%';
  refreshMap();
});

// ── Charts ─────────────────────────────────────────────────────────────
const chartDefaults = {
  color: '#94a3b8',
  font: { family: "'Inter', sans-serif" }
};
Chart.defaults.color = chartDefaults.color;
Chart.defaults.font.family = chartDefaults.font.family;

// Doughnut
new Chart(document.getElementById('pieChart'), {
  type: 'doughnut',
  data: {
    labels: ['Clear', 'Smoke', 'Cloud', 'Active Fire'],
    datasets: [{
      data: [
        PREDICTIONS.filter(p=>p.cls===0).length,
        PREDICTIONS.filter(p=>p.cls===1).length,
        PREDICTIONS.filter(p=>p.cls===2).length,
        PREDICTIONS.filter(p=>p.cls===3).length,
      ],
      backgroundColor: ['#22c55e','#f59e0b','#94a3b8','#ff4040'],
      borderWidth: 0,
      hoverOffset: 8
    }]
  },
  options: {
    responsive: true, maintainAspectRatio: false,
    cutout: '65%',
    plugins: {
      legend: { position:'right', labels:{ boxWidth:10, font:{ size:10, weight:600 }, color:'#94a3b8' } }
    }
  }
});

// Regional bar
new Chart(document.getElementById('regionChart'), {
  type: 'bar',
  data: {
    labels: Object.keys(REGION_DATA).map(r => r.charAt(0).toUpperCase()+r.slice(1)),
    datasets: [{
      label: 'Points',
      data: Object.values(REGION_DATA),
      backgroundColor: ['#6366f1','#f59e0b','#22c55e'],
      borderRadius: 5,
      borderSkipped: false
    }]
  },
  options: {
    responsive: true, maintainAspectRatio: false,
    scales: {
      y: { beginAtZero:true, grid:{ color:'rgba(255,255,255,.04)' }, ticks:{ font:{size:9} } },
      x: { grid:{ display:false }, ticks:{ font:{size:10} } }
    },
    plugins: { legend:{ display:false } }
  }
});

// Confidence histogram
new Chart(document.getElementById('histChart'), {
  type: 'bar',
  data: {
    labels: CONF_HIST.map(b => b.label),
    datasets: [{
      label: 'Count',
      data: CONF_HIST.map(b => b.count),
      backgroundColor: CONF_HIST.map(b => {
        const v = b.mid;
        if (v < .4) return '#94a3b8';
        if (v < .6) return '#f59e0b';
        return '#ff4040';
      }),
      borderRadius: 3,
      borderSkipped: false
    }]
  },
  options: {
    responsive: true, maintainAspectRatio: false,
    scales: {
      y: { beginAtZero:true, grid:{ color:'rgba(255,255,255,.04)' }, ticks:{ font:{size:9} } },
      x: { grid:{ display:false }, ticks:{ font:{size:9} } }
    },
    plugins: { legend:{ display:false } }
  }
});

refreshMap();
</script>
</body>
</html>
"""

# ── Data generation helpers ──────────────────────────────────────────────────
def assign_region(idx: int, total: int) -> str:
    """Assign a region label based on sample index (deterministic split)."""
    t = idx / total
    if t < 0.50:
        return "garhwal"
    elif t < 0.80:
        return "kumaon"
    else:
        return "terai"


def synthesise_coords(region: str, rng: np.random.Generator) -> tuple[float, float]:
    """Generate a plausible lat/lon within a region bounding box."""
    box = REGION_BOXES[region]
    lat = rng.uniform(*box["lat"])
    lon = rng.uniform(*box["lon"])
    return round(float(lat), 5), round(float(lon), 5)


def build_confidence_histogram(probs: np.ndarray, n_bins: int = 10) -> list[dict]:
    bins = np.linspace(0, 1, n_bins + 1)
    hist = []
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        count = int(((probs >= lo) & (probs < hi)).sum())
        hist.append({"label": f"{int(lo*100)}-{int(hi*100)}%", "mid": (lo + hi) / 2, "count": count})
    return hist


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("  CNN WILDFIRE DASHBOARD GENERATOR")
    print("=" * 70)

    # 1. Load model
    print("\n[1/4] Loading CNN model...")
    try:
        from tensorflow.keras.models import load_model
        model = load_model(MODEL_PATH, compile=False)
        print(f"      ✓ Loaded: {MODEL_PATH}")
    except Exception as e:
        print(f"      ✗ Failed to load model: {e}")
        return

    # 2. Load test data + label encoder
    print("[2/4] Loading test data...")
    test_df = pd.read_csv(os.path.join(EXP_DIR, "test_scaled.csv"))
    with open(os.path.join(EXP_DIR, "label_encoder.pkl"), "rb") as f:
        le = pickle.load(f)

    X_raw = test_df.drop(columns=["class"]).values
    y_true = test_df["class"].values
    X_cnn = X_raw.reshape(X_raw.shape[0], X_raw.shape[1], 1)
    print(f"      ✓ {len(test_df):,} samples | {X_raw.shape[1]} features")

    # 3. Run inference
    print("[3/4] Running CNN inference...")
    probs_2d = model.predict(X_cnn, batch_size=256, verbose=0)
    y_pred   = np.argmax(probs_2d, axis=1)
    max_prob = np.max(probs_2d, axis=1)
    print(f"      ✓ Inference complete.")

    # 4. Load stored metrics
    metrics_path = os.path.join(RESULTS_DIR, "metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            metrics = json.load(f)
    else:
        from sklearn.metrics import accuracy_score, precision_recall_fscore_support
        acc = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted")
        metrics = {"accuracy": acc, "precision": precision, "recall": recall, "f1_score": f1}

    acc_pct  = round(metrics["accuracy"]  * 100, 1)
    prec_pct = round(metrics["precision"] * 100, 1)
    rec_pct  = round(metrics["recall"]    * 100, 1)
    f1_pct   = round(metrics["f1_score"]  * 100, 1)
    print(f"      ✓ Accuracy: {acc_pct}%  |  F1: {f1_pct}%")

    # 5. Build prediction JSON (cap at 8 000 for performance)
    print("[4/4] Building dashboard payload...")
    rng = np.random.default_rng(42)
    total = len(test_df)

    pred_list = []
    region_counts: dict[str, int] = {}
    for i, (cls, prob) in enumerate(zip(y_pred, max_prob)):
        region = assign_region(i, total)
        lat, lon = synthesise_coords(region, rng)
        pred_list.append({
            "cls": int(cls), "prob": round(float(prob), 4),
            "lat": lat, "lon": lon, "region": region
        })
        region_counts[region] = region_counts.get(region, 0) + 1

    # Sort by prob desc, cap at 8 000
    pred_list.sort(key=lambda x: x["prob"], reverse=True)
    if len(pred_list) > 8000:
        pred_list = pred_list[:8000]

    conf_hist = build_confidence_histogram(max_prob)

    # 6. Render HTML
    html = HTML.replace("__PREDICTIONS_JSON__", json.dumps(pred_list))
    html = html.replace("__REGION_JSON__",      json.dumps(region_counts))
    html = html.replace("__CONF_HIST_JSON__",   json.dumps(conf_hist))
    html = html.replace("__ACCURACY__",  str(acc_pct))
    html = html.replace("__PRECISION__", str(prec_pct))
    html = html.replace("__RECALL__",    str(rec_pct))
    html = html.replace("__F1__",        str(f1_pct))

    out_path = os.path.join(OUT_DIR, "cnn_dashboard.html")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"      ✓ Dashboard written → {out_path}")

    # 7. Serve
    url = f"http://{HOST}:{PORT}/cnn_dashboard.html"
    print(f"\n{'='*70}")
    print(f"  🌐  Dashboard live at:  {url}")
    print(f"{'='*70}")
    print("  Press Ctrl+C to stop the server.\n")

    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=OUT_DIR, **kwargs)
        def log_message(self, *args):  # silence access logs
            pass

    try:
        with socketserver.TCPServer((HOST, PORT), Handler) as httpd:
            httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n  Server stopped.")
    except OSError as e:
        print(f"\n  ✗ Port {PORT} busy: {e}")
        print(f"  Open manually: {out_path}")


if __name__ == "__main__":
    main()
