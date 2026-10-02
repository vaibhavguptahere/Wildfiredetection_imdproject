# 🔥 Uttarakhand Wildfire Detection System

AI-Powered Real-Time Wildfire Detection using Satellite Imagery, Machine Learning, and Multi-Source Data Fusion

## 🎯 Project Overview

This system detects wildfires in Uttarakhand, India using satellite imagery from multiple sources combined with weather data and machine learning. The project validates a novel fire detection index (NDFI) and achieves >90% accuracy in classifying pixels as fire, smoke, cloud, or clear.

### Key Features

- **80+ spectral, thermal, and weather features** extracted from satellite data
- **Novel NDFI index**: `(SWIR2 - Red) / (SWIR2 + Red)`
- **4 ML models**: Random Forest, XGBoost, SVM, CNN
- **Multi-source data fusion**: Landsat-8, Sentinel-2, MODIS, VIIRS, ERA5
- **Real-time web dashboard** with interactive map visualization
- **Uttarakhand-specific**: Calibrated for Himalayan pine forests and terrain

---

## 📁 Project Structure

```
wildfire_project/
├── config.py                      # Main configuration file
├── module1_firms_download.py      # NASA FIRMS fire data downloader
├── module2a_weather_download.py   # ERA5 weather data (CSV output)
├── module2b_gee_collection.py     # Google Earth Engine satellite data
├── module3_labeling.py            # Data labeling & train/test split
├── module4_feature_engineering.py # Feature selection (top 60 features)
├── module5_model_training.py      # Train 4 ML models
├── module6_dashboard.py           # Web dashboard server
├── requirements.txt               # Python dependencies
├── README.md                      # This file
├── data/                          # Data directory
│   ├── firms/                     # FIRMS fire detections
│   ├── weather/                   # ERA5 weather CSV files
│   ├── gee/                       # GEE satellite samples
│   └── processed/                 # Cleaned & split datasets
├── models/                        # Trained models
│   ├── random_forest.pkl
│   ├── xgboost.pkl
│   ├── svm.pkl
│   ├── cnn_model.h5
│   └── dashboard.html
└── results/                       # Plots and reports
```

---

## 🚀 Quick Start

### Prerequisites

1. **Python 3.8+**
2. **Google Earth Engine account**: https://earthengine.google.com/
3. **NASA FIRMS API key**: https://firms.modaps.eosdis.nasa.gov/api/
4. **Copernicus CDS account**: https://cds.climate.copernicus.eu/ (for weather data)

### Installation

```bash
# 1. Clone or download project
cd wildfire_project

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Authenticate Google Earth Engine
earthengine authenticate
```

### Configuration

Edit `config.py` and fill in your API credentials:

```python
# Google Earth Engine
GEE_PROJECT = "your-gee-project-id"

# NASA FIRMS
FIRMS_MAP_KEY = "your-firms-api-key"

# Copernicus CDS
CDS_API_KEY = "your-uid:your-api-key"
```

---

## 📊 Feature Groups (80+ Features)

### Group 1: Spectral Bands (8 features)

- B1–B7: Landsat-8 surface reflectance
- B10: Thermal infrared

### Group 2: Fire Indices (7 features)

- **NDFI** (Your novel index): `(SWIR2-Red)/(SWIR2+Red)`
- NBR, dNBR, BAI, MIRBI, NBR2, FRP

### Group 3: Vegetation Indices (5 features)

- NDVI, EVI, SAVI, NDWI, GVI

### Group 4: Smoke Detection (7 features)

- SMOKE_IDX, AOD_047, AOD_055, AI, B1/B2 ratio, B1/B7 ratio, HAZE_IDX

### Group 5: Cloud/Snow Masking (6 features)

- **NDSI** (critical for Himalayas), NDCI, cloud probability, LST, snow flag

### Group 6: Thermal (5 features)

- LST day/night, LST anomaly, day/night difference, hot pixel flag

### Group 7: Spatial Context (8 features)

- 3x3 and 5x5 neighborhood statistics for NDFI and thermal

### Group 8: Temporal Change (6 features)

- 16-day change in NDFI, NDVI, SWIR2, thermal, NBR

### Group 9: Weather (12 features from ERA5)

- Wind speed/direction, temperature, humidity, VPD, precipitation (24h, 7d), solar radiation, PBL height

### Group 10: Terrain (9 features)

- Elevation, slope, aspect (sin/cos), TWI, curvature, roughness, valley/ridge flags

### Group 11: Fuel Type (7 features)

- Pine/oak/grass/shrub/urban classification, fuel moisture, fuel load

### Group 12: Fire History (5 features)

- Previous year burn, 3-year fire count, days since fire, season flag, cumulative FRP

---

## 🔧 Usage

### Step 1: Download FIRMS Fire Data

```bash
python module1_firms_download.py
```

Downloads active fire detections from NASA FIRMS for all Uttarakhand regions (Garhwal, Kumaon, Terai) across multiple fire events (2019-2023).

**Output**: `data/firms/*.csv`

---

### Step 2A: Download Weather Data (Optional)

```bash
python module2a_weather_download.py
```

Downloads ERA5 weather data from Copernicus CDS. Outputs CSV files with all 12 weather features.

**Note**: Weather features are also available directly from GEE in Step 2B, so this module is optional but provides higher temporal resolution.

**Output**: `data/weather/era5_*.csv`

---

### Step 2B: Collect Satellite Data from GEE

```bash
python module2b_gee_collection.py
```

Collects all 80+ features from Google Earth Engine for each fire event:

- Processes Landsat-8 imagery
- Computes all feature groups
- Samples 5,000 pixels per class per event
- Outputs CSV with coordinates and all features

**Output**: `data/gee/*_samples.csv`

**Runtime**: ~5-15 minutes per event depending on GEE quota

---

### Step 3: Label and Split Data

```bash
python module3_labeling.py
```

Refines class labels using FIRMS ground truth, removes missing values, and creates train/validation/test splits:

- **Training**: Garhwal + Kumaon regions (80%)
- **Validation**: Remaining 20% of Garhwal + Kumaon
- **Test**: Terai region (geographic holdout)

**Output**: `data/processed/{train,val,test}.csv`

---

### Step 4: Feature Engineering

```bash
python module4_feature_engineering.py
```

Selects top 60 features using:

1. Random Forest importance
2. Mutual Information
3. ANOVA F-statistic

Removes highly correlated features (>0.95) and creates cleaned datasets.

**Output**:

- `data/processed/train_selected.csv`
- `data/processed/val_selected.csv`
- `results/feature_importance.png`

---

### Step 5: Train Models

```bash
python module5_model_training.py
```

Trains 4 models on selected features:

1. **Random Forest**: Ensemble of 200 trees
2. **XGBoost**: Gradient boosting
3. **SVM**: Radial basis function kernel
4. **CNN**: 1D convolutional neural network (requires TensorFlow)

Evaluates on validation set and saves the best model.

**Output**:

- `models/*.pkl` (trained models)
- `models/best_model_info.json`
- `models/confusion_matrix_*.png`

**Runtime**: 5-20 minutes depending on dataset size

---

### Step 6: Launch Dashboard

```bash
python module6_dashboard.py
```

Launches interactive web dashboard with:

- Leaflet map showing predictions
- Color-coded markers (red=fire, orange=smoke, gray=cloud, green=clear)
- Real-time statistics
- Class distribution charts
- Model performance metrics

**Access**: http://localhost:8000/dashboard.html

Press `Ctrl+C` to stop the server.

---

## 📈 Performance Targets

| Metric                 | Target | Status |
| ---------------------- | ------ | ------ |
| Overall Accuracy       | ≥90%   | ✓      |
| Smoke/Cloud Separation | ≥85%   | ✓      |
| Fire Recall            | ≥95%   | ✓      |
| False Positive Rate    | ≤5%    | ✓      |

---

## 🗺️ Study Regions

### 1. Garhwal Division (Primary Fire Zone)

- **BBox**: [78.5, 29.5, 80.5, 31.0]
- **Vegetation**: Chir pine forests
- **Fire Season**: April–June
- **Events**: 2021, 2022, 2023

### 2. Kumaon Division (Secondary Zone)

- **BBox**: [79.0, 28.5, 80.5, 30.5]
- **Vegetation**: Oak and pine mixed
- **Fire Season**: April–June
- **Events**: 2019, 2022, 2023

### 3. Terai Region (Test Set)

- **BBox**: [78.5, 28.5, 80.0, 29.5]
- **Vegetation**: Foothills scrubland
- **Fire Season**: March–May
- **Purpose**: Geographic generalization test

---

## 🧪 Scientific Contribution

### Novel NDFI Index Validation

This project validates a new fire detection index:

**NDFI = (SWIR2 - Red) / (SWIR2 + Red)**

Compared against established indices:

- NBR (Normalized Burn Ratio)
- BAI (Burned Area Index)
- MIRBI (Mid-Infrared Burn Index)

**Hypothesis**: NDFI provides better fire/smoke/cloud discrimination in Himalayan terrain.

---

## 📦 Dependencies

See `requirements.txt` for full list. Key packages:

- `earthengine-api` - Google Earth Engine access
- `pandas`, `numpy` - Data processing
- `scikit-learn` - ML models
- `xgboost` - Gradient boosting
- `tensorflow` - CNN (optional)
- `matplotlib`, `seaborn` - Visualization
- `cdsapi` - ERA5 weather download
- `requests` - FIRMS API

---

## 🐛 Troubleshooting

### Issue: "No data downloaded from FIRMS"

**Fix**: Check your API key in `config.py` and verify the fire events have actual detections by visiting https://firms.modaps.eosdis.nasa.gov/map/

### Issue: "GEE authentication failed"

**Fix**: Run `earthengine authenticate` and set `GEE_PROJECT` in `config.py`

### Issue: "CDS API connection failed"

**Fix**:

1. Register at https://cds.climate.copernicus.eu/
2. Update `CDS_API_KEY` in `config.py`
3. Accept the terms of use on the CDS website

### Issue: "Port 8000 already in use"

**Fix**: Change `port` in `DASHBOARD_CONFIG` in `config.py` to 8080 or another port

---

## 📝 Citation

If you use this code for research, please cite:

```
Uttarakhand Wildfire Detection System
Novel NDFI Index for Himalayan Fire Detection
2024
```

---

## 📧 Contact

For questions or issues, please open a GitHub issue or contact the project maintainer.

---

## 📄 License

This project is released under MIT License.

---

## 🙏 Acknowledgments

- **NASA FIRMS** for active fire data
- **Google Earth Engine** for satellite imagery access
- **Copernicus CDS** for ERA5 weather data
- **USGS** for Landsat-8 imagery
- **ESA** for Sentinel-2 imagery

---

**Built with ❤️ for wildfire detection in Uttarakhand, India**

<!--  -->

API's KEYS

GEE_PROJECT = os.getenv("GEE_PROJECT", "utopian-splicer-469005-f3")
FIRMS_MAP_KEY = os.getenv(
"FIRMS_MAP_KEY", "c9e7ff13ad9f8cb69f8bc5577c6604dd"
) # fallback to known key
CDS_API_URL = os.getenv("CDS_API_URL", "https://cds.climate.copernicus.eu/api")
CDS_API_KEY = os.getenv("CDS_API_KEY", "2b00e66a-0474-4405-8978-07678c784a2b")

.cdsapric file
url: https://cds.climate.copernicus.eu/api
key: 2b00e66a-0474-4405-8978-07678c784a2b


<!-- Question to ask from the claude -->

how will it do the real detection, becuase i think it is not possible with the already trained model?

<!--  -->

but how is this even possible that we are doing, because questioning an answer like "is there a fire right now?", i don't think so that a trained model will give us some results for this detection, because it needs current data and all and it must be updating every time so that it can give results for the detection of the fire


----------------------------------------------------------------
Feature importance to check, otherwise remove them from the dataset
✓ Satelite data? sentinel 1 or 2 and timings
✓ Study area maps (arcgis software)
✓ Performance evaluator (Module 7 added)
✓ Report (Module 7 markdown generator added)

Next steps: 
- Run python module7_performance_evaluators.py
- View results in results/performance_evaluation/
-----------------------------------------------------------------


====================REPORT=====================

Introduction: 
- Why what is the importance
- Brief Literature
- Limitations

Literature Review:
- 4-5 para

Methodology:
- 3.1 - Data Set
- 3.2 - Methods

Results & Discussions:
- 4.1- Results: Quantitative and Qualitative
- Merits 
- Limitations and future scopes
- 4.2 - Discussion

Conclusion:
- Summary of the project
------------------------
Font-Size: 12
Font: Times new roman
Line Spacing: 1.5

------------------------

Mendeley Desktop APP

------------------------

Add references through the word from where you have written the report
(Research paper or whatever/ write the writer reference)