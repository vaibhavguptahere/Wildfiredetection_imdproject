"""
CONFIGURATION FILE - UTTARAKHAND WILDFIRE DETECTION SYSTEM
All regions, features, and parameters defined here
"""

import os

# ============================================================================
# PROJECT PATHS
# ============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

# Create directories
for d in [DATA_DIR, MODELS_DIR, RESULTS_DIR]:
    os.makedirs(d, exist_ok=True)
    
os.makedirs(os.path.join(DATA_DIR, "firms"), exist_ok=True)
os.makedirs(os.path.join(DATA_DIR, "weather"), exist_ok=True)
os.makedirs(os.path.join(DATA_DIR, "gee"), exist_ok=True)

# ============================================================================
# API CREDENTIALS
# ============================================================================
# Google Earth Engine
GEE_PROJECT = os.getenv("GEE_PROJECT", "utopian-splicer-469005-f3")

FIRMS_MAP_KEY = os.getenv(
"FIRMS_MAP_KEY", "c9e7ff13ad9f8cb69f8bc5577c6604dd"
) # fallback to known key

CDS_API_URL = os.getenv("CDS_API_URL", "https://cds.climate.copernicus.eu/api")
CDS_API_KEY = os.getenv("CDS_API_KEY", "2b00e66a-0474-4405-8978-07678c784a2b")

# ============================================================================
# UTTARAKHAND REGIONS - Three sub-regions for comprehensive coverage
# ============================================================================
REGIONS = {
    # Garhwal Division - Highest fire frequency zone
    "garhwal": {
        "bbox": [78.5, 29.5, 80.5, 31.0],  # [min_lon, min_lat, max_lon, max_lat]
        "description": "Garhwal Himalayas - Primary fire zone with chir pine forests",
        "fire_events": [
            {
                "name": "Garhwal_Fire_2023",
                "start": "2023-04-01",
                "end": "2023-06-30",
                "description": "Major forest fires across Pauri and Tehri districts"
            },
            {
                "name": "Garhwal_Fire_2022",
                "start": "2022-04-01",
                "end": "2022-05-31",
                "description": "Severe fire season with 4500+ incidents"
            },
            {
                "name": "Garhwal_Fire_2021",
                "start": "2021-03-15",
                "end": "2021-05-31",
                "description": "Extended fire season starting in March"
            }
        ]
    },
    
    # Kumaon Division - Secondary fire zone
    "kumaon": {
        "bbox": [79.0, 28.5, 80.5, 30.5],
        "description": "Kumaon Himalayas - Oak and pine mixed forests",
        "fire_events": [
            {
                "name": "Kumaon_Fire_2023",
                "start": "2023-04-15",
                "end": "2023-06-15",
                "description": "Nainital and Almora district fires"
            },
            {
                "name": "Kumaon_Fire_2022",
                "start": "2022-04-10",
                "end": "2022-05-31",
                "description": "Peak fire activity in May"
            },
            {
                "name": "Kumaon_Fire_2019",
                "start": "2019-04-01",
                "end": "2019-05-31",
                "description": "Baseline year for model training"
            }
        ]
    },
    
    # Terai Region - Geographic test set (withheld from training)
    "terai": {
        "bbox": [78.5, 28.5, 80.0, 29.5],
        "description": "Terai foothills - Testing generalization to lower elevations",
        "fire_events": [
            {
                "name": "Terai_Fire_2023",
                "start": "2023-03-15",
                "end": "2023-05-31",
                "description": "Agricultural and scrubland fires"
            },
            {
                "name": "Terai_Fire_2022",
                "start": "2022-03-20",
                "end": "2022-05-15",
                "description": "Early season fires in foothills"
            }
        ]
    }
}

# ============================================================================
# SAMPLING PARAMETERS
# ============================================================================
SAMPLES_PER_CLASS = 5000  # Samples per class per event
TRAIN_TEST_SPLIT = 0.8    # 80% train, 20% test
RANDOM_SEED = 42

# Class labels
CLASSES = {
    0: "clear",      # No fire, no smoke, no clouds
    1: "smoke",      # Smoke from fire
    2: "cloud",      # Cloud cover
    3: "active_fire" # Active burning pixels
}

# ============================================================================
# FEATURE GROUPS - 80+ features organized by type
# ============================================================================

# GROUP 1: Raw Spectral Bands (8 features)
SPECTRAL_BANDS = [
    'B1',  # Coastal Aerosol (0.43-0.45 μm)
    'B2',  # Blue (0.45-0.51 μm)
    'B3',  # Green (0.53-0.59 μm)
    'B4',  # Red (0.64-0.67 μm)
    'B5',  # NIR (0.85-0.88 μm)
    'B6',  # SWIR1 (1.57-1.65 μm)
    'B7',  # SWIR2 (2.11-2.29 μm)
    'B10', # Thermal (10.6-11.19 μm)
]

# GROUP 2: Fire Indices (7 features)
FIRE_INDICES = [
    'NDFI',   # Normalized Difference Fire Index (SWIR2-Red)/(SWIR2+Red) - YOUR NOVEL INDEX
    'NBR',    # Normalized Burn Ratio (NIR-SWIR2)/(NIR+SWIR2)
    'dNBR',   # Delta NBR (pre-fire NBR - post-fire NBR)
    'BAI',    # Burned Area Index 1/((0.1-Red)^2 + (0.06-NIR)^2)
    'MIRBI',  # Mid-Infrared Burn Index 10*SWIR2 - 9.8*SWIR1 + 2
    'NBR2',   # Normalized Burn Ratio 2 (SWIR1-SWIR2)/(SWIR1+SWIR2)
    'FRP',    # Fire Radiative Power from MODIS (MW)
]

# GROUP 3: Vegetation Indices (5 features)
VEGETATION_INDICES = [
    'NDVI',   # Normalized Difference Vegetation Index (NIR-Red)/(NIR+Red)
    'EVI',    # Enhanced Vegetation Index
    'SAVI',   # Soil Adjusted Vegetation Index
    'NDWI',   # Normalized Difference Water Index (Green-NIR)/(Green+NIR)
    'GVI',    # Green Vegetation Index
]

# GROUP 4: Smoke Detection (7 features)
SMOKE_INDICES = [
    'SMOKE_IDX',    # (B1-B7)/(B1+B7) - Aerosol detection
    'AOD_047',      # Aerosol Optical Depth at 0.47μm from MODIS
    'AOD_055',      # Aerosol Optical Depth at 0.55μm from MODIS
    'AI',           # Aerosol Index from B1/B2 ratio
    'B1_B2_ratio',  # Blue band ratio for haze
    'B1_B7_ratio',  # Coastal/SWIR ratio for smoke
    'HAZE_IDX',     # (B1-B5)/(B1+B5) - Atmospheric scattering
]

# GROUP 5: Cloud and Snow Masking (6 features)
CLOUD_SNOW_INDICES = [
    'NDSI',         # Normalized Difference Snow Index (Green-SWIR1)/(Green+SWIR1)
    'NDCI',         # Normalized Difference Cloud Index (B3-B7)/(B3+B7)
    'B2_B7_ratio',  # Blue/SWIR ratio for cloud detection
    'CLOUD_PROB',   # Cloud probability from Sentinel-2 QA
    'LST',          # Land Surface Temperature (K)
    'SNOW_FLAG',    # Binary flag: NDSI > 0.4
]

# GROUP 6: Thermal Features (5 features)
THERMAL_FEATURES = [
    'LST_day',         # Daytime Land Surface Temperature from MODIS (K)
    'LST_night',       # Nighttime Land Surface Temperature from MODIS (K)
    'LST_anomaly',     # LST - 30-day baseline mean
    'LST_day_night',   # Daytime - Nighttime difference
    'HOT_PIXEL',       # Binary flag: B10 > 320K
]

# GROUP 7: Spatial Context (8 features)
SPATIAL_FEATURES = [
    'NDFI_mean_3x3',   # Mean NDFI in 3x3 neighborhood
    'NDFI_std_3x3',    # Std dev NDFI in 3x3 neighborhood
    'NDFI_max_3x3',    # Max NDFI in 3x3 neighborhood
    'NDFI_mean_5x5',   # Mean NDFI in 5x5 neighborhood
    'NDFI_std_5x5',    # Std dev NDFI in 5x5 neighborhood
    'B10_mean_3x3',    # Mean thermal in 3x3 neighborhood
    'B10_max_3x3',     # Max thermal in 3x3 neighborhood
    'NDFI_zscore',     # Z-score of NDFI relative to image
]

# GROUP 8: Temporal Change (6 features)
TEMPORAL_FEATURES = [
    'NDFI_change_16d',  # NDFI change vs 16 days prior
    'NDVI_change_16d',  # NDVI change vs 16 days prior
    'B7_change_16d',    # SWIR2 change vs 16 days prior
    'B10_change_16d',   # Thermal change vs 16 days prior
    'NBR_change_16d',   # NBR change vs 16 days prior
    'CHANGE_MAG',       # Euclidean distance in spectral space
]

# GROUP 9: Weather Features (12 features from ERA5)
WEATHER_FEATURES = [
    'wind_speed',      # 10m wind speed (m/s)
    'wind_direction',  # 10m wind direction (degrees)
    'temperature_2m',  # 2m air temperature (K)
    'dewpoint_2m',     # 2m dewpoint temperature (K)
    'rh',              # Relative humidity (%)
    'vpd',             # Vapor Pressure Deficit (kPa)
    'precip_24h',      # 24-hour accumulated precipitation (mm)
    'precip_7d',       # 7-day accumulated precipitation (mm)
    'solar_rad',       # Surface solar radiation (W/m²)
    'pbl_height',      # Planetary boundary layer height (m)
    'wind_u',          # U-component of wind (m/s)
    'wind_v',          # V-component of wind (m/s)
]

# GROUP 10: Terrain Features (9 features)
TERRAIN_FEATURES = [
    'elevation',       # Elevation above sea level (m)
    'slope',           # Terrain slope (degrees)
    'aspect_sin',      # Aspect angle sine component
    'aspect_cos',      # Aspect angle cosine component
    'twi',             # Topographic Wetness Index
    'curvature',       # Surface curvature
    'roughness',       # Terrain roughness index
    'valley_flag',     # Binary: TWI > 10 (valley areas)
    'ridge_flag',      # Binary: elevation - mean_5x5 > 50m
]

# GROUP 11: Fuel Type Classification (7 features)
FUEL_FEATURES = [
    'fuel_pine',       # Chir pine forest probability
    'fuel_oak',        # Oak forest probability
    'fuel_grass',      # Grassland probability
    'fuel_shrub',      # Shrubland probability
    'fuel_urban',      # Urban/barren probability
    'fuel_moisture',   # Fuel moisture index (NDWI-based)
    'fuel_load',       # Fuel load proxy (NDVI * B5)
]

# GROUP 12: Fire History (5 features)
FIRE_HISTORY_FEATURES = [
    'prev_year_burn',  # Binary: burned in previous year
    'fire_count_3yr',  # Number of fires in past 3 years
    'days_since_fire', # Days since last fire detection
    'season_flag',     # Binary: in fire season (April-June)
    'cumulative_frp',  # Cumulative FRP in past 30 days
]

# Combined feature list (80+ features)
ALL_FEATURES = (
    SPECTRAL_BANDS +
    FIRE_INDICES +
    VEGETATION_INDICES +
    SMOKE_INDICES +
    CLOUD_SNOW_INDICES +
    THERMAL_FEATURES +
    SPATIAL_FEATURES +
    TEMPORAL_FEATURES +
    WEATHER_FEATURES +
    TERRAIN_FEATURES +
    FUEL_FEATURES +
    FIRE_HISTORY_FEATURES
)

# ============================================================================
# SPECTRAL THRESHOLDS - Calibrated for Uttarakhand chir pine forests
# ============================================================================
THRESHOLDS = {
    # Fire detection
    'fire': {
        'B10_min': 320,      # Thermal minimum for fire (K)
        'B7_min': 0.15,      # SWIR2 minimum for fire
        'NDFI_min': 0.3,     # Your NDFI threshold
        'FRP_min': 5.0,      # Fire Radiative Power minimum (MW)
    },
    
    # Smoke detection
    'smoke': {
        'AOD_min': 0.3,      # Aerosol Optical Depth threshold
        'SMOKE_IDX_min': 0.1,
        'B1_high': 0.15,     # High blue reflectance
    },
    
    # Cloud masking
    'cloud': {
        'NDCI_min': 0.5,     # Cloud index threshold
        'B2_high': 0.3,      # High blue reflectance
        'CLOUD_PROB_min': 0.65,
    },
    
    # Snow masking (critical for Himalayas)
    'snow': {
        'NDSI_min': 0.4,     # Snow index threshold
        'B2_high': 0.3,      # High blue reflectance
        'LST_max': 280,      # Cold surface temperature (K)
    },
    
    # Clear pixels
    'clear': {
        'NDVI_range': (0.2, 0.8),  # Healthy vegetation
        'B10_range': (285, 310),    # Normal temperature range
    }
}

# ============================================================================
# MODEL PARAMETERS
# ============================================================================
MODEL_PARAMS = {
    'random_forest': {
        'n_estimators': 200,
        'max_depth': 25,
        'min_samples_split': 10,
        'min_samples_leaf': 4,
        'max_features': 'sqrt',
        'n_jobs': -1,
        'random_state': RANDOM_SEED
    },
    
    'xgboost': {
        'n_estimators': 200,
        'max_depth': 10,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 3,
        'random_state': RANDOM_SEED,
        'n_jobs': -1
    },
    
    'svm': {
        'C': 10.0,
        'gamma': 'scale',
        'kernel': 'rbf',
        'probability': True,
        'random_state': RANDOM_SEED
    },
    
    'cnn': {
        'epochs': 50,
        'batch_size': 64,
        'learning_rate': 0.001,
        'dropout': 0.3,
        'conv_filters': [32, 64, 128],
        'dense_units': [256, 128]
    }
}

# ============================================================================
# GOOGLE EARTH ENGINE COLLECTIONS
# ============================================================================
GEE_COLLECTIONS = {
    'landsat8': 'LANDSAT/LC08/C02/T1_L2',
    'sentinel2': 'COPERNICUS/S2_SR_HARMONIZED',
    'modis_lst': 'MODIS/006/MOD11A1',
    'modis_aod': 'MODIS/006/MCD19A2_GRANULES',
    'modis_fire': 'MODIS/006/MOD14A1',
    'era5_land': 'ECMWF/ERA5_LAND/HOURLY',
    'srtm': 'USGS/SRTMGL1_003',
    'esa_landcover': 'ESA/WorldCover/v100'
}

# ============================================================================
# DASHBOARD CONFIGURATION
# ============================================================================
DASHBOARD_CONFIG = {
    'host': 'localhost',
    'port': 8000,
    'map_center': [30.0, 79.5],  # Center on Uttarakhand
    'map_zoom': 8,
    'tile_layer': 'https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',
    'marker_colors': {
        'clear': 'green',
        'smoke': 'orange',
        'cloud': 'gray',
        'active_fire': 'red'
    }
}

# ============================================================================
# VALIDATION METRICS
# ============================================================================
TARGET_METRICS = {
    'overall_accuracy': 0.90,      # ≥90% target
    'smoke_cloud_separation': 0.85, # ≥85% target
    'fire_recall': 0.95,           # ≥95% fire detection rate
    'false_positive_rate': 0.05    # ≤5% false alarms
}

print("✓ Configuration loaded successfully")
print(f"  Regions: {len(REGIONS)} (Garhwal, Kumaon, Terai)")
print(f"  Total fire events: {sum(len(r['fire_events']) for r in REGIONS.values())}")
print(f"  Total features: {len(ALL_FEATURES)}")
print(f"  Target accuracy: {TARGET_METRICS['overall_accuracy']*100}%")