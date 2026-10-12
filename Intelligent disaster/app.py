import os
import pickle
import warnings
from io import BytesIO

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import shap
import streamlit as st
import folium
from streamlit_folium import st_folium

from recommendation_engine import generate_tn_recommendations, SEVERITY

warnings.filterwarnings("ignore")

TN_CENTER = {"lat": 11.1271, "lon": 78.6569, "zoom": 7}
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_CSV = os.path.join(BASE_DIR, "tamilnadu_disaster_data.csv")
DATA_CSV_ALT = os.path.join(BASE_DIR, "disaster_data.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "best_model.pkl")
BENCH_PATH = os.path.join(BASE_DIR, "models", "benchmarks.csv")
IMP_PATH = os.path.join(BASE_DIR, "models", "feature_importance.csv")

DISTRICTS = [
    "Chennai", "Cuddalore", "Thiruvallur", "Nilgiris", "Nagapattinam",
    "Kanchipuram", "Madurai", "Coimbatore", "Kanyakumari", "Tirunelveli",
    "Thanjavur", "Salem", "Tiruchirappalli", "Vellore", "Erode",
]

DISTRICT_META = {
    "Chennai":       {"lat": 13.0827, "lon": 80.2707, "elev": 6.0,   "slope": 2.0},
    "Cuddalore":     {"lat": 11.7474, "lon": 79.7694, "elev": 8.0,   "slope": 3.0},
    "Thiruvallur":   {"lat": 13.1193, "lon": 79.9817, "elev": 12.0,  "slope": 4.0},
    "Nilgiris":      {"lat": 11.4906, "lon": 76.7343, "elev": 2000.0,"slope": 38.0},
    "Nagapattinam":  {"lat": 10.7641, "lon": 79.8439, "elev": 5.0,   "slope": 2.0},
    "Kanchipuram":   {"lat": 12.8343, "lon": 79.7037, "elev": 72.0,  "slope": 6.0},
    "Madurai":       {"lat": 9.9252,  "lon": 78.1198, "elev": 157.0, "slope": 8.0},
    "Coimbatore":    {"lat": 11.0168, "lon": 76.9558, "elev": 411.0, "slope": 15.0},
    "Kanyakumari":   {"lat": 8.0883,  "lon": 77.5385, "elev": 32.0,  "slope": 9.0},
    "Tirunelveli":   {"lat": 8.7139,  "lon": 77.7567, "elev": 111.0, "slope": 10.0},
    "Thanjavur":     {"lat": 10.7867, "lon": 79.1378, "elev": 47.0,  "slope": 3.0},
    "Salem":         {"lat": 11.6643, "lon": 78.1460, "elev": 278.0, "slope": 12.0},
    "Tiruchirappalli":{"lat": 10.7905,"lon": 78.7047, "elev": 78.0,  "slope": 5.0},
    "Vellore":       {"lat": 12.9165, "lon": 79.1325, "elev": 226.0, "slope": 11.0},
    "Erode":         {"lat": 11.3307, "lon": 77.7172, "elev": 193.0, "slope": 10.0},
}

st.set_page_config(
    page_title="TN Satellite Disaster AI | Intelligence Platform",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
    [data-testid="stSidebar"] {background: linear-gradient(180deg,#0b1e3a,#163969 60%,#1f4f8f); color:#fff;}
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3, [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] p, [data-testid="stSidebar"] span {color:#e9f1ff !important;}
    [data-testid="stMetricValue"] {font-size: 1.6rem !important; font-weight: 700;}
    .risk-pill {display:inline-block; padding:4px 12px; border-radius:999px; font-weight:700;
                font-size:0.85rem; color:#fff; letter-spacing:0.3px;}
    .pill-safe {background:#2ecc71;}
    .pill-warn {background:#f39c12;}
    .pill-crit {background:#e74c3c;}
    .rec-card {background: #f8fbff; border-left:5px solid #2563eb; padding:14px 18px;
               border-radius:10px; margin:8px 0; box-shadow: 0 2px 8px rgba(30,60,120,0.08); color:#000000 !important;}
    .rec-card * {color:#000000 !important;}
    .sev-critical .rec-card {border-left-color:#dc2626; background:#fff5f5;}
    .sev-high .rec-card {border-left-color:#ea580c; background:#fff7ed;}
    .sev-medium .rec-card {border-left-color:#ca8a04; background:#fffbeb;}
    .sev-low .rec-card {border-left-color:#0891b2; background:#ecfeff;}
    .sev-safe .rec-card {border-left-color:#16a34a; background:#f0fdf4;}
    h1 span.hl {background: linear-gradient(90deg,#2563eb,#7c3aed);
                -webkit-background-clip:text; color:transparent;}
    div[data-testid="stDataFrame"] td, div[data-testid="stDataFrame"] th {
        color: #000000 !important;
        font-weight: 600;
        background-color: #ffffff !important;
    }
    div[data-testid="stDataFrame"] tbody td {
        background-color: #ffffff !important;
    }
    .foot {margin-top:2rem; opacity:0.55; font-size:0.75rem; text-align:center;}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def _auto_detect_cols(df: pd.DataFrame):
    cols_lower = {str(c).lower().strip(): c for c in df.columns}

    def find(opts):
        for opt in opts:
            if opt in cols_lower:
                return cols_lower[opt]
        return None

    district_col = find([
        "district", "districts", "place", "location", "region", "area",
        "taluk", "block", "city", "town", "state",
    ])
    lat_col = find(["latitude", "lat", "lat_", "y"])
    lon_col = find(["longitude", "longitude_", "lon", "lng", "long", "x"])

    elev_col = find([
        "elevation_m", "elevation", "elev", "elev_m", "height", "altitude",
        "altitude_m", "dem", "srtm",
    ])
    slope_col = find([
        "slope_deg", "slope", "slope_", "slope_degrees", "terrain_slope",
    ])
    temp_col = find([
        "temperature_c", "temperature", "temp", "temp_c", "t2m", "air_temp",
    ])
    hum_col = find([
        "humidity_pct", "humidity", "hum", "rh", "relative_humidity",
    ])
    wind_col = find([
        "wind_speed_kmh", "wind_speed", "wind", "wspd", "ws", "wind_kmh",
        "wind_kmph",
    ])
    rain_col = find([
        "rainfall_7d_mm", "rainfall", "rain", "precip", "precipitation",
        "rain_mm", "pr", "rainfall_7d", "cum_rain",
    ])
    ndvi_col = find(["ndvi", "ndvi_", "ndvi_mean", "modis_ndvi"])
    ndwi_col = find(["ndwi", "ndwi_", "ndwi_mean", "mndwi", "modis_ndwi"])

    flood_col = find(["flood_risk", "flood", "flood_flag"])
    cyclone_col = find(["cyclone_risk", "cyclone", "cyclone_flag"])
    wildfire_col = find(["wildfire_risk", "wildfire", "landslide_risk",
                         "landslide", "wildfire_landslide"])
    dtype_col = find(["disaster_type", "hazard_type", "type", "hazard"])

    remap = {}
    if district_col and "district" not in df.columns:
        remap[district_col] = "district"
    if lat_col and "latitude" not in df.columns:
        remap[lat_col] = "latitude"
    if lon_col and "longitude" not in df.columns:
        remap[lon_col] = "longitude"
    if elev_col and "elevation_m" not in df.columns:
        remap[elev_col] = "elevation_m"
    if slope_col and "slope_deg" not in df.columns:
        remap[slope_col] = "slope_deg"
    if temp_col and "temperature_c" not in df.columns:
        remap[temp_col] = "temperature_c"
    if hum_col and "humidity_pct" not in df.columns:
        remap[hum_col] = "humidity_pct"
    if wind_col and "wind_speed_kmh" not in df.columns:
        remap[wind_col] = "wind_speed_kmh"
    if rain_col and "rainfall_7d_mm" not in df.columns:
        remap[rain_col] = "rainfall_7d_mm"
    if ndvi_col and "ndvi" not in df.columns:
        remap[ndvi_col] = "ndvi"
    if ndwi_col and "ndwi" not in df.columns:
        remap[ndwi_col] = "ndwi"
    if flood_col and "flood_risk" not in df.columns:
        remap[flood_col] = "flood_risk"
    if cyclone_col and "cyclone_risk" not in df.columns:
        remap[cyclone_col] = "cyclone_risk"
    if wildfire_col and "wildfire_risk" not in df.columns:
        remap[wildfire_col] = "wildfire_risk"
    if dtype_col and "disaster_type" not in df.columns:
        remap[dtype_col] = "disaster_type"
    return remap


def normalize_dataset(df_raw: pd.DataFrame):
    remap = _auto_detect_cols(df_raw)
    df = df_raw.rename(columns=remap).copy()

    if "district" not in df.columns:
        obj_cols = [c for c in df.columns
                    if df[c].dtype == object or str(df[c].dtype).startswith("category")]
        df["district"] = (
            df[obj_cols[0]].astype(str)
            if obj_cols
            else "Unknown_Area"
        )

    np.random.seed(42)
    if "latitude" not in df.columns:
        df["latitude"] = np.random.uniform(8.07, 13.50, size=len(df))
    if "longitude" not in df.columns:
        df["longitude"] = np.random.uniform(76.23, 80.34, size=len(df))

    rng = np.random.default_rng(42)
    for col, lo, hi, default_mean in [
        ("elevation_m", 0, 3000, 150),
        ("slope_deg", 0, 45, 10),
        ("temperature_c", 15, 45, 32),
        ("humidity_pct", 10, 100, 65),
        ("wind_speed_kmh", 0, 150, 30),
        ("rainfall_7d_mm", 0, 800, 80),
        ("ndvi", -0.3, 0.9, 0.3),
        ("ndwi", -0.9, 0.9, 0.0),
    ]:
        if col not in df.columns:
            df[col] = rng.uniform(lo, hi, size=len(df))
        df[col] = pd.to_numeric(df[col], errors="coerce")
        nan_mask = df[col].isna().to_numpy()
        if nan_mask.any():
            fill_vals = rng.uniform(lo, hi, size=int(nan_mask.sum()))
            df.loc[nan_mask, col] = fill_vals
        df[col] = df[col].astype(float).clip(lo, hi)


    if "flood_risk" not in df.columns:
        fp = (
            (df["rainfall_7d_mm"] > 250).astype(float) * 0.45
            + (df["rainfall_7d_mm"] > 120).astype(float) * 0.2
            + (df["ndwi"] > 0.3).astype(float) * 0.25
            + (df["elevation_m"] < 30).astype(float) * 0.2
            + (df["humidity_pct"] > 80).astype(float) * 0.15
        )
        df["flood_risk"] = ((fp + np.random.normal(0, 0.1, len(df))) > 0.5).astype(int)
    else:
        df["flood_risk"] = pd.to_numeric(df["flood_risk"], errors="coerce").fillna(0).astype(int)

    if "cyclone_risk" not in df.columns:
        cp = (
            (df["wind_speed_kmh"] > 70).astype(float) * 0.4
            + (df["wind_speed_kmh"] > 40).astype(float) * 0.2
            + (df["humidity_pct"] > 75).astype(float) * 0.2
            + (df["elevation_m"] < 50).astype(float) * 0.25
        )
        df["cyclone_risk"] = ((cp + np.random.normal(0, 0.1, len(df))) > 0.5).astype(int)
    else:
        df["cyclone_risk"] = pd.to_numeric(df["cyclone_risk"], errors="coerce").fillna(0).astype(int)

    if "wildfire_risk" not in df.columns:
        wp = (
            (df["temperature_c"] > 35).astype(float) * 0.4
            + (df["temperature_c"] > 30).astype(float) * 0.2
            + (df["humidity_pct"] < 50).astype(float) * 0.2
            + (df["ndvi"] < 0.1).astype(float) * 0.25
            + (df["slope_deg"] > 20).astype(float) * 0.15
            + ((df["elevation_m"] > 1000) & (df["slope_deg"] > 20)).astype(float) * 0.2
        )
        df["wildfire_risk"] = ((wp + np.random.normal(0, 0.1, len(df))) > 0.5).astype(int)
    else:
        df["wildfire_risk"] = pd.to_numeric(df["wildfire_risk"], errors="coerce").fillna(0).astype(int)

    if "disaster_type" not in df.columns:
        risk_types = []
        for _, r in df.iterrows():
            d = {
                "Flood": r["flood_risk"] * (0.5 + r["rainfall_7d_mm"] / 900),
                "Cyclone": r["cyclone_risk"] * (0.5 + r["wind_speed_kmh"] / 240),
                "Wildfire/Landslide": r["wildfire_risk"] * (0.5 + r["temperature_c"] / 90),
            }
            best = max(d, key=d.get)
            risk_types.append(best if d[best] > 0.35 else "Safe")
        df["disaster_type"] = risk_types
    return df, remap


@st.cache_data(show_spinner=False, ttl=3600)
def load_dataset_from_bytes(file_bytes):
    df_raw = pd.read_csv(BytesIO(file_bytes))
    df_norm, remap = normalize_dataset(df_raw)
    return df_norm, remap, df_raw


@st.cache_data(show_spinner=False, ttl=3600)
def load_dataset(_path=None):
    paths_to_try = []
    if _path is not None:
        paths_to_try.append(_path)
    paths_to_try += [DATA_CSV, DATA_CSV_ALT]
    for p in paths_to_try:
        if p and os.path.exists(p):
            df_raw = pd.read_csv(p)
            df_norm, remap = normalize_dataset(df_raw)
            return df_norm, remap, df_raw, p
    return None, {}, None, None


@st.cache_data(show_spinner=False)
def load_benchmarks():
    if os.path.exists(BENCH_PATH):
        return pd.read_csv(BENCH_PATH)
    return None


@st.cache_data(show_spinner=False)
def load_feature_importance():
    if os.path.exists(IMP_PATH):
        return pd.read_csv(IMP_PATH)
    return None


@st.cache_resource(show_spinner=False)
def load_model_artifact():
    if not os.path.exists(MODEL_PATH):
        return None
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


def empty_dashboard_frame():
    return pd.DataFrame({
        "district": pd.Series(dtype="object"),
        "latitude": pd.Series(dtype=float),
        "longitude": pd.Series(dtype=float),
        "elevation_m": pd.Series(dtype=float),
        "slope_deg": pd.Series(dtype=float),
        "temperature_c": pd.Series(dtype=float),
        "humidity_pct": pd.Series(dtype=float),
        "wind_speed_kmh": pd.Series(dtype=float),
        "rainfall_7d_mm": pd.Series(dtype=float),
        "ndvi": pd.Series(dtype=float),
        "ndwi": pd.Series(dtype=float),
        "flood_risk": pd.Series(dtype=float),
        "cyclone_risk": pd.Series(dtype=float),
        "wildfire_risk": pd.Series(dtype=float),
        "disaster_type": pd.Series(dtype="object"),
    })


def _normalize_shap_values(values):
    arr = np.asarray(values)
    if arr.size == 0:
        return arr.reshape((0, 0))
    if arr.ndim == 3 and arr.shape[-1] == 2:
        return arr[:, :, 1]
    if arr.ndim == 3 and arr.shape[0] == 1:
        return arr[0]
    return arr


def _safe_scalar(value):
    if isinstance(value, (list, tuple, np.ndarray)):
        arr = np.asarray(value)
        if arr.size == 0:
            return 0.0
        if arr.ndim > 1:
            arr = arr.reshape(-1)
        if arr.size == 2:
            return float(arr[1])
        return float(arr[0])
    return float(value)


def district_risk_aggregates(df: pd.DataFrame):
    if df is None or df.empty:
        return pd.DataFrame(columns=[
            "district", "n_rows", "latitude", "longitude", "elevation_m",
            "temperature_c", "humidity_pct", "wind_speed_kmh", "rainfall_7d_mm",
            "ndvi", "ndwi", "flood_risk_rate", "cyclone_risk_rate",
            "wildfire_risk_rate", "disaster_type", "risk_score",
        ])

    grouped = df.groupby("district").agg(
        n_rows=("district", "size"),
        latitude=("latitude", "mean"),
        longitude=("longitude", "mean"),
        elevation_m=("elevation_m", "mean"),
        temperature_c=("temperature_c", "mean"),
        humidity_pct=("humidity_pct", "mean"),
        wind_speed_kmh=("wind_speed_kmh", "mean"),
        rainfall_7d_mm=("rainfall_7d_mm", "mean"),
        ndvi=("ndvi", "mean"),
        ndwi=("ndwi", "mean"),
        flood_risk_rate=("flood_risk", "mean"),
        cyclone_risk_rate=("cyclone_risk", "mean"),
        wildfire_risk_rate=("wildfire_risk", "mean"),
    ).reset_index()

    def pick_disaster(row):
        scores = {
            "Flood": row["flood_risk_rate"],
            "Cyclone": row["cyclone_risk_rate"],
            "Wildfire/Landslide": row["wildfire_risk_rate"],
        }
        best = max(scores, key=scores.get)
        best_val = scores[best]
        if best_val < 0.25:
            return "Safe", 0.10
        return best, round(best_val, 3)

    grouped[["disaster_type", "risk_score"]] = grouped.apply(
        lambda r: pd.Series(pick_disaster(r)), axis=1
    )
    return grouped


def _severity_from_score(s: float):
    if s >= 0.85:
        return "critical"
    if s >= 0.70:
        return "high"
    if s >= 0.40:
        return "medium"
    if s >= 0.15:
        return "low"
    return "safe"


def _color_from_severity(sev: str):
    return {"safe": "#2ecc71", "low": "#27ae60", "medium": "#f39c12",
            "high": "#e67e22", "critical": "#e74c3c"}.get(sev, "#95a5a6")


def _icon_from_severity(sev: str):
    return {"safe": "shield", "low": "leaf", "medium": "exclamation-triangle",
            "high": "exclamation-circle", "critical": "fire"}.get(sev, "info-sign")


FOLIUM_TILES = [
    ("OpenStreetMap", "OpenStreetMap"),
    ("Esri World Imagery", "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"),
]


def build_folium_map(agg_df: pd.DataFrame, selected_severities=None, tile_choice=0):
    tile_name, tile_url = FOLIUM_TILES[tile_choice if tile_choice < len(FOLIUM_TILES) else 0]
    m = folium.Map(
        location=[TN_CENTER["lat"], TN_CENTER["lon"]],
        zoom_start=TN_CENTER["zoom"],
        tiles=None,
    )
    attr = (
        "&copy; <a href='https://www.openstreetmap.org/copyright'>OpenStreetMap</a> contributors"
        if tile_name.startswith("OpenStreetMap")
        else (
            "&copy; <a href='https://carto.com/attributions'>CARTO</a>"
            if tile_name.startswith("CartoDB")
            else "Tiles &copy; Esri &mdash; Source: Esri, i-cubed, USDA, USGS, AEX, GeoEye, Getmapping, Aerogrid, IGN, IGP, UPR-EGP, and the GIS User Community"
        )
    )
    folium.TileLayer(tiles=tile_url, attr=attr, name=tile_name, overlay=False, control=True).add_to(m)
    folium.LayerControl(collapsed=True).add_to(m)

    bounds = []
    if agg_df is not None and len(agg_df) > 0:
        for _, row in agg_df.iterrows():
            sev = _severity_from_score(float(row.get("risk_score", 0.0)))
            if selected_severities and sev not in selected_severities:
                continue
            color = _color_from_severity(sev)
            icon = _icon_from_severity(sev)
            popup_html = f"""
            <div style="min-width:230px;font-family:Segoe UI,Arial,sans-serif">
              <b style="font-size:1.05rem;">{row.get('district','Unknown')}</b><br>
              <span style="color:{color};font-weight:700;">{sev.upper()} &bull; {row.get('disaster_type','Safe')}</span><br><br>
              <b>Risk Score:</b> {float(row.get('risk_score',0.0))*100:.1f}%<br>
              <b>Flood Risk Rate:</b> {float(row.get('flood_risk_rate',0.0))*100:.1f}%<br>
              <b>Cyclone Risk Rate:</b> {float(row.get('cyclone_risk_rate',0.0))*100:.1f}%<br>
              <b>Wildfire/Landslide:</b> {float(row.get('wildfire_risk_rate',0.0))*100:.1f}%<hr style="margin:6px 0">
              <b>7-Day Rainfall:</b> {float(row.get('rainfall_7d_mm',0.0)):.0f} mm<br>
              <b>Wind Speed:</b> {float(row.get('wind_speed_kmh',0.0)):.0f} km/h<br>
              <b>Temperature:</b> {float(row.get('temperature_c',0.0)):.1f} &deg;C<br>
              <b>Humidity:</b> {float(row.get('humidity_pct',0.0)):.0f} %<br>
              <b>Elevation:</b> {float(row.get('elevation_m',0.0)):.0f} m<br>
              <b>NDVI / NDWI:</b> {float(row.get('ndvi',0.0)):.2f} / {float(row.get('ndwi',0.0)):.2f}
            </div>
            """
            try:
                lat = float(row["latitude"])
                lon = float(row["longitude"])
            except Exception:
                continue
            folium.Marker(
                location=[lat, lon],
                popup=folium.Popup(popup_html, max_width=320),
                tooltip=(
                    f"<b>{row.get('district','Unknown')}</b> &bull; "
                    f"{row.get('disaster_type','Safe')} "
                    f"({float(row.get('risk_score',0.0))*100:.0f}%)"
                ),
                icon=folium.Icon(color=_folium_color(color), icon=icon, prefix="fa"),
            ).add_to(m)
            bounds.append([lat, lon])
    if bounds:
        m.fit_bounds(bounds, padding=(20, 20))
    return m


def _folium_color(css_color: str):
    mapping = {
        "#2ecc71": "green", "#27ae60": "green",
        "#f39c12": "orange", "#e67e22": "orange",
        "#e74c3c": "red", "#95a5a6": "gray",
    }
    return mapping.get(css_color, "blue")


def predict_multi_risk(artifact, X_df: pd.DataFrame):
    model = artifact["model"]
    scaler = artifact["scaler"]
    mode = artifact["scaler_mode"]
    feature_cols = artifact["feature_cols"]

    X = X_df[feature_cols]
    if mode == "scaled":
        X_s = scaler.transform(X)
    else:
        X_s = X

    flood_prob = None
    if artifact["target_col"] == "flood_risk" and hasattr(model, "predict_proba"):
        flood_prob = float(model.predict_proba(X_s)[0, 1])

    r = X_df.iloc[0]
    fp = 0.0
    fp += 0.45 if r["rainfall_7d_mm"] > 250 else 0.2 if r["rainfall_7d_mm"] > 120 else 0.0
    fp += 0.25 if r["ndwi"] > 0.3 else 0.1 if r["ndwi"] > 0.0 else 0.0
    fp += 0.2 if r["elevation_m"] < 30 else 0.1 if r["elevation_m"] < 100 else 0.0
    fp += 0.15 if r["humidity_pct"] > 80 else 0.0
    flood_heur = float(np.clip(fp + np.random.normal(0, 0.05), 0.0, 1.0))
    flood_final = 0.7 * (flood_prob if flood_prob is not None else flood_heur) + 0.3 * flood_heur

    cp = 0.0
    cp += 0.4 if r["wind_speed_kmh"] > 70 else 0.2 if r["wind_speed_kmh"] > 40 else 0.0
    cp += 0.2 if r["humidity_pct"] > 75 else 0.0
    cp += 0.25 if r["elevation_m"] < 50 else 0.1 if r["elevation_m"] < 150 else 0.0
    if X_df.iloc[0].get("district", "") in ["Nagapattinam", "Cuddalore", "Chennai", "Kanyakumari"]:
        cp += 0.15
    cyclone_final = float(np.clip(cp + np.random.normal(0, 0.05), 0.0, 1.0))

    wp = 0.0
    wp += 0.4 if r["temperature_c"] > 35 else 0.2 if r["temperature_c"] > 30 else 0.0
    wp += 0.2 if r["humidity_pct"] < 50 else 0.0
    wp += 0.25 if r["ndvi"] < 0.1 else 0.1 if r["ndvi"] < 0.3 else 0.0
    wp += 0.15 if r["slope_deg"] > 20 else 0.0
    if r["elevation_m"] > 1000 and r["slope_deg"] > 20:
        wp += 0.2
    wildfire_final = float(np.clip(wp + np.random.normal(0, 0.05), 0.0, 1.0))

    scores = {
        "Flood": flood_final,
        "Cyclone": cyclone_final,
        "Wildfire/Landslide": wildfire_final,
    }
    best = max(scores, key=scores.get)
    if scores[best] < 0.35:
        best_type = "Safe"
        best_score = 0.10
    else:
        best_type = best
        best_score = scores[best]
    return scores, best_type, best_score


def gauge_chart(title: str, value: float, color: str):
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=value * 100,
        domain={"x": [0, 1], "y": [0, 1]},
        title={"text": title, "font": {"size": 16}},
        delta={"reference": 50, "increasing": {"color": color},
               "decreasing": {"color": "#2ecc71"}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#888"},
            "bar": {"color": color, "thickness": 0.55},
            "bgcolor": "white",
            "borderwidth": 2,
            "bordercolor": "#e5e7eb",
            "steps": [
                {"range": [0, 40], "color": "#eafaf1"},
                {"range": [40, 70], "color": "#fef9e7"},
                {"range": [70, 85], "color": "#fbeee6"},
                {"range": [85, 100], "color": "#fae5e2"},
            ],
            "threshold": {
                "line": {"color": "#111827", "width": 3},
                "thickness": 0.8, "value": 70,
            },
        },
        number={"suffix": " %", "font": {"size": 26}},
    ))
    fig.update_layout(height=240, margin=dict(l=10, r=10, t=40, b=10))
    return fig


def plot_shap_summary(artifact, X_sample: pd.DataFrame):
    model = artifact["model"]
    feature_cols = artifact["feature_cols"]
    scaler_mode = artifact["scaler_mode"]
    scaler = artifact["scaler"]

    X = X_sample[feature_cols].copy()
    X_bg = X.sample(min(100, len(X)), random_state=42).reset_index(drop=True)

    if scaler_mode == "scaled":
        X_scaled = pd.DataFrame(scaler.transform(X), columns=feature_cols)
        X_bg_s = pd.DataFrame(scaler.transform(X_bg), columns=feature_cols)
    else:
        X_scaled = X.reset_index(drop=True)
        X_bg_s = X_bg

    try:
        explainer = shap.TreeExplainer(model, data=X_bg_s)
        shap_values = explainer.shap_values(X_scaled, check_additivity=False)
        if isinstance(shap_values, list):
            shap_vals = shap_values[1] if len(shap_values) == 2 else shap_values[0]
        else:
            shap_vals = shap_values
    except Exception:
        try:
            bg = shap.kmeans(X_scaled, k=min(10, len(X_scaled)))
            explainer = shap.KernelExplainer(
                lambda x: model.predict_proba(x)[:, 1]
                if hasattr(model, "predict_proba") else model.predict(x),
                bg,
            )
            samp = X_scaled.sample(min(40, len(X_scaled)), random_state=42)
            shap_vals = explainer.shap_values(samp, silent=True, nsamples=64)
            X_scaled = samp
        except Exception as ex:
            return None, ex

    plt.close("all")
    fig, ax = plt.subplots(figsize=(9, 6))
    shap.summary_plot(shap_vals, X_scaled, plot_type="bar", show=False, max_display=10, color="#2563eb")
    ax.set_xlabel("mean(|SHAP value|) — Feature Contribution Magnitude", fontsize=10)
    ax.set_title("SHAP Feature Importance (All classes mean |SHAP|)", fontsize=13, pad=10)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    plt.tight_layout()
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf, None


def sidebar_ui():
    with st.sidebar:
        st.markdown("## 🛰️ TN DISASTER AI")
        st.markdown("#### Tamil Nadu Satellite Intelligence Platform")
        st.markdown("---")
        st.markdown(
            "**Mission** — Fuse weather, elevation & satellite spectral "
            "(NDVI / NDWI) signals into a unified Flood / Cyclone / Wildfire "
            "early warning for TN district EOCs."
        )
        st.markdown("---")
        st.markdown("### 📂 Paste Your CSV Here")
        st.info(
            "Upload a CSV below to load a dataset. No project-folder files are loaded automatically."
        )
        uploaded = st.file_uploader(
            "Upload CSV (any columns work — columns auto-detected)",
            type=["csv"], accept_multiple_files=False, key="csv_upload",
            help="Supports district, lat, lon, elevation, temperature, humidity, wind, rain, ndvi, ndwi, flood_risk, cyclone_risk, wildfire_risk, disaster_type column names & synonyms.",
        )
        st.markdown("### 🧭 Data Inputs")
        data_ready = uploaded is not None
        model_ready = os.path.exists(MODEL_PATH)
        if data_ready:
            st.markdown(f"✅ Dataset selected: `{uploaded.name}`")
        else:
            st.markdown("❌ No dataset found — upload a CSV above.")
        st.markdown(f"Model  : {'✅ `models/best_model.pkl`' if model_ready else '❌ Missing — run `model_trainer.py`'}")
        st.markdown("---")
        st.markdown("### 🛠️ Quick Actions")
        c1, c2 = st.columns(2)
        if c1.button("🔄 Reset Data", use_container_width=True):
            st.cache_data.clear()
            st.cache_resource.clear()
            st.success("Data caches cleared. Upload a CSV to load a dataset.")
        if c2.button("🧪 Reload Model", use_container_width=True):
            st.cache_resource.clear()
            st.success("Model cache cleared — model reloaded next render.")
        st.markdown("---")
        st.markdown("### 🚀 Launch Instructions")
        st.code(
            "pip install -r requirements.txt\n"
            "# (optional: generate sample TN data)\n"
            "python data_generator.py\n"
            "# Train on your CSV (paste in folder first)\n"
            "python model_trainer.py your_csv.csv flood_risk\n"
            "# Launch dashboard\n"
            "streamlit run app.py",
            language="bash",
        )
        st.markdown(
            '<p class="foot">v1.1 • Custom-CSV Ready<br>'
            "For research/prototyping only — not for live ops.</p>",
            unsafe_allow_html=True,
        )
        return uploaded


def tab1_map_overview(df, agg, raw_cols=None, remap=None, source_path=None):
    st.markdown(
        '<h1>🛰️ Tamil Nadu — <span class="hl">Satellite Disaster Intelligence</span> Map</h1>',
        unsafe_allow_html=True,
    )
    st.caption(
        "GIS overview of districts monitored for Flood, Cyclone & Wildfire/Landslide "
        "risk using fused weather + elevation + NDVI/NDWI satellite indicators."
    )

    no_data = (df is None) or (agg is None) or (len(df) == 0) or (len(agg) == 0)

    info_cols = st.columns([2, 2, 3, 1.3])
    with info_cols[0]:
        if no_data:
            st.warning("📂 No dataset loaded yet.")
        elif source_path:
            st.success(f"📂 Dataset loaded from `{os.path.basename(source_path)}`")
        else:
            st.success("📂 Dataset loaded via CSV upload (in-memory)")
    with info_cols[1]:
        if remap and not no_data:
            with st.popover(f"🔀 {len(remap)} columns auto-mapped"):
                st.markdown("**Your columns → internal names**")
                for k, v in remap.items():
                    st.markdown(f"- `{k}` → **{v}**")
        elif no_data:
            st.info("💡 Upload CSV in sidebar or paste file in the project folder.")
        else:
            st.info("🎯 Columns matched internal schema exactly (no remap needed)")
    with info_cols[2]:
        if raw_cols is not None and not no_data:
            st.caption(
                f"**Original columns in your CSV ({len(raw_cols)}):** "
                + "`" + "` · `".join(raw_cols[:14]) + ("` ..." if len(raw_cols) > 14 else "`")
            )

    total_nodes = 0 if no_data else int(agg["n_rows"].sum())
    high_risk = 0 if no_data else int(((agg["risk_score"] >= 0.7).sum()))
    avg_rain = 0.0 if no_data else float(agg["rainfall_7d_mm"].mean())
    total_high_events = 0 if no_data else int(((df["disaster_type"] != "Safe").sum()))
    n_districts = 0 if no_data else int(agg["district"].nunique())

    mc1, mc2, mc3, mc4 = st.columns(4)
    mc1.metric("📡 District Nodes Monitored", f"{total_nodes:,}", f"{n_districts} districts")
    mc2.metric("🚨 High-Risk Zones Alert", high_risk, "score ≥ 70%", delta_color="inverse")
    mc3.metric("🌧️ Avg 7-Day Rainfall", f"{avg_rain:.0f} mm", "across data")
    mc4.metric("🌀 Active Hazard Event Nodes", f"{total_high_events:,}",
               f"{0 if no_data else (total_high_events / max(1, total_nodes)) * 100:.1f}% of samples")

    st.markdown("---")

    fc1, fc2, fc3, fc4 = st.columns([1, 1, 0.8, 2])
    with fc1:
        sev_filter = st.multiselect(
            "Filter by Severity",
            ["safe", "low", "medium", "high", "critical"],
            default=["medium", "high", "critical", "low", "safe"],
            format_func=lambda x: x.title(),
            disabled=no_data,
        )
    with fc2:
        type_opts = ["All Hazards", "Flood", "Cyclone", "Wildfire/Landslide", "Safe"]
        show_type = st.selectbox(
            "Disaster Type Highlight", type_opts,
            disabled=no_data,
        )
    with fc3:
        tile_labels = [t[0] for t in FOLIUM_TILES]
        tile_pick = st.selectbox("Base Map", tile_labels, index=0,
                                 help="Switch between OpenStreetMap and Esri imagery. No API key is required.")
        tile_choice = tile_labels.index(tile_pick) if tile_pick in tile_labels else 0
    with fc4:
        st.caption(
            "🧭 **How to read**: Red = Critical (≥85%), Orange = Warning (70–85%), "
            "Yellow = Medium (40–70%), Green = Safe/Low (<40%). Click a marker for full weather and risk breakdown."
        )

    map_agg = None if no_data else agg.copy()
    if (not no_data) and show_type != "All Hazards":
        map_agg = map_agg[map_agg["disaster_type"] == show_type]

    folium_map = build_folium_map(map_agg, selected_severities=sev_filter, tile_choice=tile_choice)
    with st.container(border=True):
        if no_data:
            st.info(
                "🗺️ **Map is empty** because no dataset is loaded.  \n"
                "To populate markers:\n"
                "- **Option A:** Drop your CSV in the sidebar file uploader, OR\n"
                "- **Option B:** Paste a CSV named `tamilnadu_disaster_data.csv` or `disaster_data.csv` "
                f"into `{BASE_DIR}` and click **🔄 Reset Data** in the sidebar."
            )
        st_folium(folium_map, width=1400, height=620, returned_objects=[])

    st.markdown("#### District Risk Table")
    if no_data:
        st.info("No rows to display — 0 districts, 0 samples. Upload a CSV to populate.")
        empty = pd.DataFrame({
            "district": [], "disaster_type": [], "risk_score": [],
            "flood_risk_rate": [], "cyclone_risk_rate": [], "wildfire_risk_rate": [],
            "rainfall_7d_mm": [], "wind_speed_kmh": [], "temperature_c": [],
            "humidity_pct": [], "elevation_m": [], "ndvi": [], "ndwi": [],
        })
        st.dataframe(empty, use_container_width=True, height=480, hide_index=True)
        return

    show_df = agg[[
        "district", "disaster_type", "risk_score",
        "flood_risk_rate", "cyclone_risk_rate", "wildfire_risk_rate",
        "rainfall_7d_mm", "wind_speed_kmh", "temperature_c", "humidity_pct",
        "elevation_m", "ndvi", "ndwi",
    ]].copy()
    for col in ["flood_risk_rate", "cyclone_risk_rate", "wildfire_risk_rate"]:
        show_df[col] = (show_df[col] * 100).round(1).astype(str) + " %"
    show_df["risk_score"] = (show_df["risk_score"] * 100).round(1).astype(str) + " %"
    st.dataframe(show_df, use_container_width=True, height=480, hide_index=True)


def tab2_prediction(artifact, df):
    if df is None or df.empty:
        st.markdown(
            '<h1>🔮 Live Disaster <span class="hl">Prediction & AI Response</span></h1>',
            unsafe_allow_html=True,
        )
        st.info("No dataset loaded. Upload a CSV or paste one into the project folder to enable district prediction.")
        return

    st.markdown(
        '<h1>🔮 Live Disaster <span class="hl">Prediction & AI Response</span></h1>',
        unsafe_allow_html=True,
    )
    st.caption(
        "Select a district (pulled from your CSV), or override the weather sliders, to predict Flood, Cyclone "
        "and Wildfire/Landslide probabilities and trigger automated SDMA action protocols."
    )

    left, right = st.columns([1.1, 1.4], gap="large")

    with left:
        st.markdown("### 🌐 District & Environmental Controls")
        district_options = sorted(
            list({str(d) for d in df["district"].dropna().unique()})
        ) if df is not None and "district" in df.columns else list(DISTRICT_META.keys())
        default_idx = 0
        district = st.selectbox(
            "Select District (from your CSV)", district_options,
            index=default_idx,
        )

        if district in DISTRICT_META:
            meta = DISTRICT_META[district]
        elif df is not None and "district" in df.columns:
            sub = df[df["district"].astype(str) == str(district)]
            lat0 = float(sub["latitude"].median()) if len(sub) else TN_CENTER["lat"]
            lon0 = float(sub["longitude"].median()) if len(sub) else TN_CENTER["lon"]
            elev0 = float(sub["elevation_m"].median()) if len(sub) else 100.0
            slope0 = float(sub["slope_deg"].median()) if len(sub) else 8.0
            meta = {"lat": lat0, "lon": lon0, "elev": elev0, "slope": slope0}
            for key in ["rainfall_7d_mm", "wind_speed_kmh", "temperature_c",
                        "humidity_pct", "ndvi", "ndwi"]:
                if len(sub):
                    meta[key] = float(sub[key].median())
        else:
            meta = {"lat": TN_CENTER["lat"], "lon": TN_CENTER["lon"], "elev": 100.0, "slope": 10.0}

        st.markdown("#### Weather & Satellite Sliders")
        default_rain = meta.get("rainfall_7d_mm", 80.0)
        default_wind = meta.get("wind_speed_kmh", 25.0)
        default_temp = meta.get("temperature_c", 32.0)
        default_hum = meta.get("humidity_pct", 65.0)
        default_ndvi = meta.get("ndvi", 0.25)
        default_ndwi = meta.get("ndwi", 0.0)

        rf = st.slider("7-Day Rainfall (mm)", 0.0, 800.0, float(default_rain), 5.0)
        ws = st.slider("Wind Speed (km/h)", 0.0, 150.0, float(default_wind), 1.0)
        tp = st.slider("Temperature (°C)", 10.0, 50.0, float(default_temp), 0.5)
        hu = st.slider("Humidity (%)", 5.0, 100.0, float(default_hum), 1.0)
        ndvi = st.slider("NDVI (Vegetation Index)", -0.30, 0.90, float(default_ndvi), 0.01)
        ndwi = st.slider("NDWI (Water Index)", -0.90, 0.90, float(default_ndwi), 0.01)
        elev = st.slider("Elevation (m)", 0.0, 3000.0, float(meta["elev"]), 1.0)
        slp = st.slider("Slope (degrees)", 0.0, 45.0, float(meta["slope"]), 0.5)

        st.info(
            f"📍 **{district}** baseline — elev={meta['elev']:.0f} m, "
            f"slope={meta['slope']:.0f}°, centroid ({meta['lat']:.2f}°, {meta['lon']:.2f}°)"
        )

    X_row = pd.DataFrame([{
        "latitude": meta["lat"],
        "longitude": meta["lon"],
        "elevation_m": elev,
        "slope_deg": slp,
        "temperature_c": tp,
        "humidity_pct": hu,
        "wind_speed_kmh": ws,
        "rainfall_7d_mm": rf,
        "ndvi": ndvi,
        "ndwi": ndwi,
        "district": district,
    }])

    with right:
        st.markdown("### 🎯 Probabilistic Hazard Assessment")
        if artifact is None:
            st.warning("⚠️ Model artifact not found (`models/best_model.pkl`). Running heuristic fallback scorer.")
            scores, dtype, dscore = predict_multi_risk(
                {"model": None, "scaler": None, "scaler_mode": "raw",
                 "feature_cols": ["latitude","longitude","elevation_m","slope_deg",
                                  "temperature_c","humidity_pct","wind_speed_kmh",
                                  "rainfall_7d_mm","ndvi","ndwi"],
                 "target_col": "flood_risk"},
                X_row,
            )
        else:
            scores, dtype, dscore = predict_multi_risk(artifact, X_row)

        gc1, gc2, gc3 = st.columns(3)
        gc1.plotly_chart(gauge_chart("🌊 Flood Probability",
                                      scores["Flood"], "#2563eb"),
                         use_container_width=True)
        gc2.plotly_chart(gauge_chart("🌀 Cyclone Probability",
                                      scores["Cyclone"], "#7c3aed"),
                         use_container_width=True)
        gc3.plotly_chart(gauge_chart("🔥 Wildfire / Landslide Probability",
                                      scores["Wildfire/Landslide"], "#dc2626"),
                         use_container_width=True)

        scores_df = pd.DataFrame([
            {"Hazard": "🌊 Flood", "Probability (%)": round(scores["Flood"] * 100, 1)},
            {"Hazard": "🌀 Cyclone", "Probability (%)": round(scores["Cyclone"] * 100, 1)},
            {"Hazard": "🔥 Wildfire/Landslide", "Probability (%)": round(scores["Wildfire/Landslide"] * 100, 1)},
        ])
        fig = px.bar(scores_df, x="Hazard", y="Probability (%)", color="Probability (%)",
                     color_continuous_scale=["#2ecc71", "#f1c40f", "#e67e22", "#e74c3c"],
                     text="Probability (%)", range_y=[0, 100],
                     title="Hazard Probability Bar View")
        fig.update_traces(texttemplate="%{text} %", textposition="outside")
        fig.update_layout(showlegend=False, margin=dict(l=20, r=20, t=50, b=20))
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        rec = generate_tn_recommendations(district, dtype, dscore)
        sev = rec["severity"]
        sev_pill_color = {
            "safe": "pill-safe", "low": "pill-safe", "medium": "pill-warn",
            "high": "pill-warn", "critical": "pill-crit",
        }.get(sev, "pill-warn")

        st.markdown(f"## {rec['alert_title']}  "
                    f'<span class="risk-pill {sev_pill_color}">{sev.upper()} • {dscore*100:.0f}%</span>',
                    unsafe_allow_html=True)

        st.markdown(
            f"<div class=\"sev-{sev}\"><div class=\"rec-card\"><b>📋 AI Situation Summary:</b> {rec['summary']}</div></div>",
            unsafe_allow_html=True,
        )

        with st.container(border=True):
            st.markdown("### 🚨 Actionable AI Protocols (SDMA / DEOC)")
            actions = [str(a).strip() for a in rec.get("key_actions", []) if str(a).strip()]
            if not actions:
                actions = ["No district-specific action required. Continue routine monitoring."]
            for i, action in enumerate(actions, 1):
                st.markdown(f"**Step {i}.** {action}")

        ac1, ac2 = st.columns(2)
        with ac1.container(border=True):
            st.markdown("#### 🏛️ Lead Response Agencies")
            for ag in rec["agencies"]:
                st.markdown(f"- 🟦 {ag}")
        with ac2.container(border=True):
            st.markdown("#### 📞 Emergency Helplines (TN)")
            for k, v in rec["helplines"].items():
                st.markdown(f"**{k}:** `{v}`")

        if rec["infra"]:
            with st.expander("🏗️ Pre-Positioned Infrastructure Check-list", expanded=True):
                for i, item in enumerate(rec["infra"], 1):
                    st.markdown(f"{i}. ☐ {item}")


def tab3_benchmarks(artifact, benchmarks, feat_imp, df=None):
    st.markdown(
        '<h1>🧬 ML Model <span class="hl">Benchmarks & Data Science</span></h1>',
        unsafe_allow_html=True,
    )
    st.caption(
        "Flood-risk classifier comparison across Random Forest, XGBoost, LightGBM and CatBoost. "
        "Best model saved to `models/best_model.pkl`."
    )

    if df is None or df.empty:
        st.info("No dataset is loaded, so model metrics are unavailable. Upload a CSV or paste one into the project folder to enable the benchmark section.")
        return

    if benchmarks is None or benchmarks.empty:
        st.warning("Benchmark file not found. Please run `python model_trainer.py`.")
        return

    bc1, bc2, bc3 = st.columns(3)
    best_row = benchmarks.iloc[0]
    bc1.metric("🏆 Best Model", best_row["Model"], f"F1={best_row['F1-Score']:.3f}")
    bc2.metric("🎯 F1-Score (Top)", f"{best_row['F1-Score']:.4f}",
               f"vs. Worst {benchmarks['F1-Score'].min():.3f}")
    bc3.metric("📈 ROC-AUC (Top)", f"{best_row['ROC-AUC']:.4f}",
               f"Accuracy {best_row['Accuracy']:.3f}")

    st.markdown("---")
    st.markdown("### 📊 Model Comparison Table")
    styled = benchmarks.style.highlight_max(subset=["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"],
                                             color="#ffffff")
    styled = styled.set_properties(**{"color": "#000000", "font-weight": "600", "background-color": "#ffffff"})
    st.dataframe(styled, use_container_width=True, hide_index=True, height=240)

    ch1, ch2 = st.columns(2, gap="large")
    with ch1:
        fig_f1 = px.bar(benchmarks, x="Model", y=["F1-Score", "ROC-AUC"],
                        barmode="group",
                        title="F1-Score vs ROC-AUC by Classifier",
                        color_discrete_sequence=["#2563eb", "#7c3aed"],
                        text_auto=".3f")
        fig_f1.update_layout(legend_title_text="Metric", yaxis_range=[0.5, 1.02],
                             margin=dict(l=20, r=20, t=60, b=20))
        st.plotly_chart(fig_f1, use_container_width=True)

    with ch2:
        metrics = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
        melted = benchmarks.melt(id_vars="Model", value_vars=metrics, var_name="Metric", value_name="Score")
        fig_radar = px.line_polar(melted, r="Score", theta="Metric", color="Model",
                                  line_close=True, markers=True,
                                  title="Multi-Metric Radar Comparison",
                                  range_r=[0.6, 1.02])
        fig_radar.update_layout(polar=dict(radialaxis_tickformat=".2f"),
                                margin=dict(l=20, r=20, t=60, b=20),
                                legend=dict(orientation="h", yanchor="top", y=-0.1))
        st.plotly_chart(fig_radar, use_container_width=True)

    st.markdown("---")
    st.markdown("### 🧩 Feature Importance (Best Model Global)")
    if feat_imp is not None and not feat_imp.empty:
        fi = feat_imp.sort_values("Importance", ascending=True)
        fig_fi = go.Figure(go.Bar(
            x=fi["Importance"], y=fi["Feature"],
            orientation="h",
            marker={"color": fi["Importance"], "colorscale": "Blugrn"},
            text=fi["Importance"].round(4), textposition="outside",
        ))
        fig_fi.update_layout(title=f"Feature Importance — {best_row['Model']} (Mean Decrease Impurity)",
                             xaxis_title="Importance Score", yaxis_title="",
                             height=520, margin=dict(l=10, r=20, t=60, b=20))
        st.plotly_chart(fig_fi, use_container_width=True)

        with st.expander("📋 Feature Glossary", expanded=False):
            glossary = pd.DataFrame([
                ["rainfall_7d_mm", "Cumulative 7-day rainfall (mm)", "Primary flood driver; 250mm+ triggers high risk"],
                ["ndwi", "Normalized Difference Water Index (satellite)", "Standing water / soil wetness indicator"],
                ["elevation_m", "Elevation above MSL (metres)", "<30 m = coastal flood / cyclone exposure"],
                ["wind_speed_kmh", "10-m sustained wind (km/h)", "≥70 km/h = cyclone landfall conditions"],
                ["humidity_pct", "Relative humidity (%)", "Low humidity wildfire; high humidity cyclone/flood"],
                ["temperature_c", "2-m air temperature (°C)", ">35°C amplifies wildfire danger"],
                ["ndvi", "Normalized Difference Vegetation Index (satellite)", "<0.1 = bare/dry fire fuel"],
                ["slope_deg", "Terrain slope (degrees)", ">20° amplifies landslide runout"],
                ["latitude/longitude", "District centroid coordinates", "Spatial prior for coastal vs inland"],
            ], columns=["Feature", "Definition", "Operational Relevance"])
            st.table(glossary)
    else:
        st.info("Feature importance file missing — regenerate with `model_trainer.py`.")


def tab4_shap(artifact, df):
    st.markdown(
        '<h1>🔍 Explainable AI — <span class="hl">SHAP Interpretation</span></h1>',
        unsafe_allow_html=True,
    )
    st.caption(
        "SHAP (SHapley Additive exPlanations) decomposes predictions to show which sensors "
        "(rainfall, NDWI, elevation, wind, ...) pushed the flood-risk score up or down."
    )

    if df is None or df.empty:
        st.info("No dataset is loaded, so SHAP cannot explain anything yet. Upload a CSV to populate the dashboard.")
        return

    if artifact is None:
        st.warning("Model artifact not found. Please run `python model_trainer.py` first.")
        return

    sc1, sc2, sc3 = st.columns([1.4, 1, 1])
    with sc1:
        district_opts = ["All"] + sorted(
            list({str(d) for d in df["district"].dropna().unique()})
        ) if df is not None and "district" in df.columns else ["All"] + list(DISTRICT_META.keys())
        shap_district = st.selectbox(
            "District context for background data", district_opts,
            index=0, key="shap_dist",
        )
    with sc2:
        n_background = st.slider("Background size", 30, 200, 100, 10)
    with sc3:
        n_eval = st.slider("Rows to explain", 20, 150, 80, 5)

    df_local = df.copy()
    if shap_district != "All":
        df_local = df_local[df_local["district"] == shap_district]
    if len(df_local) < 30:
        df_local = df.copy()

    st.info(
        f"Running SHAP on {min(n_eval, len(df_local))} rows, background={n_background}. "
        f"Model = **{artifact.get('best_model_name','N/A')}**."
    )

    feature_cols = artifact["feature_cols"]
    X_sample = df_local.sample(min(n_eval, len(df_local)), random_state=42).reset_index(drop=True)

    X = X_sample[feature_cols].copy()
    model = artifact["model"]
    mode = artifact["scaler_mode"]
    scaler = artifact["scaler"]
    if mode == "scaled":
        X_bg = pd.DataFrame(scaler.transform(X.sample(min(n_background, len(X)), random_state=42)),
                            columns=feature_cols)
        X_scaled = pd.DataFrame(scaler.transform(X), columns=feature_cols)
    else:
        X_bg = X.sample(min(n_background, len(X)), random_state=42).reset_index(drop=True)
        X_scaled = X.reset_index(drop=True)

    shap_vals = None
    explainer = None
    shap_err = None
    try:
        explainer = shap.TreeExplainer(model, data=X_bg)
        shap_vals_raw = explainer.shap_values(X_scaled, check_additivity=False)
        if isinstance(shap_vals_raw, list):
            shap_vals = shap_vals_raw[1] if len(shap_vals_raw) == 2 else shap_vals_raw[0]
        else:
            shap_vals = shap_vals_raw
        shap_vals = _normalize_shap_values(shap_vals)
    except Exception as ex1:
        try:
            bg = shap.kmeans(X_scaled, k=min(8, len(X_scaled)))
            pred_fn = (lambda x: model.predict_proba(x)[:, 1]) \
                if hasattr(model, "predict_proba") else (lambda x: model.predict(x))
            explainer = shap.KernelExplainer(pred_fn, bg)
            shap_vals = explainer.shap_values(X_scaled.head(40), silent=True, nsamples=80)
            shap_vals = _normalize_shap_values(shap_vals)
            X_scaled = X_scaled.head(40)
            X_sample = X_sample.head(40)
        except Exception as ex2:
            shap_err = (ex1, ex2)

    if shap_err is not None:
        st.error(f"SHAP explainer failed. Tree fallback: {shap_err[0]}. Kernel fallback: {shap_err[1]}")
        return

    if shap_vals is None or np.asarray(shap_vals).size == 0:
        st.warning("SHAP produced no values for this dataset. Please upload a valid CSV with modeled features.")
        return

    col1, col2 = st.columns(2, gap="large")

    with col1:
        plt.close("all")
        fig1, ax1 = plt.subplots(figsize=(9, 6))
        shap.summary_plot(_normalize_shap_values(shap_vals), X_scaled, plot_type="bar",
                          show=False, max_display=10, color="#2563eb")
        ax1.set_title("SHAP Bar: Feature Importance (mean |SHAP|)", fontsize=13, pad=10)
        for spine in ["top", "right"]:
            ax1.spines[spine].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig1, use_container_width=True)
        plt.close(fig1)

    with col2:
        plt.close("all")
        fig2, ax2 = plt.subplots(figsize=(9, 6))
        shap.summary_plot(_normalize_shap_values(shap_vals), X_scaled, plot_type="dot",
                          show=False, max_display=10)
        ax2.set_title("SHAP Beeswarm: Feature Direction & Magnitude", fontsize=13, pad=10)
        for spine in ["top", "right"]:
            ax2.spines[spine].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig2, use_container_width=True)
        plt.close(fig2)

    st.markdown("---")
    st.markdown("### 🧪 Single-Row SHAP Explanation (Waterfall)")
    wc1, wc2 = st.columns([1, 3])
    with wc1:
        row_choice = st.selectbox("Choose row by index", list(range(len(X_sample))), index=0)
    one_x = X_scaled.iloc[[row_choice]].reset_index(drop=True)
    selected_district_name = str(X_sample.iloc[row_choice].get("district", shap_district))
    district_label = "All districts (mixed sample)" if shap_district == "All" else selected_district_name
    try:
        if explainer is not None:
            if hasattr(explainer, "expected_value"):
                base = _safe_scalar(explainer.expected_value)
                shap_matrix = _normalize_shap_values(shap_vals)
                if shap_matrix.ndim == 1:
                    shap_matrix = shap_matrix.reshape(1, -1)
                shap_row = np.asarray(shap_matrix[row_choice], dtype=float)
                if shap_row.size == 0:
                    st.info("Selected row has no SHAP contributions.")
                    return
                plt.close("all")
                fig3, ax3 = plt.subplots(figsize=(11, 6))
                order = np.argsort(-np.abs(shap_row))
                sorted_idx = order[:10]
                names = one_x.columns[sorted_idx]
                vals = shap_row[sorted_idx]
                colors = ["#2563eb" if v >= 0 else "#e74c3c" for v in vals]
                ypos = np.arange(len(names))
                bars = []
                lefts = []
                last = base
                for v in vals:
                    lefts.append(last)
                    bars.append(abs(v))
                    last = last + v
                ax3.barh(ypos, bars, left=lefts, color=colors, edgecolor="white", linewidth=1.2)
                ax3.axvline(base, color="#111827", linestyle="--", linewidth=1.2,
                            label=f"Base (E[f(x)] = {base:.3f})")
                ax3.axvline(base + vals.sum(), color="#22c55e", linestyle=":", linewidth=1.5,
                            label=f"Prediction = {base + vals.sum():.3f}")
                for i, (nm, vl, lf) in enumerate(zip(names, vals, lefts)):
                    ax3.text(lf + abs(vl) + 0.001, i, f"{vl:+.3f}",
                             va="center", fontsize=9, fontweight="bold",
                             color="#111827")
                ax3.set_yticks(ypos)
                ax3.set_yticklabels(names, fontsize=10)
                ax3.invert_yaxis()
                ax3.set_xlabel("SHAP value contribution to flood-risk log-odds / score")
                ax3.set_title(f"SHAP Waterfall — Row #{row_choice} ({district_label})",
                              fontsize=13, pad=12)
                for spine in ["top", "right"]:
                    ax3.spines[spine].set_visible(False)
                ax3.legend(loc="lower right")
                plt.tight_layout()
                st.pyplot(fig3, use_container_width=True)
                plt.close(fig3)

                with st.expander("🔬 Raw sensor values for the selected row", expanded=True):
                    raw_row = X_sample.iloc[row_choice][[
                        "district","rainfall_7d_mm","wind_speed_kmh","temperature_c",
                        "humidity_pct","ndvi","ndwi","elevation_m","slope_deg"]].to_dict()
                    if shap_district == "All":
                        raw_row["district"] = "All districts (mixed sample)"
                    st.json({k: (round(v, 3) if isinstance(v, float) else v) for k, v in raw_row.items()})
    except Exception as e:
        st.error(f"Waterfall error: {e}")


def main():
    uploaded = sidebar_ui()
    if "has_used_upload" not in st.session_state:
        st.session_state.has_used_upload = False

    df = None
    remap = {}
    raw_cols = None
    source_path = None
    if uploaded is not None:
        st.session_state.has_used_upload = True
        try:
            file_bytes = uploaded.getvalue()
            df, remap, raw_df = load_dataset_from_bytes(file_bytes)
            raw_cols = list(raw_df.columns)
        except Exception as ex:
            st.error(f"Failed to read uploaded CSV: {ex}")
            df, remap, raw_df, source_path = None, {}, None, None
    elif st.session_state.get("has_used_upload"):
        df = empty_dashboard_frame()
        remap = {}
        raw_cols = []
        source_path = None
    else:
        df = empty_dashboard_frame()
        remap = {}
        raw_cols = []
        source_path = None

    if df is None:
        df = empty_dashboard_frame()
        remap = {}
        raw_cols = []
        source_path = None

    artifact = load_model_artifact()
    benchmarks = load_benchmarks()
    feat_imp = load_feature_importance()

    if df is None or df.empty:
        st.title("📂 No Dataset Found")
        st.warning(
            "No dataset is loaded. Upload your CSV using the sidebar file uploader to populate the dashboard."
        )
        with st.expander("🧾 Expected CSV Column Names (any order, any case)"):
            cols_help = pd.DataFrame([
                ["district*", "district / place / location / region / taluk / city / block",
                 "If missing: first text column is used, else generated 'Unknown_Area'"],
                ["latitude*", "latitude / lat / y",
                 "If missing + no longitude: TN-range lat/lon generated automatically"],
                ["longitude*", "longitude / lon / lng / long / x", ""],
                ["elevation_m", "elevation / elev / elev_m / height / altitude / altitude_m / dem / srtm",
                 "Fill: clipped uniform [0, 3000] m"],
                ["slope_deg", "slope / slope_degrees / terrain_slope",
                 "Fill: clipped uniform [0, 45] °"],
                ["temperature_c", "temperature / temp / temp_c / t2m",
                 "Fill: clipped uniform [15, 45] °C"],
                ["humidity_pct", "humidity / hum / rh / relative_humidity",
                 "Fill: clipped uniform [10, 100] %"],
                ["wind_speed_kmh", "wind_speed / wind / wspd / ws / wind_kmh / wind_kmph",
                 "Fill: clipped uniform [0, 150] km/h"],
                ["rainfall_7d_mm", "rainfall / rain / precip / rain_mm / pr / rainfall_7d / cum_rain",
                 "Fill: clipped uniform [0, 800] mm"],
                ["ndvi", "ndvi / ndvi_mean / modis_ndvi",
                 "Fill: clipped uniform [-0.3, 0.9]"],
                ["ndwi", "ndwi / ndwi_mean / mndwi / modis_ndwi",
                 "Fill: clipped uniform [-0.9, 0.9]"],
                ["flood_risk", "flood_risk / flood / flood_flag",
                 "If missing + train needed: compute heuristically"],
                ["cyclone_risk", "cyclone_risk / cyclone / cyclone_flag",
                 "If missing: compute heuristically"],
                ["wildfire_risk", "wildfire_risk / wildfire / landslide_risk / landslide",
                 "If missing: compute heuristically"],
                ["disaster_type", "disaster_type / hazard_type / type / hazard",
                 "If missing: compute from three risks"],
            ], columns=["Internal name", "Your CSV column names accepted", "If missing from your CSV"])
            st.table(cols_help)
        benchmarks = None
        feat_imp = None
        st.stop()

    agg = district_risk_aggregates(df)

    t1, t2, t3, t4 = st.tabs([
        "🗺️ 1. GIS Map & Overview",
        "🔮 2. Live Prediction & AI Response",
        "🧬 3. Model Benchmarks & Science",
        "🔍 4. Explainable AI (SHAP)",
    ])

    with t1:
        tab1_map_overview(df, agg, raw_cols=raw_cols, remap=remap, source_path=source_path)
    with t2:
        tab2_prediction(artifact, df)
    with t3:
        tab3_benchmarks(artifact, benchmarks, feat_imp, df=df)
    with t4:
        tab4_shap(artifact, df)


if __name__ == "__main__":
    main()
