import os
import sys
import numpy as np
import pandas as pd
import random

np.random.seed(42)
random.seed(42)

DEFAULT_OUTPUT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "tamilnadu_disaster_data.csv",
)

TN_DISTRICTS = {
    "Chennai": {"lat": 13.0827, "lon": 80.2707, "elev": 6, "slope": 2},
    "Cuddalore": {"lat": 11.7474, "lon": 79.7694, "elev": 8, "slope": 3},
    "Thiruvallur": {"lat": 13.1193, "lon": 79.9817, "elev": 12, "slope": 4},
    "Nilgiris": {"lat": 11.4906, "lon": 76.7343, "elev": 2000, "slope": 38},
    "Nagapattinam": {"lat": 10.7641, "lon": 79.8439, "elev": 5, "slope": 2},
    "Kanchipuram": {"lat": 12.8343, "lon": 79.7037, "elev": 72, "slope": 6},
    "Madurai": {"lat": 9.9252, "lon": 78.1198, "elev": 157, "slope": 8},
    "Coimbatore": {"lat": 11.0168, "lon": 76.9558, "elev": 411, "slope": 15},
    "Kanyakumari": {"lat": 8.0883, "lon": 77.5385, "elev": 32, "slope": 9},
    "Tirunelveli": {"lat": 8.7139, "lon": 77.7567, "elev": 111, "slope": 10},
    "Thanjavur": {"lat": 10.7867, "lon": 79.1378, "elev": 47, "slope": 3},
    "Salem": {"lat": 11.6643, "lon": 78.1460, "elev": 278, "slope": 12},
    "Tiruchirappalli": {"lat": 10.7905, "lon": 78.7047, "elev": 78, "slope": 5},
    "Vellore": {"lat": 12.9165, "lon": 79.1325, "elev": 226, "slope": 11},
    "Erode": {"lat": 11.3307, "lon": 77.7172, "elev": 193, "slope": 10},
}

N_ROWS = 2000
districts_list = list(TN_DISTRICTS.keys())

rows = []
for i in range(N_ROWS):
    district = random.choice(districts_list)
    meta = TN_DISTRICTS[district]

    latitude = meta["lat"] + np.random.normal(0, 0.15)
    longitude = meta["lon"] + np.random.normal(0, 0.15)
    elevation_m = max(0.0, meta["elev"] + np.random.normal(0, meta["elev"] * 0.1 + 10))
    slope_deg = min(45.0, max(0.0, meta["slope"] + np.random.normal(0, 4)))

    temperature_c = np.random.uniform(20.0, 42.0)
    if meta["elev"] > 1000:
        temperature_c = np.random.uniform(15.0, 28.0)
    elif meta["elev"] > 200:
        temperature_c = np.random.uniform(22.0, 38.0)

    humidity_pct = np.random.uniform(30.0, 95.0)
    if district in ["Nagapattinam", "Cuddalore", "Kanyakumari", "Thiruvallur", "Chennai"]:
        humidity_pct = np.random.uniform(60.0, 95.0)

    wind_speed_kmh = np.random.uniform(5.0, 120.0)
    if district in ["Nagapattinam", "Cuddalore", "Kanyakumari", "Chennai"]:
        wind_speed_kmh = np.random.uniform(20.0, 120.0)

    rainfall_7d_mm = np.random.uniform(0.0, 450.0)
    if district in ["Nilgiris", "Kanyakumari", "Cuddalore", "Nagapattinam"]:
        rainfall_7d_mm = np.random.uniform(50.0, 450.0)
    elif district in ["Madurai", "Tirunelveli"]:
        rainfall_7d_mm = np.random.uniform(0.0, 200.0)

    ndvi = np.random.uniform(-0.2, 0.8)
    if meta["elev"] > 500:
        ndvi = np.random.uniform(0.2, 0.8)
    elif district in ["Chennai", "Madurai"]:
        ndvi = np.random.uniform(-0.1, 0.4)

    ndwi = np.random.uniform(-0.8, 0.8)
    if district in ["Cuddalore", "Nagapattinam", "Thanjavur"]:
        ndwi = np.random.uniform(-0.1, 0.8)
    elif district in ["Madurai", "Tirunelveli", "Coimbatore"]:
        ndwi = np.random.uniform(-0.8, 0.2)

    flood_prob = 0.0
    flood_prob += 0.45 if rainfall_7d_mm > 250 else 0.2 if rainfall_7d_mm > 120 else 0.0
    flood_prob += 0.25 if ndwi > 0.3 else 0.1 if ndwi > 0.0 else 0.0
    flood_prob += 0.2 if elevation_m < 30 else 0.1 if elevation_m < 100 else 0.0
    flood_prob += 0.15 if humidity_pct > 80 else 0.0
    flood_risk = 1 if (flood_prob + np.random.normal(0, 0.1)) > 0.5 else 0

    cyclone_prob = 0.0
    cyclone_prob += 0.4 if wind_speed_kmh > 70 else 0.2 if wind_speed_kmh > 40 else 0.0
    cyclone_prob += 0.2 if humidity_pct > 75 else 0.0
    cyclone_prob += 0.25 if elevation_m < 50 else 0.1 if elevation_m < 150 else 0.0
    cyclone_prob += 0.15 if district in ["Nagapattinam", "Cuddalore", "Chennai", "Kanyakumari"] else 0.0
    cyclone_risk = 1 if (cyclone_prob + np.random.normal(0, 0.1)) > 0.5 else 0

    wildfire_prob = 0.0
    wildfire_prob += 0.4 if temperature_c > 35 else 0.2 if temperature_c > 30 else 0.0
    wildfire_prob += 0.2 if humidity_pct < 50 else 0.0
    wildfire_prob += 0.25 if ndvi < 0.1 else 0.1 if ndvi < 0.3 else 0.0
    wildfire_prob += 0.15 if slope_deg > 20 else 0.0
    if elevation_m > 1000 and slope_deg > 20:
        wildfire_prob += 0.2
    wildfire_risk = 1 if (wildfire_prob + np.random.normal(0, 0.1)) > 0.5 else 0

    risk_scores = [
        ("Safe", 0.0),
        ("Flood", flood_prob if flood_risk else flood_prob * 0.5),
        ("Cyclone", cyclone_prob if cyclone_risk else cyclone_prob * 0.5),
        ("Wildfire/Landslide", wildfire_prob if wildfire_risk else wildfire_prob * 0.5),
    ]
    risk_scores.sort(key=lambda x: x[1], reverse=True)
    disaster_type = risk_scores[0][0] if risk_scores[0][1] > 0.35 else "Safe"

    rows.append({
        "district": district,
        "latitude": round(latitude, 4),
        "longitude": round(longitude, 4),
        "elevation_m": round(elevation_m, 1),
        "slope_deg": round(slope_deg, 2),
        "temperature_c": round(temperature_c, 1),
        "humidity_pct": round(humidity_pct, 1),
        "wind_speed_kmh": round(wind_speed_kmh, 1),
        "rainfall_7d_mm": round(rainfall_7d_mm, 1),
        "ndvi": round(ndvi, 3),
        "ndwi": round(ndwi, 3),
        "flood_risk": flood_risk,
        "cyclone_risk": cyclone_risk,
        "wildfire_risk": wildfire_risk,
        "disaster_type": disaster_type,
    })

df = pd.DataFrame(rows)
output_path = DEFAULT_OUTPUT
if len(sys.argv) > 1 and sys.argv[1].strip():
    output_path = sys.argv[1].strip().strip('"').strip("'")
out_dir = os.path.dirname(os.path.abspath(output_path))
if out_dir:
    os.makedirs(out_dir, exist_ok=True)
df.to_csv(output_path, index=False)
print(f"✅ Generated {len(df)} rows x {len(df.columns)} cols")
print(f"📂 Saved to: {os.path.abspath(output_path)}")

header_line = ",".join(df.columns)
print(f"\n🧾 CSV header:\n{header_line}")
print("\nDistrict counts:")
print(df["district"].value_counts().to_string())
print("\nDisaster type counts:")
print(df["disaster_type"].value_counts().to_string())
print("\nTarget class balance:")
print(f"flood_risk=1   : {df['flood_risk'].sum()} ({df['flood_risk'].mean()*100:.1f}%)")
print(f"cyclone_risk=1 : {df['cyclone_risk'].sum()} ({df['cyclone_risk'].mean()*100:.1f}%)")
print(f"wildfire_risk=1: {df['wildfire_risk'].sum()} ({df['wildfire_risk'].mean()*100:.1f}%)")

print("\n💡 How to use with the platform:")
print(f"   · Project folder: {os.path.dirname(os.path.abspath(__file__))}")
print( "   · Acceptable paste-in filenames: tamilnadu_disaster_data.csv, disaster_data.csv")
print( "   · OR Train on custom path:")
print(f'       python model_trainer.py "{os.path.abspath(output_path)}"')

