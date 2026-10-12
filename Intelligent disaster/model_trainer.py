import os
import sys
import pickle
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

try:
    import xgboost as xgb
except Exception:  # pragma: no cover
    xgb = None
try:
    import lightgbm as lgb
except Exception:  # pragma: no cover
    lgb = None
try:
    from catboost import CatBoostClassifier
except Exception:  # pragma: no cover
    CatBoostClassifier = None

warnings.filterwarnings("ignore")

RANDOM_STATE = 42

DEFAULT_FEATURE_COLS = [
    "latitude", "longitude", "elevation_m", "slope_deg",
    "temperature_c", "humidity_pct", "wind_speed_kmh",
    "rainfall_7d_mm", "ndvi", "ndwi",
]
DEFAULT_TARGET_COL = "flood_risk"
DEFAULT_CSV_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "tamilnadu_disaster_data.csv",
)


def detect_columns(df: pd.DataFrame, target_col_hint=None):
    """Auto-detect feature columns, district column and target from any CSV."""
    cols_lower = {str(c).lower().strip(): c for c in df.columns}

    def find(options):
        for opt in options:
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

    numeric_cols = list(df.select_dtypes(include=[np.number]).columns)

    feature_cols = [
        c for c in [
            lat_col, lon_col, elev_col, slope_col, temp_col, hum_col,
            wind_col, rain_col, ndvi_col, ndwi_col,
        ] if c is not None
    ]
    required = 3
    if len(feature_cols) < required:
        fallback = [c for c in DEFAULT_FEATURE_COLS if c in df.columns]
        if len(fallback) >= required:
            feature_cols = fallback
        else:
            feature_cols = [c for c in numeric_cols if c not in cols_lower.values()
                            or True]
            if target_col_hint in feature_cols:
                feature_cols = [c for c in feature_cols if c != target_col_hint]
            blacklist = ["flood_risk", "cyclone_risk", "wildfire_risk",
                         "disaster_type", "risk", "label", "target"]
            feature_cols = [c for c in numeric_cols
                            if str(c).lower() not in blacklist]

    target_col = target_col_hint
    if target_col is None or target_col not in df.columns:
        for t in [
            DEFAULT_TARGET_COL, "flood_risk", "risk", "label", "target",
            "class", "y", "hazard", "hazard_risk",
        ]:
            if t in df.columns:
                target_col = t
                break
        if target_col is None and len(numeric_cols) > 0:
            last_numeric = [c for c in numeric_cols if c not in feature_cols]
            target_col = last_numeric[-1] if last_numeric else numeric_cols[-1]

    district_col_final = district_col
    if district_col_final is None:
        for c in df.columns:
            if df[c].dtype == object or str(df[c].dtype).startswith("category"):
                if df[c].nunique() <= df.shape[0] * 0.6 and df[c].nunique() >= 2:
                    district_col_final = c
                    break
        if district_col_final is None:
            district_col_final = "district"
            if district_col_final not in df.columns:
                df = df.copy()
                df[district_col_final] = np.tile(
                    ["Area_" + str(i % 15) for i in range(15)],
                    int(np.ceil(len(df) / 15)),
                )[: len(df)]

    if lat_col is None:
        df = df.copy()
        if "latitude" not in df.columns:
            df["latitude"] = np.random.uniform(8.07, 13.50, size=len(df))
        if "longitude" not in df.columns:
            df["longitude"] = np.random.uniform(76.23, 80.34, size=len(df))
        feature_cols = [c for c in feature_cols
                        if c not in (lat_col, lon_col)] + ["latitude", "longitude"]

    return df, feature_cols, target_col, district_col_final


def load_and_split_data(csv_path=None, target_col=None):
    if csv_path is None:
        csv_path = DEFAULT_CSV_PATH
        alt = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "disaster_data.csv",
        )
        if not os.path.exists(csv_path) and os.path.exists(alt):
            csv_path = alt
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"CSV not found: {csv_path}\n"
            "Please paste your CSV file inside the project folder "
            f"(e.g. next to this script), then run:\n"
            f"    python model_trainer.py \"{csv_path}\"\n"
            "Or specify its full path as first argument."
        )

    print(f"[1/4] Loading CSV: {csv_path}")
    df_raw = pd.read_csv(csv_path)
    print(f"      Rows={len(df_raw)}, Columns={len(df_raw.columns)}")
    print(f"      Columns: {list(df_raw.columns)}")

    df, feature_cols, detected_target, district_col = detect_columns(
        df_raw, target_col_hint=target_col
    )
    if target_col is None:
        target_col = detected_target
    print(f"[2/4] Auto-detected columns:")
    print(f"      District col : {district_col}")
    print(f"      Feature cols : {feature_cols}")
    print(f"      Target col   : {target_col}")

    df[feature_cols] = df[feature_cols].apply(pd.to_numeric, errors="coerce")
    df = df.dropna(subset=feature_cols + [target_col]).reset_index(drop=True)
    df[target_col] = pd.to_numeric(df[target_col], errors="coerce").fillna(0).astype(int)
    print(f"      Clean rows after dropna: {len(df)}")

    X = df[feature_cols].copy()
    y = df[target_col].copy()
    if y.nunique(dropna=False) < 2:
        raise ValueError(
            f"Target column '{target_col}' has only {y.nunique(dropna=False)} class(es). "
            "Need at least 2 classes (0/1) for binary classification."
        )
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y
        )
    except ValueError:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=RANDOM_STATE
        )
    return X_train, X_test, y_train, y_test, df, feature_cols, target_col, district_col


def build_models():
    models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=10, min_samples_leaf=2,
            random_state=RANDOM_STATE, n_jobs=-1
        ),
    }
    if xgb is not None:
        models["XGBoost"] = xgb.XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.08,
            subsample=0.9, colsample_bytree=0.8,
            use_label_encoder=False, eval_metric="logloss",
            random_state=RANDOM_STATE, n_jobs=-1
        )
    if lgb is not None:
        models["LightGBM"] = lgb.LGBMClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.08,
            subsample=0.9, colsample_bytree=0.8,
            random_state=RANDOM_STATE, n_jobs=-1, verbose=-1
        )
    if CatBoostClassifier is not None:
        models["CatBoost"] = CatBoostClassifier(
            iterations=200, depth=6, learning_rate=0.08,
            l2_leaf_reg=3, random_seed=RANDOM_STATE, verbose=0
        )
    return models


def evaluate_model(name, model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    else:
        y_prob = y_pred.astype(float)
    return {
        "Model": name,
        "Accuracy": round(accuracy_score(y_test, y_pred), 4),
        "Precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "Recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
        "F1-Score": round(f1_score(y_test, y_pred, zero_division=0), 4),
        "ROC-AUC": round(roc_auc_score(y_test, y_prob), 4),
    }


def get_feature_importance(best_model, feature_names):
    if hasattr(best_model, "feature_importances_"):
        importances = best_model.feature_importances_
    elif hasattr(best_model, "get_feature_importance"):
        importances = best_model.get_feature_importance()
    else:
        importances = np.ones(len(feature_names)) / len(feature_names)
    return pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances,
    }).sort_values("Importance", ascending=False).reset_index(drop=True)


def main(csv_path=None, target_col=None, output_dir=None):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    if output_dir is None:
        output_dir = os.path.join(base_dir, "models")
    os.makedirs(output_dir, exist_ok=True)

    split = load_and_split_data(csv_path=csv_path, target_col=target_col)
    X_train, X_test, y_train, y_test, full_df, feature_cols, target_col, district_col = split

    print(f"[3/4] Training classifiers on '{target_col}' using {len(feature_cols)} features...")
    print(f"      Train: {X_train.shape[0]} rows | Test: {X_test.shape[0]} rows")
    print(f"      Target positives in test: {int(y_test.sum())}/{len(y_test)} "
          f"({y_test.mean()*100:.1f}%)\n")

    scaler = StandardScaler()
    try:
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
    except Exception:
        scaler.fit(X_train.head(1))
        X_train_scaled = X_train.values
        X_test_scaled = X_test.values

    models = build_models()
    results = []
    trained_models = {}

    for name, model in models.items():
        print(f"  · Training {name} ...")
        if name == "Random Forest":
            res = evaluate_model(name, model, X_train_scaled, X_test_scaled, y_train, y_test)
            trained_models[name] = (model, scaler, "scaled")
        else:
            res = evaluate_model(name, model, X_train, X_test, y_train, y_test)
            trained_models[name] = (model, scaler, "raw")
        results.append(res)
        print(f"      -> Accuracy={res['Accuracy']:.4f}  "
              f"F1={res['F1-Score']:.4f}  ROC-AUC={res['ROC-AUC']:.4f}\n")

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values("F1-Score", ascending=False).reset_index(drop=True)

    print("=" * 70)
    print("MODEL BENCHMARK (sorted by F1-Score)")
    print("=" * 70)
    print(results_df.to_string(index=False))
    print()

    best_name = results_df.iloc[0]["Model"]
    best_model, best_scaler, scaler_mode = trained_models[best_name]
    print(f"BEST MODEL: {best_name}")

    feature_imp = get_feature_importance(best_model, feature_cols)
    print("\nTop 5 Feature Importances:")
    print(feature_imp.head().to_string(index=False))

    best_artifact = {
        "model": best_model,
        "scaler": best_scaler,
        "scaler_mode": scaler_mode,
        "feature_cols": feature_cols,
        "target_col": target_col,
        "district_col": district_col,
        "source_csv": os.path.abspath(csv_path) if csv_path else DEFAULT_CSV_PATH,
        "best_model_name": best_name,
        "benchmarks": results_df,
        "feature_importance": feature_imp,
        "X_train_sample": X_train.head(100).copy(),
        "district_values": full_df[district_col].astype(str).unique().tolist()
        if district_col in full_df.columns else [],
    }

    with open(os.path.join(output_dir, "best_model.pkl"), "wb") as f:
        pickle.dump(best_artifact, f)

    results_df.to_csv(os.path.join(output_dir, "benchmarks.csv"), index=False)
    feature_imp.to_csv(os.path.join(output_dir, "feature_importance.csv"), index=False)

    print("\n[4/4] Saved artifacts to", os.path.abspath(output_dir), "/")
    print("      · best_model.pkl")
    print("      · benchmarks.csv")
    print("      · feature_importance.csv")
    return results_df


if __name__ == "__main__":
    csv_arg = None
    target_arg = None
    if len(sys.argv) > 1 and sys.argv[1].strip():
        csv_arg = sys.argv[1].strip().strip('"').strip("'")
    if len(sys.argv) > 2 and sys.argv[2].strip():
        target_arg = sys.argv[2].strip().strip('"').strip("'")
    try:
        main(csv_path=csv_arg, target_col=target_arg)
    except FileNotFoundError as fnfe:
        print("\n" + "=" * 70)
        print("💡 WHERE TO PASTE YOUR CSV:")
        print("=" * 70)
        print(f"Project folder: {os.path.dirname(os.path.abspath(__file__))}")
        print()
        print("Option A — Drop in your CSV in the project folder:")
        print('  Rename it either "tamilnadu_disaster_data.csv" (default)')
        print('  OR "disaster_data.csv" (auto-detected fallback)')
        print()
        print("Option B — Keep the CSV anywhere and pass its path:")
        print('  python model_trainer.py "D:\\path\\to\\your_file.csv"')
        print()
        print("Option C — Override default target column name:")
        print('  python model_trainer.py "your.csv" flood_risk')
        print("=" * 70)
        print("\nError details:", fnfe)
        sys.exit(1)
    except Exception as exc:
        print("\n[FAIL]", type(exc).__name__, ":", exc)
        sys.exit(2)
