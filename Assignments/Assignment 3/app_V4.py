# app_V3.py — E.coli Risk Checker
# ------------------------------------------------------------
# - 3 classes: Safe / Caution / Danger
# - predict_proba + per-class thresholds re-map
# - date → year / month_number
# - robust preprocessing (one-hot + training-column alignment)
# - NO try/except in model loading and prediction branch
# ------------------------------------------------------------

import json
from pathlib import Path
from datetime import date as date_cls

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import joblib

# ---------------- Page config ----------------
st.set_page_config(page_title="E.coli Check", layout="wide")

# ---------------- Constants ------------------
CLASS_IDX_TO_NAME = {0: "Safe", 1: "Caution", 2: "Danger"}
CLASS_ORDER = [0, 1, 2]
DEFAULT_THRESHOLDS = {0: 0.33, 1: 0.33, 2: 0.33}

BASE_DIR = Path(__file__).parent
MODEL_FILE       = BASE_DIR / "Cat_SMOTE.joblib"
CSV_CANDIDATES   = ["06.1_selected_features.csv", "test_data.csv"]
THRESHOLDS_FILE  = BASE_DIR / "best_thresholds_Cat.json"

TRAINING_COL_CANDIDATES = [
    BASE_DIR / "training_columns.json",
    BASE_DIR / "training_columns.joblib",
    BASE_DIR / "feature_columns.json",
    BASE_DIR / "feature_columns.joblib",]

NON_FEATURE_COLUMNS = {"ecoli_risk", "date", "river", "location", "site_id","catchment_name", "swqaltitude"}

# ---------------- Helpers ----------------
def _unwrap_model(obj):
    if hasattr(obj, "predict"):
        return obj
    if isinstance(obj, (list, tuple)):
        for item in obj:
            if hasattr(item, "predict"):
                return item
    if isinstance(obj, dict):
        for k in ["model", "estimator", "pipeline", "clf", "best_estimator_"]:
            if k in obj and hasattr(obj[k], "predict"):
                return obj[k]
        if "grid" in obj and hasattr(obj["grid"], "best_estimator_"):
            return obj["grid"].best_estimator_
    if hasattr(obj, "best_estimator_"):
        return obj.best_estimator_
    st.error("Loaded object is not a sklearn estimator and couldn't be unwrapped.")
    st.stop()

def _load_thresholds() -> dict:
    if not THRESHOLDS_FILE.exists():
        return DEFAULT_THRESHOLDS
    data = json.loads(THRESHOLDS_FILE.read_text(encoding="utf-8"))
    if isinstance(data, dict) and all(k in data for k in ["Safe", "Caution", "Danger"]):
        return {0: float(data["Safe"]), 1: float(data["Caution"]), 2: float(data["Danger"])}
    return {int(k): float(v) for k, v in data.items()}

def _reclassify_with_thresholds(proba: np.ndarray, thresholds: dict) -> np.ndarray:
    out = []
    for row in proba:
        cands = [(row[i], i) for i in CLASS_ORDER if row[i] >= thresholds.get(i, 0.0)]
        cands.sort(reverse=True, key=lambda x: x[0])
        out.append(cands[0][1] if cands else int(np.argmax(row)))
    return np.array(out, dtype=int)

def _month_to_number(name: str) -> float:
    try:
        return float(pd.to_datetime(name, format="%B").month)
    except Exception:
        return np.nan

def _load_training_columns():
    for fp in TRAINING_COL_CANDIDATES:
        if fp.exists():
            if fp.suffix == ".json":
                cols = json.loads(fp.read_text(encoding="utf-8"))
            else:
                cols = joblib.load(fp)
            if isinstance(cols, dict) and "feature_columns" in cols:
                cols = cols["feature_columns"]
            if isinstance(cols, (list, tuple)):
                return list(cols)

    if hasattr(model, "get_booster"):
        booster = model.get_booster()
        if getattr(booster, "feature_names", None):
            return list(booster.feature_names)
    return None

def _preprocess_single_row(df_one: pd.DataFrame, training_cols: list | None):
    x = df_one.copy()

    if "date" in x.columns:
        x["date"] = pd.to_datetime(x["date"])
        x["year"] = x["date"].dt.year
        if "month" in x.columns and x["month"].dtype == object:
            x["month_number"] = x["month"].astype(str).map(_month_to_number)
        else:
            x["month_number"] = x["date"].dt.month
    else:
        x["year"] = pd.NA
        x["month_number"] = pd.NA

    if "upstream_flag" in x.columns:
        x["upstream_flag"] = pd.to_numeric(x["upstream_flag"], errors="coerce").fillna(0).astype(int)
    if "wider_canterbury" in x.columns:
        x["wider_canterbury"] = x["wider_canterbury"].astype("category")

    x = x.drop(columns=[c for c in x.columns if c in NON_FEATURE_COLUMNS], errors="ignore")

    for c in list(x.columns):
        if pd.api.types.is_datetime64_any_dtype(x[c]):
            x = x.drop(columns=[c])

    x_encoded = x.copy()

    for c in x_encoded.columns:
        if not pd.api.types.is_numeric_dtype(x_encoded[c]):
            x_encoded[c] = x_encoded[c].astype(str)
    

    if training_cols is None:
        for c in ["year", "month_number"]:
            if c in x_encoded.columns:
                x_encoded[c] = pd.to_numeric(x_encoded[c], errors="coerce").fillna(0)
        for c in x_encoded.columns:
            if x_encoded[c].dtype == "bool":
                x_encoded[c] = x_encoded[c].astype(int)
            elif not pd.api.types.is_numeric_dtype(x_encoded[c]):
                x_encoded[c] = x_encoded[c].astype("category").cat.codes
        return x_encoded, list(x_encoded.columns)
    else:
        for col in training_cols:
            if col not in x_encoded.columns:
                x_encoded[col] = 0
        x_encoded = x_encoded[training_cols]
        for c in x_encoded.columns:
            if x_encoded[c].dtype == "bool":
                x_encoded[c] = x_encoded[c].astype(int)
            elif not pd.api.types.is_numeric_dtype(x_encoded[c]):
                x_encoded[c] = pd.to_numeric(x_encoded[c], errors="coerce").fillna(0)
        return x_encoded, training_cols

@st.cache_resource
def load_model():
    if not MODEL_FILE.exists():
        st.error(f"Model file not found: {MODEL_FILE}")
        st.stop()
    raw = joblib.load(MODEL_FILE)
    est = _unwrap_model(raw)
    return est

@st.cache_data
def load_sites_df():
    for name in CSV_CANDIDATES:
        fp = BASE_DIR / name
        if fp.exists():
            return pd.read_csv(fp)
    st.error(f"Sites CSV not found. Tried: {', '.join(CSV_CANDIDATES)} in {BASE_DIR}")
    st.stop()

@st.cache_data
def load_thresholds_cached():
    return _load_thresholds()

# ---------------- Load assets ----------------
model = load_model()
sites_df = load_sites_df()
best_thresholds = load_thresholds_cached()
training_columns = _load_training_columns()  

# ---------------- UI ----------------
st.title("E.coli Check")

left, right = st.columns([1, 1])

has_river = "river" in sites_df.columns
has_loc   = "location" in sites_df.columns

with left:
    if has_river:
        rivers = sorted(sites_df["river"].dropna().astype(str).unique().tolist())
        selected_river = st.selectbox("Select a river:", rivers, index=0 if rivers else None)
        df_r = sites_df[sites_df["river"].astype(str) == str(selected_river)]
    else:
        selected_river = st.text_input("River (CSV does not have this column, manual input):", value="Avon")
        df_r = sites_df.copy()

    if has_loc:
        locations = sorted(df_r["location"].dropna().astype(str).unique().tolist())
        selected_location = st.selectbox("Select a location:", locations, index=0 if locations else None)
        df_site = df_r[df_r["location"].astype(str) == str(selected_location)]
    else:
        selected_location = st.text_input("Location (CSV does not have this column, manual input):", value="Loop Rd")
        df_site = df_r.copy()

with right:
    rain  = st.number_input("Rain today (mm):", min_value=0.0, step=0.1, value=0.0)
    rain3 = st.number_input("Rain two days ago (mm):",min_value=0.0, step=0.1, value=0.0)
    rain4 = st.number_input("Rain three days ago (mm):", min_value=0.0, step=0.1, value=0.0)

st.write("You selected:", selected_river, "—", selected_location)

# ---------------- Build X then Predict ----------------
if df_site.empty:
    st.error("No baseline row found for the selected site.")
    st.stop()

row = df_site.iloc[[0]].copy()
row["rainfall"]= float(rain)
row["rainfall_lagged_2"]= float(rain3)
row["rainfall_lagged_3"]= float(rain4)

X, used_cols = _preprocess_single_row(row, training_columns)

expected_cols = model.feature_names_

for c in expected_cols:
    if c not in X.columns:
        X[c] = 0

X = X[expected_cols]

if st.button("Predict E. coli Risk"):
    has_proba = hasattr(model, "predict_proba")
    has_decfn = hasattr(model, "decision_function")

    if has_proba:
        proba = model.predict_proba(X)
    elif has_decfn:
        scores = model.decision_function(X)
        exp = np.exp(scores - np.max(scores, axis=1, keepdims=True))
        proba = exp / exp.sum(axis=1, keepdims=True)
    else:
        pred_raw = model.predict(X)
        label = (CLASS_IDX_TO_NAME[int(pred_raw[0])]
                 if isinstance(pred_raw[0], (int, np.integer))
                 else str(pred_raw[0]))
        st.warning("Model has no probability output; showing direct prediction.")
        st.markdown(f"Predicted E. coli risk: **{label}**")
        st.stop()

    mapped_idx = _reclassify_with_thresholds(proba, best_thresholds)
    final_idx  = int(mapped_idx[0])
    final_label= CLASS_IDX_TO_NAME.get(final_idx, str(final_idx))
    st.markdown(f"Predicted E. coli risk: **{final_label}**")

    proba_series = pd.Series(proba[0, CLASS_ORDER],
                             index=[CLASS_IDX_TO_NAME[i] for i in CLASS_ORDER])
    fig, ax = plt.subplots(figsize=(5, 3))
    proba_series.plot(kind="bar", ax=ax)
    ax.set_ylim(0, 1); ax.set_ylabel("Probability"); ax.set_title("Class probabilities")
    ax.grid(True, axis="y", linestyle="--", alpha=0.5) 
    st.pyplot(fig)

    with st.expander("Show thresholds & feature columns used"):
        th_df = pd.DataFrame({
            "class": [CLASS_IDX_TO_NAME[i] for i in CLASS_ORDER],
            "threshold": [best_thresholds.get(i, np.nan) for i in CLASS_ORDER],
            "probability": proba_series.values})
        st.dataframe(th_df, use_container_width=True)
        st.caption("19 features used")
