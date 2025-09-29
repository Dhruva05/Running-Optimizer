import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Personal 5K Time", page_icon="🏃", layout="centered")

# ---------- Helpers ----------
def fmt_time(sec: float) -> str:
    m = int(sec // 60)
    s = int(round(sec - 60 * m))
    if s == 60:
        m += 1; s = 0
    return f"{m}:{s:02d}"

def clean_runs(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prefer Moving Time; numeric coercion; practical bounds; strict pause filter.
    """
    df = df.copy()
    # Keep only runs (if the column exists)
    if "Activity Type" in df.columns:
        df = df[df["Activity Type"] == "Run"].copy()

    # Choose time column (prefer Moving Time)
    time_col = "Moving Time" if "Moving Time" in df.columns else "Elapsed Time"
    if time_col not in df.columns:
        raise ValueError("Neither 'Moving Time' nor 'Elapsed Time' found in CSV.")

    # Coerce numeric
    df["TimeSec"]  = pd.to_numeric(df[time_col], errors="coerce")
    if "Distance" not in df.columns:
        raise ValueError("Column 'Distance' (km) not found in CSV.")
    df["Distance"] = pd.to_numeric(df["Distance"], errors="coerce")

    # Drop invalids
    df = df.dropna(subset=["TimeSec", "Distance"])
    df = df[(df["TimeSec"] > 0) & (df["Distance"] > 0)]

    # Practical bounds (remove walks/hikes/ultras)
    df = df[(df["Distance"] >= 2.0) & (df["Distance"] <= 25.0)]
    df = df[df["TimeSec"] <= 4 * 3600]

    # Stricter pause filter if both times exist (<= 5% stopped time)
    if {"Elapsed Time", "Moving Time"}.issubset(df.columns):
        et = pd.to_numeric(df["Elapsed Time"], errors="coerce")
        mt = pd.to_numeric(df["Moving Time"],  errors="coerce")
        pause_ratio = et / mt
        df = df[pause_ratio < 1.05]

    return df

def fit_personal_curve(df: pd.DataFrame):
    """
    Fit T = c * D^k via log–log linear regression with a residual trim.
    Returns: k, c, inlier_mask
    """
    D = df["Distance"].to_numpy()
    T = df["TimeSec"].to_numpy()
    logD, logT = np.log(D), np.log(T)

    # First pass
    k1, logc1 = np.polyfit(logD, logT, 1)
    resid = logT - (k1 * logD + logc1)

    # Trim extremes (keep ~96%)
    lo, hi = np.quantile(resid, [0.02, 0.98])
    mask = (resid >= lo) & (resid <= hi)

    # Refit on inliers
    k, logc = np.polyfit(logD[mask], logT[mask], 1)
    c = float(np.exp(logc))
    return float(k), c, mask

def personal_5k_summary(df: pd.DataFrame):
    """Compute single personal 5K time + diagnostics."""
    df = clean_runs(df)
    if df.empty:
        raise ValueError("No valid run records after cleaning.")

    k, c, mask = fit_personal_curve(df)
    D_target = 5.0

    # Curve estimate at exactly 5k
    T5_curve = c * (D_target ** k)

    # Per-activity 5k equivalents using personal k
    D = df["Distance"].to_numpy()
    T = df["TimeSec"].to_numpy()
    T5_each = T * (D_target / D) ** k

    # Robust aggregates
    T5_median = float(np.median(T5_each))
    if len(T5_each) >= 10:
        s = pd.Series(T5_each).sort_values()
        lo = int(0.10 * len(s))
        hi = int(0.90 * len(s))
        T5_trim = float(s.iloc[lo:hi].mean())
    else:
        T5_trim = float(np.mean(T5_each))

    # Optional: race-like (fastest 30%)
    top_n = max(5, int(0.30 * len(T5_each)))
    race_like_sec = float(pd.Series(T5_each).nsmallest(top_n).median()) if len(T5_each) else np.nan

    summary = {
        "k": float(k),
        "c": float(c),
        "curve_5k_sec": float(T5_curve),
        "median_5k_sec": float(T5_median),
        "trimmed_5k_sec": float(T5_trim),
        "race_like_5k_sec": float(race_like_sec),
        "n_runs": int(len(df)),
        "mask_inliers": mask,
    }
    return df, summary

# ---------- UI ----------
st.title("🏃 My Personal 5K Time")
st.caption("Fits a personal distance–time curve from my own runs and aggregates 5K-equivalents.")

# Always load your repo CSV (no upload UI)
CSV_PATH = "activities.csv"
try:
    df_raw = pd.read_csv(CSV_PATH)
except Exception as e:
    st.error(f"Couldn't load `{CSV_PATH}`: {e}")
    st.stop()

# Compute summary
try:
    runs, s = personal_5k_summary(df_raw)
except Exception as e:
    st.error(f"Error computing personal 5K time: {e}")
    st.stop()

# ---------- Headline metrics ----------
col1, col2, col3 = st.columns(3)
col1.metric("Personal 5K (median)",  fmt_time(s["median_5k_sec"]))
col2.metric("Trimmed mean 5K",       fmt_time(s["trimmed_5k_sec"]))
col3.metric("Curve at 5K",           fmt_time(s["curve_5k_sec"]))
st.caption(f"Runs after cleaning: {s['n_runs']} • k={s['k']:.3f}")

# ---------- Diagnostics ----------
from sklearn.metrics import r2_score, mean_absolute_error, median_absolute_error

with st.expander("Diagnostics"):
    # R² of log–log fit
    logD = np.log(runs["Distance"].to_numpy())
    logT = np.log(runs["TimeSec"].to_numpy())
    r2 = r2_score(logT, np.log(s["c"]) + s["k"] * logD)
    st.write(f"R² (log–log): **{r2:.3f}**")

    # Evaluate on ~5Ks with a tight band and remove obvious jogs
    band = runs[(runs["Distance"] >= 4.8) & (runs["Distance"] <= 5.2)].copy()
    pace = (band["TimeSec"] / 60.0) / band["Distance"]  # min/km
    band = band[pace < 9.0]

    if len(band) >= 3:
        y_true = (band["TimeSec"] / 60.0).to_numpy()
        y_pred = np.full_like(y_true, s["curve_5k_sec"] / 60.0)
        mae   = mean_absolute_error(y_true, y_pred)
        medae = median_absolute_error(y_true, y_pred)
        mape  = float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)
        st.write(f"~5K accuracy (n={len(band)}): MAE **{mae:.2f}** min · MedAE **{medae:.2f}** min · MAPE **{mape:.1f}%**")
    else:
        st.info("Not enough ~5Ks to score (need ≥3 after filtering).")

    # Optional race-like number
    if np.isfinite(s["race_like_5k_sec"]):
        st.write("Race-like 5K (fastest 30% of 5K-equivalents): **", fmt_time(s["race_like_5k_sec"]), "**")
