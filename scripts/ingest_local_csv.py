# scripts/ingest_local_csv.py
from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW = PROJECT_ROOT / "data" / "raw" / "runs.csv"
OUT = PROJECT_ROOT / "data" / "processed" / "runs_clean.csv"

# Map raw → standardized (adjust if your headers differ)
RAW_TO_STD = {
    "Activity Date": "date",
    "Distance.1": "distance_m",      # meters
    "Moving Time": "duration_sec",   # seconds
    # Optional:
    # "Average Heart Rate": "avg_hr",
    # "Elevation Gain": "elev_gain_m",
}

MIN_DISTANCE_KM = 0.5  # drop <500 m (tweak if you want to keep very short warmups)
MIN_DURATION_S  = 60   # drop <1 min

if __name__ == "__main__":
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df_raw = pd.read_csv(RAW)

    # rename → standardize
    df = df_raw.rename(columns=RAW_TO_STD).copy()

    # distance to km (if provided in meters)
    if "distance_m" in df.columns and "distance_km" not in df.columns:
        df["distance_km"] = pd.to_numeric(df["distance_m"], errors="coerce") / 1000.0

    # Parse numerics/dates
    df["date"] = pd.to_datetime(df.get("date"), errors="coerce")
    df["duration_sec"] = pd.to_numeric(df.get("duration_sec"), errors="coerce")
    df["distance_km"] = pd.to_numeric(df.get("distance_km"), errors="coerce")

    # ---- Drop junk rows here ----
    n0 = len(df)
    m_nan      = df["distance_km"].isna() | df["duration_sec"].isna()
    m_nonpos   = (df["distance_km"] <= 0) | (df["duration_sec"] <= 0)
    m_too_short= (df["distance_km"] < MIN_DISTANCE_KM) | (df["duration_sec"] < MIN_DURATION_S)
    bad = m_nan | m_nonpos | m_too_short
    print(f"ingest: dropping {int(bad.sum())}/{n0} rows "
          f"(nan={int(m_nan.sum())}, nonpos={int(m_nonpos.sum())}, short={int(m_too_short.sum())})")
    df = df.loc[~bad].copy()

    # Derived pace
    df["pace_sec_per_km"] = df["duration_sec"] / df["distance_km"].clip(lower=1e-6)

    # Keep minimal columns (+ optional if present)
    keep = ["date", "distance_km", "duration_sec", "pace_sec_per_km"]
    for opt in ["avg_hr", "elev_gain_m"]:
        if opt in df.columns:
            keep.append(opt)
    df = df[keep]

    df.to_csv(OUT, index=False)
    print(f"Wrote {OUT} with {len(df)} rows")
