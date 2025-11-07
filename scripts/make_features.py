# scripts/make_features.py
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INP = PROJECT_ROOT / "data" / "processed" / "runs_clean.csv"
OUT = PROJECT_ROOT / "data" / "processed" / "features.csv"

# --- thresholds (tune as you like) ---
MIN_DISTANCE_KM = 0.5      # drop super short misc logs
MIN_DURATION_S  = 60       # drop < 1 minute
PACE_MIN = 180             # 3:00 /km (too fast -> likely wrong units)
PACE_MAX = 1200            # 20:00 /km (too slow -> likely wrong units)

def add_basic_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["pace_sec_per_km"] = df["duration_sec"] / df["distance_km"].clip(lower=1e-6)
    df["is_race_like"] = False
    if "workout_type" in df.columns:
        df["is_race_like"] = (
            df["workout_type"].astype(str).str.lower().isin(["race","tempo","interval"])
        )
    return df

def quality_filter(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows with NaNs/zeros/extremes and print a brief summary."""
    n0 = len(df)
    reasons = {}

    # basic validity
    m_nan = df["distance_km"].isna() | df["duration_sec"].isna()
    reasons["nan distance/duration"] = int(m_nan.sum())

    m_nonpos = (df["distance_km"] <= 0) | (df["duration_sec"] <= 0)
    reasons["non-positive distance/duration"] = int(m_nonpos.sum())

    # minimal thresholds
    m_short = (df["distance_km"] < MIN_DISTANCE_KM) | (df["duration_sec"] < MIN_DURATION_S)
    reasons["below min dist/duration"] = int(m_short.sum())

    # pace plausibility (needs pace column)
    pace = df["duration_sec"] / df["distance_km"].clip(lower=1e-6)
    m_pace = (pace < PACE_MIN) | (pace > PACE_MAX) | ~np.isfinite(pace)
    reasons["implausible pace"] = int(m_pace.sum())

    # combine all reasons
    bad = m_nan | m_nonpos | m_short | m_pace
    df = df.loc[~bad].copy()

    kept = len(df)
    dropped = n0 - kept
    print(f"Quality filter: kept {kept}/{n0} rows, dropped {dropped}. Breakdown:")
    for k,v in reasons.items():
        print(f"  - {k}: {v}")
    return df

def target_5k_seconds(df: pd.DataFrame, method: str = "riegel") -> pd.Series:
    if method == "pace":
        return 5.0 * df["pace_sec_per_km"]
    t1 = df["duration_sec"].to_numpy(float)
    d1 = np.clip(df["distance_km"].to_numpy(float), 1e-6, None)
    tgt = t1 * (5.0 / d1) ** 1.06
    return pd.Series(tgt, index=df.index, name="target_5k_time_sec")

if __name__ == "__main__": 
    print("INP =", INP); print("OUT =", OUT)
    OUT.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INP)
    for c in ["duration_sec","distance_km"]:
        if c not in df.columns:
            raise SystemExit(f"Missing column: {c}")

    # add features -> filter -> compute target
    df = add_basic_features(df)
    df = quality_filter(df)
    df["target_5k_time_sec"] = target_5k_seconds(df, method="riegel")

    df.to_csv(OUT, index=False)
    print(f"Wrote {OUT} with {len(df)} rows")
