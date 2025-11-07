Table: runs (CSV in data/raw/runs.csv)

date (ISO, e.g., 2025-10-01)

distance_km (float)

duration_sec (int)

avg_hr (int, optional)

elev_gain_m (float, optional)

surface (categorical: road/trail/track/indoor)

workout_type (easy/tempo/interval/long/race)

sleep_h (float, optional)

rpe (1–10, optional)

shoes (string, optional)

Derived (placed in processed file by Day 1 script)

pace_sec_per_km = duration_sec / distance_km

is_race_like = workout_type in {"race","interval","tempo"}

target_5k_time_sec — label we’ll train on (see below)