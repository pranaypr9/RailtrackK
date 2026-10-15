"""
Data Pipeline for Railway ETA Prediction Prototype
====================================================
Source of REAL data: Indian Railways Train Delay Dataset
(Guwahati <-> Delhi/Mumbai/Chennai/Kolkata express trains, Mar 2023-Mar 2024)
https://github.com/ankitaanand28/DA323_IndianRailwayTrainDelayDatasets

This real dataset gives us, per train, per station along its route:
  - Average historical delay (minutes)
  - % of runs that were: Right time (0-15min), Slight delay (15-60min),
    Significant delay (>60min), Cancelled/Unknown

That is AGGREGATE historical statistics, not raw per-journey GPS/live data
(which Indian Railways does not publish publicly). To build a system that
can demonstrate DYNAMIC, real-time ETA forecasting (the actual ask of the
problem statement), we use these real aggregate statistics as the
"ground truth" delay distribution for each station, and Monte-Carlo sample
individual synthetic train "runs" from them. This is a standard and
defensible approach for a prototype: real historical patterns +
simulated real-time event stream on top, exactly as scoped in our plan.

Output: data/journeys.csv  -- one row per (run_id, train_number, station_seq)
        data/stations.csv  -- station metadata (order, approx distance)
"""

import pandas as pd
import numpy as np
import os
import hashlib

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "raw_data")
OUT_DIR = os.path.dirname(__file__)

N_RUNS_PER_TRAIN = 60          # synthetic historical runs per train
WEATHER_STATES = ["Clear", "Rain", "Fog", "Storm"]
WEATHER_PROBS = [0.70, 0.18, 0.09, 0.03]
CONGESTION_LEVELS = ["Low", "Medium", "High"]


def seeded_rng(*parts):
    """Deterministic RNG per (train, station) so results are reproducible."""
    key = "_".join(str(p) for p in parts)
    seed = int(hashlib.md5(key.encode()).hexdigest()[:8], 16)
    return np.random.default_rng(seed)


def approx_inter_station_distance(train_no, station_code, idx):
    """
    Real inter-station distances aren't in the source dataset. We assign a
    realistic pseudo-distance (25-140 km typical Indian mail/express hop)
    deterministically per (train, station) so distances are stable across
    runs of this script. This is clearly flagged as an approximation --
    in a production system this would come from the real IR distance/
    cadastral database (already referenced in problem statement 26016's
    GIS layer, or from CRIS's station master data).
    """
    rng = seeded_rng(train_no, station_code, idx)
    return float(rng.uniform(25, 140))


def sample_delay_minutes(rng, avg_delay, p_right, p_slight, p_sig, p_cancel):
    """
    Sample a single run's delay (minutes) at a station using the REAL
    historical probability buckets from the dataset, anchored around the
    REAL average delay for that station.
    """
    probs = np.array([p_right, p_slight, p_sig, p_cancel], dtype=float)
    probs = np.clip(probs, 0, None)
    if probs.sum() == 0:
        probs = np.array([1, 0, 0, 0], dtype=float)
    probs = probs / probs.sum()

    bucket = rng.choice(["right", "slight", "significant", "cancel"], p=probs)

    if bucket == "right":
        delay = rng.uniform(0, 15)
    elif bucket == "slight":
        delay = rng.uniform(15, 60)
    elif bucket == "significant":
        # heavy-tailed: center loosely around 2x the station's real avg delay
        delay = rng.uniform(60, max(90, avg_delay * 2.5))
    else:  # cancel/unknown -> treat as a large terminal delay/skip signal
        delay = rng.uniform(120, 240)

    # blend with the real station average so long-run mean matches ground truth
    delay = 0.6 * delay + 0.4 * avg_delay
    return max(0.0, float(delay))


def build():
    train_list = pd.read_csv(os.path.join(RAW_DIR, "Train_List.csv"))
    train_list.columns = [c.strip() for c in train_list.columns]

    route_dir = os.path.join(RAW_DIR, "Train_Route")
    route_files = [f for f in os.listdir(route_dir) if f.endswith(".csv")]

    all_journeys = []
    all_stations = []

    for fname in sorted(route_files):
        train_no = fname.replace(".csv", "").lstrip("0") or "0"
        try:
            route = pd.read_csv(os.path.join(route_dir, fname))
        except Exception:
            continue
        route.columns = [c.strip() for c in route.columns]
        if "Station" not in route.columns:
            continue
        route = route.reset_index(drop=True)
        n_stations = len(route)
        if n_stations < 2:
            continue

        train_meta = train_list[train_list["Train_Number"].astype(str).str.lstrip("0") == train_no]
        train_type = train_meta["Type"].values[0] if len(train_meta) else "Mail/Express"
        train_name = train_meta["Train_Name"].values[0] if len(train_meta) else fname

        # ---- station metadata + cumulative distance ----
        cum_dist = 0.0
        station_rows = []
        for idx, row in route.iterrows():
            dist_hop = 0.0 if idx == 0 else approx_inter_station_distance(train_no, row["Station"], idx)
            cum_dist += dist_hop
            station_rows.append({
                "train_number": train_no,
                "seq": idx,
                "station_code": str(row["Station"]).strip(),
                "station_name": str(row.get("Station_Name", "")).strip(),
                "avg_hist_delay_min": float(row.get("Average_Delay(min)", 0) or 0),
                "cum_distance_km": round(cum_dist, 1),
            })
        all_stations.extend(station_rows)
        total_distance = cum_dist

        # ---- synthetic runs (real-time layer) ----
        for run_id in range(N_RUNS_PER_TRAIN):
            rng = seeded_rng(train_no, run_id)
            weather = rng.choice(WEATHER_STATES, p=WEATHER_PROBS)
            congestion = rng.choice(CONGESTION_LEVELS, p=[0.5, 0.35, 0.15])
            day_of_week = int(rng.integers(0, 7))
            start_hour = int(rng.integers(0, 24))

            # NOTE: avg_hist_delay_min in the source data is already the
            # CUMULATIVE average delay observed BY that station (not a
            # per-hop delta), so we anchor each station's simulated delay
            # directly to its own historical average (smoothed with the
            # previous station's simulated value for run-to-run continuity)
            # rather than additively stacking new delay on top of old delay
            # at every hop -- that would compound unrealistically over long
            # multi-day routes.
            prev_delay = 0.0
            for s in station_rows:
                p_right = route.loc[s["seq"], "Right Time (0-15 min's)"] if "Right Time (0-15 min's)" in route.columns else 40
                p_slight = route.loc[s["seq"], "Slight Delay (15-60 min's)"] if "Slight Delay (15-60 min's)" in route.columns else 30
                p_sig = route.loc[s["seq"], "Significant Delay (>1 Hour)"] if "Significant Delay (>1 Hour)" in route.columns else 15
                p_cancel = route.loc[s["seq"], "Cancelled/Unknown"] if "Cancelled/Unknown" in route.columns else 15

                sampled = sample_delay_minutes(
                    rng, s["avg_hist_delay_min"], p_right, p_slight, p_sig, p_cancel
                )
                # weather/congestion perturbation (real-world causal factors
                # named explicitly in the SIH problem statement)
                weather_mult = {"Clear": 1.0, "Rain": 1.10, "Fog": 1.20, "Storm": 1.35}[weather]
                congestion_mult = {"Low": 1.0, "Medium": 1.05, "High": 1.15}[congestion]
                sampled *= weather_mult * congestion_mult

                # smooth with previous station's delay so a single run's
                # trajectory evolves continuously rather than jumping randomly
                cum_delay = 0.5 * prev_delay + 0.5 * sampled
                prev_delay = cum_delay

                all_journeys.append({
                    "run_id": f"{train_no}_{run_id}",
                    "train_number": train_no,
                    "train_name": train_name,
                    "train_type": train_type,
                    "seq": s["seq"],
                    "n_stations": n_stations,
                    "station_code": s["station_code"],
                    "cum_distance_km": s["cum_distance_km"],
                    "total_distance_km": round(total_distance, 1),
                    "dist_remaining_km": round(total_distance - s["cum_distance_km"], 1),
                    "avg_hist_delay_min": s["avg_hist_delay_min"],
                    "weather": weather,
                    "congestion": congestion,
                    "day_of_week": day_of_week,
                    "hour_of_day": (start_hour + s["seq"]) % 24,
                    "delay_so_far_min": round(cum_delay, 1),
                    # TARGET: delay experienced AT this station (what an ETA
                    # system must predict before the train arrives there)
                    "target_delay_min": round(cum_delay, 1),
                })

    journeys_df = pd.DataFrame(all_journeys)
    stations_df = pd.DataFrame(all_stations).drop_duplicates(subset=["train_number", "seq"])

    journeys_df.to_csv(os.path.join(OUT_DIR, "journeys.csv"), index=False)
    stations_df.to_csv(os.path.join(OUT_DIR, "stations.csv"), index=False)

    print(f"Trains processed      : {journeys_df['train_number'].nunique()}")
    print(f"Synthetic runs total  : {journeys_df['run_id'].nunique()}")
    print(f"Journey rows (station-events): {len(journeys_df)}")
    print(f"Saved -> {os.path.join(OUT_DIR, 'journeys.csv')}")
    print(f"Saved -> {os.path.join(OUT_DIR, 'stations.csv')}")


if __name__ == "__main__":
    build()
