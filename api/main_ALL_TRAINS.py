"""
RailConnect FastAPI backend
- ALL trains can be searched through /trains and /trains/catalogue
- ML-supported trains use the LSTM ETA model
- Any train available in the RailRadar catalogue can be opened for live tracking
- Non-ML catalogue trains use the existing LSTM with a generalized RailRadar route feature mapping for prototype ETA forecasting

Run:
    python -m uvicorn api.main:app --reload --port 8001
"""

import os
import json
from typing import Optional
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import torch
from fastapi import FastAPI, HTTPException, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from api.live import get_live_train_status
from api.weather import get_weather_status
from models.model import ETALSTM, FEATURE_COLS


# ===============================================================
# PATHS
# ===============================================================

BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")
DASHBOARD_DIR = os.path.join(BASE_DIR, "dashboard")

TRAINS_FILE = os.path.join(DATA_DIR, "trains.csv")
RAILRADAR_TRAINS_FILE = os.path.join(DATA_DIR, "railradar_trains.csv")
RAILRADAR_PRS_TRAINS_FILE = os.path.join(DATA_DIR, "railradar_prs_trains.csv")
STATIONS_FILE = os.path.join(DATA_DIR, "stations.csv")


# ===============================================================
# HELPERS
# ===============================================================

def normalize_train_number(value) -> str:
    """Keep train numbers consistent across CSV files and API calls."""
    if pd.isna(value):
        return ""
    return (
        str(value)
        .strip()
        .replace(".0", "")
    )


def find_column(df, candidates):
    """Return the first matching column name from candidates."""
    lookup = {str(c).strip().lower(): c for c in df.columns}

    for candidate in candidates:
        key = candidate.strip().lower()
        if key in lookup:
            return lookup[key]

    return None


# ===============================================================
# LOAD TRAIN DATA
# ===============================================================

# ML-supported train list
try:
    trains_df = pd.read_csv(
        TRAINS_FILE,
        dtype={"train_number": str}
    )

    if "train_number" in trains_df.columns:
        trains_df["train_number"] = (
            trains_df["train_number"]
            .map(normalize_train_number)
        )

except Exception as e:
    print("WARNING: Could not load trains.csv:", e)
    trains_df = pd.DataFrame(
        columns=["train_number", "train_name"]
    )


# Complete train catalogue
try:
    railradar_trains_df = pd.read_csv(
        RAILRADAR_TRAINS_FILE,
        dtype={"train_number": str}
    )

    number_col = find_column(
        railradar_trains_df,
        ["train_number", "train_no", "trainnumber", "number"]
    )

    if number_col and number_col != "train_number":
        railradar_trains_df = railradar_trains_df.rename(
            columns={number_col: "train_number"}
        )

    if "train_number" in railradar_trains_df.columns:
        railradar_trains_df["train_number"] = (
            railradar_trains_df["train_number"]
            .map(normalize_train_number)
        )

    name_col = find_column(
        railradar_trains_df,
        ["train_name", "trainname", "name"]
    )

    if name_col and name_col != "train_name":
        railradar_trains_df = railradar_trains_df.rename(
            columns={name_col: "train_name"}
        )

    if "train_name" not in railradar_trains_df.columns:
        railradar_trains_df["train_name"] = ""

    railradar_trains_df["train_name"] = (
        railradar_trains_df["train_name"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    railradar_trains_df = (
        railradar_trains_df[
            ["train_number", "train_name"]
        ]
        .drop_duplicates(subset=["train_number"])
    )

    print(
        f"Loaded {len(railradar_trains_df)} trains "
        "from RailRadar catalogue."
    )

except Exception as e:
    print("WARNING: Could not load railradar_trains.csv:", e)

    railradar_trains_df = pd.DataFrame(
        columns=["train_number", "train_name"]
    )


try:
    prs_trains_df = pd.read_csv(RAILRADAR_PRS_TRAINS_FILE, dtype={"train_number": str})
    prs_trains_df["train_number"] = prs_trains_df["train_number"].map(normalize_train_number)
except Exception:
    prs_trains_df = pd.DataFrame(
        columns=["train_number", "train_name", "source_code", "destination_code"]
    )


# ===============================================================
# LOAD STATION / MODEL DATA
# ===============================================================

stations_df = pd.read_csv(
    STATIONS_FILE,
    dtype={"train_number": str}
)

stations_df["train_number"] = (
    stations_df["train_number"]
    .map(normalize_train_number)
)


with open(
    os.path.join(MODEL_DIR, "norm_stats.json"),
    encoding="utf-8"
) as f:
    STATS = json.load(f)


with open(
    os.path.join(MODEL_DIR, "eval_report.json"),
    encoding="utf-8"
) as f:
    EVAL_REPORT = json.load(f)


model = ETALSTM(
    n_features=len(FEATURE_COLS)
)

model.load_state_dict(
    torch.load(
        os.path.join(MODEL_DIR, "eta_lstm.pt"),
        map_location="cpu"
    )
)

model.eval()


# ===============================================================
# FASTAPI
# ===============================================================

app = FastAPI(
    title="RailConnect - Dynamic Railway ETA Prediction API",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)


# ===============================================================
# CONSTANTS / STATE
# ===============================================================

WEATHER_STATES = [
    "Clear",
    "Rain",
    "Fog",
    "Storm"
]

CONGESTION_LEVELS = [
    "Low",
    "Medium",
    "High"
]

# In-memory live run store.
# Production version can use Redis/PostgreSQL.
LIVE_RUNS = {}


# ===============================================================
# MODEL FUNCTIONS
# ===============================================================

def norm(col, val):
    s = STATS[col]

    std = s["std"]
    if std == 0:
        std = 1

    return (val - s["mean"]) / std


def denorm(col, val):
    s = STATS[col]

    return (
        val * s["std"]
        + s["mean"]
    )


def _number(value, default=0.0):
    """Safely convert optional API values to numbers."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _route_value(station, *keys):
    """Get the first meaningful value from differently-shaped route data."""
    for key in keys:
        value = station.get(key)
        if value not in (None, ""):
            return value
    return None


def _live_eta_datetime(value, now=None):
    """Convert a provider ETA into an aware datetime when possible."""
    if value in (None, ""):
        return None

    now = now or datetime.now().astimezone()
    text = str(value).strip()

    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        return parsed.replace(tzinfo=now.tzinfo) if parsed.tzinfo is None else parsed
    except ValueError:
        pass

    for time_format in ("%H:%M", "%H:%M:%S", "%I:%M %p", "%I:%M%p"):
        try:
            parsed_time = datetime.strptime(text.upper(), time_format).time()
            candidate = now.replace(
                hour=parsed_time.hour,
                minute=parsed_time.minute,
                second=parsed_time.second,
                microsecond=0,
            )
            # A clock-only ETA earlier than now belongs to the following day.
            if candidate < now - timedelta(minutes=5):
                candidate += timedelta(days=1)
            return candidate
        except ValueError:
            continue

    return None


def _journey_date(value):
    """Validate the selected service's origin-date query parameter."""
    if not value:
        return None
    try:
        return datetime.strptime(value, "%Y-%m-%d").date().isoformat()
    except (TypeError, ValueError):
        raise HTTPException(
            status_code=422,
            detail="Journey date must use YYYY-MM-DD.",
        )


def _stop_code(stop):
    return str(_route_value(stop, "stationCode", "station_code", "code") or "").strip().upper()


def _stop_sequence(stop, default=0):
    try:
        return int(_route_value(stop, "sequence", "seq") or default)
    except (TypeError, ValueError):
        return default


def _halts_from_route(route):
    """Fallback when a provider does not honour the haltsOnly parameter."""
    return [
        stop for stop in route
        if stop.get("isHalt") is True or stop.get("isStop") is True
        or str(stop.get("stopType", "")).lower() in {"halt", "stop", "origin", "destination"}
    ]


def _rate_limit_fallback(train_number, journey_date, original_error):
    """Keep the demo usable when the upstream live API returns HTTP 429.

    This deliberately labels the response as a local fallback; it never
    presents invented telemetry as RailRadar live data.
    """
    local_route = (
        stations_df[stations_df["train_number"] == train_number]
        .sort_values("seq")
        .copy()
    )
    if not local_route.empty:
        route = [
            {
                "sequence": int(row.seq),
                "stationCode": str(row.station_code),
                "stationName": str(row.station_name).strip(),
                "distanceFromOriginKm": float(row.cum_distance_km),
                "isHalt": True,
            }
            for row in local_route.itertuples()
        ]
    else:
        prs = prs_trains_df[prs_trains_df["train_number"] == train_number]
        origin = str(prs.iloc[0].get("source_code", "Origin")) if not prs.empty else "Origin"
        destination = str(prs.iloc[0].get("destination_code", "Destination")) if not prs.empty else "Destination"
        route = [
            {"sequence": 0, "stationCode": origin, "stationName": origin, "distanceFromOriginKm": 0, "isHalt": True},
            {"sequence": 1, "stationCode": destination, "stationName": destination, "distanceFromOriginKm": 0, "isHalt": True},
        ]

    return {
        "_fallback": True,
        "journeyDate": journey_date,
        # Do not invent a current station or delay when the provider has
        # rate-limited us. The route below is static/reference data only.
        "currentLocation": {},
        "delayMinutes": None,
        "runningStatus": "RATE_LIMITED",
        "route": route,
        "nextHalt": {},
        "lastUpdatedAt": datetime.now().astimezone().isoformat(),
        "fallbackMessage": "RailRadar rate limit reached; showing local route fallback until live service recovers.",
        "providerError": str(original_error),
    }


def get_live_data_or_fallback(train_number, journey_date=None):
    try:
        return get_live_train_status(train_number, journey_date)
    except Exception as error:
        if "429" in str(error) or "Too Many Requests" in str(error):
            return _rate_limit_fallback(train_number, journey_date, error)
        raise


def add_arrival_estimates(predictions, live_route=None, live_delay=0.0):
    """Attach platform and arrival-time data to each forecast station.

    RailRadar supplies scheduled/actual station times and platforms for a live
    journey.  Local ML routes do not contain a timetable, so they receive an
    explicitly labelled AI travel-time estimate instead of a misleading
    platform or scheduled time.
    """
    live_stops = {}
    for stop in live_route or []:
        code = str(_route_value(stop, "stationCode", "station_code", "code") or "")
        if code:
            live_stops[code] = stop

    now = datetime.now().astimezone()
    first_distance = _number(predictions[0].get("distance_km")) if predictions else 0.0

    for prediction in predictions:
        stop = live_stops.get(str(prediction.get("station_code", "")), {})
        platform = _route_value(
            stop,
            "platform", "platformNumber", "platform_number", "platformNo",
            "arrivalPlatform", "arrival_platform",
        )
        scheduled = _route_value(stop, "scheduledArrival", "scheduled_arrival")
        actual = _route_value(
            stop,
            "expectedArrival", "expected_arrival", "actualArrival", "actual_arrival",
        )
        station_delay = _number(
            _route_value(stop, "delayArrival", "delay_arrival", "delayMinutes"),
            live_delay,
        )

        prediction["platform"] = str(platform) if platform not in (None, "") else None
        prediction["scheduled_arrival_at"] = scheduled

        if actual:
            prediction["estimated_arrival_at"] = actual
            prediction["arrival_source"] = "RailRadar live update"
            continue

        if scheduled:
            try:
                scheduled_time = datetime.fromisoformat(str(scheduled).replace("Z", "+00:00"))
                prediction["estimated_arrival_at"] = (
                    scheduled_time + timedelta(minutes=station_delay)
                ).isoformat()
                prediction["arrival_source"] = "RailRadar schedule + live delay"
                continue
            except ValueError:
                pass

        distance_from_current = max(
            0.0,
            _number(prediction.get("distance_km")) - first_distance,
        )
        travel_minutes = (distance_from_current / 55.0) * 60.0
        model_delay = max(0.0, _number(prediction.get("predicted_delay_min")))
        prediction["estimated_arrival_at"] = (
            now + timedelta(minutes=travel_minutes + model_delay)
        ).isoformat()
        prediction["arrival_source"] = "AI travel-time estimate"

    return predictions


def build_feature_row(
    station_row,
    delay_so_far,
    weather,
    congestion,
    hour,
    day
):
    hour_sin = np.sin(
        2 * np.pi * hour / 24
    )

    hour_cos = np.cos(
        2 * np.pi * hour / 24
    )

    day_sin = np.sin(
        2 * np.pi * day / 7
    )

    day_cos = np.cos(
        2 * np.pi * day / 7
    )

    row = {
        "dist_remaining_km_norm":
            norm(
                "dist_remaining_km",
                station_row["dist_remaining_km"]
            ),

        "cum_distance_km_norm":
            norm(
                "cum_distance_km",
                station_row["cum_distance_km"]
            ),

        "avg_hist_delay_min_norm":
            norm(
                "avg_hist_delay_min",
                station_row["avg_hist_delay_min"]
            ),

        "delay_so_far_min_norm":
            norm(
                "delay_so_far_min",
                delay_so_far
            ),

        "hour_sin": hour_sin,
        "hour_cos": hour_cos,

        "day_sin": day_sin,
        "day_cos": day_cos,

        "weather_clear":
            float(weather == "Clear"),

        "weather_rain":
            float(weather == "Rain"),

        "weather_fog":
            float(weather == "Fog"),

        "weather_storm":
            float(weather == "Storm"),

        "congestion_low":
            float(congestion == "Low"),

        "congestion_medium":
            float(congestion == "Medium"),

        "congestion_high":
            float(congestion == "High"),
    }

    return [
        row[c]
        for c in FEATURE_COLS
    ]


def run_inference(
    train_number: str,
    delay_so_far: float,
    weather: str,
    congestion: str,
    start_hour: int,
    day: int,
    from_seq: int = 0
):
    """
    Predict delay at every remaining station
    for an ML-supported train.
    """

    train_number = normalize_train_number(
        train_number
    )

    route = (
        stations_df[
            stations_df["train_number"] == train_number
        ]
        .sort_values("seq")
        .copy()
    )

    if route.empty:
        raise HTTPException(
            status_code=404,
            detail=(
                f"Train {train_number} does not "
                "have an ML route dataset."
            )
        )

    total_distance = float(
        route["cum_distance_km"].max()
    )

    remaining = route[
        route["seq"] >= from_seq
    ].copy()

    remaining["dist_remaining_km"] = (
        total_distance
        - remaining["cum_distance_km"]
    )

    feats = []

    hour = start_hour

    for _, station in remaining.iterrows():

        feats.append(
            build_feature_row(
                station,
                delay_so_far,
                weather,
                congestion,
                hour % 24,
                day
            )
        )

        hour += 1

    if not feats:
        return []

    x = torch.tensor(
        [feats],
        dtype=torch.float32
    )

    with torch.no_grad():

        pred_norm = (
            model(x)
            .squeeze(0)
            .numpy()
        )

    pred_delay = denorm(
        "target_delay_min",
        pred_norm
    )

    pred_delay = np.clip(
        pred_delay,
        0,
        None
    )

    results = []

    for (_, station), delay in zip(
        remaining.iterrows(),
        pred_delay
    ):

        results.append({

            "seq":
                int(station["seq"]),

            "station_code":
                str(station["station_code"]),

            "station_name":
                str(
                    station["station_name"]
                ).strip(),

            "distance_km":
                float(
                    station["cum_distance_km"]
                ),

            "predicted_delay_min":
                round(
                    float(delay),
                    1
                ),

            "static_baseline_delay_min":
                round(
                    float(
                        station[
                            "avg_hist_delay_min"
                        ]
                    ),
                    1
                )
        })

    return results
def run_universal_inference(
    train_number: str,
    route: list,
    delay_so_far: float,
    weather: str = "Clear",
    congestion: str = "Low",
    start_hour: int = 8,
    day: int = 0,
    from_seq: int = 0
):
    """
    Run the existing LSTM against any available train route.

    For the 42 trained trains, the original route data is used.
    For other trains, RailRadar route data is converted into
    the same feature structure expected by the LSTM.
    """

    if not route:
        return []

    # Convert RailRadar route into a dataframe
    rows = []

    for i, station in enumerate(route):
        rows.append({
            "seq": int(
                station.get("sequence", station.get("seq", i))
            ),
            "station_code": str(
                station.get(
                    "stationCode",
                    station.get("station_code", "")
                )
            ),
            "station_name": str(
                station.get(
                    "stationName",
                    station.get("station_name", "")
                )
            ),
            "cum_distance_km": _number(
                _route_value(
                    station,
                    "distanceFromOriginKm",
                    "cum_distance_km",
                    "distance",
                ),
                0,
            ),
            "avg_hist_delay_min": float(
                station.get(
                    "avg_hist_delay_min",
                    0
                ) or 0
            )
        })

    df = pd.DataFrame(rows)

    if df.empty:
        return []

    df = df.sort_values("seq").reset_index(drop=True)

    total_distance = float(
        df["cum_distance_km"].max()
    )

    df["dist_remaining_km"] = (
        total_distance -
        df["cum_distance_km"]
    )

    remaining = df[
        df["seq"] >= from_seq
    ].copy()

    if remaining.empty:
        return []

    features = []
    hour = start_hour

    for _, station in remaining.iterrows():

        features.append(
            build_feature_row(
                station,
                delay_so_far,
                weather,
                congestion,
                hour % 24,
                day
            )
        )

        hour += 1

    x = torch.tensor(
        [features],
        dtype=torch.float32
    )

    with torch.no_grad():
        pred_norm = (
            model(x)
            .squeeze(0)
            .numpy()
        )

    pred_delay = denorm(
        "target_delay_min",
        pred_norm
    )

    pred_delay = np.clip(
        pred_delay,
        0,
        None
    )

    results = []

    for (_, station), delay in zip(
        remaining.iterrows(),
        pred_delay
    ):

        results.append({
            "seq": int(station["seq"]),

            "station_code":
                str(station["station_code"]),

            "station_name":
                str(station["station_name"]).strip(),

            "distance_km":
                round(
                    float(
                        station["cum_distance_km"]
                    ),
                    1
                ),

            "predicted_delay_min":
                round(
                    float(delay),
                    1
                ),

            "static_baseline_delay_min":
                round(
                    float(
                        station["avg_hist_delay_min"]
                    ),
                    1
                )
        })

    return results

# ===============================================================
# API MODELS
# ===============================================================

class SimulateRequest(BaseModel):

    run_id: Optional[str] = "demo"

    current_seq: int = 0

    weather: str = "Clear"

    congestion: str = "Low"

    hour_of_day: int = 8

    day_of_week: int = 0

    live_delay_observed_min: float = 0.0


# ===============================================================
# ROOT
# ===============================================================

@app.get("/", include_in_schema=False)
def root():
    """Open RailConnect UI directly at http://localhost:8000/."""
    index_file = os.path.join(DASHBOARD_DIR, "index.html")
    if not os.path.isfile(index_file):
        raise HTTPException(status_code=404, detail="Dashboard index.html not found")
    # The dashboard is edited frequently during the prototype; avoid serving
    # a stale HTML shell that hides newly added controls in the browser.
    return FileResponse(index_file, headers={"Cache-Control": "no-store"})


@app.get("/api/health")
def api_health():
    """Backend health/model information for the dashboard."""
    try:
        live_poll_interval_seconds = min(
            3600,
            max(30, int(os.getenv("RAILRADAR_POLL_INTERVAL_SECONDS", "60"))),
        )
    except ValueError:
        live_poll_interval_seconds = 60
    return {
        "service": "RailConnect - Dynamic Railway ETA Prediction API",
        "model": "PyTorch LSTM sequence regressor",
        "validation_report": EVAL_REPORT,
        "ml_supported_trains": int(stations_df["train_number"].nunique()),
        "catalogue_trains": int(len(railradar_trains_df)),
        "live_poll_interval_seconds": live_poll_interval_seconds,
        "docs": "/docs",
    }


@app.get("/trains")
def list_all_trains():

    """
    Return ALL trains from the RailRadar catalogue.

    ML-supported trains are marked:
        ml_supported = True

    Other trains:
        ml_supported = False
    """

    catalogue = (
        railradar_trains_df
        .copy()
    )

    if catalogue.empty:

        # Fallback to trains.csv
        catalogue = (
            trains_df
            .copy()
        )

        if "train_name" not in catalogue.columns:

            catalogue["train_name"] = ""

    # Set of trains for which stations.csv exists
    ml_numbers = set(
        stations_df[
            "train_number"
        ]
        .astype(str)
        .unique()
    )

    out = []

    for _, train in catalogue.iterrows():

        number = normalize_train_number(
            train.get(
                "train_number",
                ""
            )
        )

        if not number:
            continue

        name = str(
            train.get(
                "train_name",
                ""
            )
        ).strip()

        route = (
            stations_df[
                stations_df["train_number"]
                == number
            ]
            .sort_values("seq")
        )

        item = {
            "train_number":
                number,

            "train_name":
                name,

            "ml_supported":
                number in ml_numbers,

            "origin":
                None,

            "destination":
                None,

            "n_stations":
                0,

            "total_distance_km":
                0
        }

        # Add route information if ML data exists
        if not route.empty:

            item["origin"] = (
                str(
                    route.iloc[0]["station_name"]
                ).strip()
            )

            item["destination"] = (
                str(
                    route.iloc[-1]["station_name"]
                ).strip()
            )

            item["n_stations"] = int(
                len(route)
            )

            item["total_distance_km"] = round(
                float(
                    route[
                        "cum_distance_km"
                    ].max()
                ),
                1
            )

        out.append(item)

    return sorted(
        out,
        key=lambda x: (
            int(x["train_number"])
            if x["train_number"].isdigit()
            else 999999999
        )
    )


# ===============================================================
# SEARCH ALL TRAINS
# ===============================================================

@app.get("/trains/catalogue")
def get_train_catalogue(
    search: Optional[str] = None,
    limit: int = 100
):
    """
    Search ALL trains by train number or train name.

    Examples:
        /trains/catalogue?search=20503
        /trains/catalogue?search=rajdhani
        /trains/catalogue?search=12919
    """

    df = (
        railradar_trains_df
        .copy()
    )

    if df.empty:
        return []

    if search:

        search = search.strip()

        search_lower = search.lower()
        compact_search = "".join(char for char in search_lower if char.isalnum())
        train_numbers = df["train_number"].astype(str)
        train_names = df["train_name"].astype(str)
        compact_names = train_names.str.lower().str.replace(
            r"[^a-z0-9]", "", regex=True
        )

        number_mask = (
            train_numbers
            .str.contains(
                search,
                case=False,
                na=False,
                regex=False
            )
        )

        name_mask = (
            train_names
            .str.contains(
                search,
                case=False,
                na=False,
                regex=False
            )
            | compact_names.str.contains(
                compact_search,
                na=False,
                regex=False
            )
        )

        df = df[
            number_mask
            | name_mask
        ].copy()

        # A user typing a train number expects that number at the top, not a
        # random substring match from the 10k-row catalogue.
        matched_numbers = df["train_number"].astype(str)
        matched_names = df["train_name"].astype(str).str.lower()
        matched_compact_names = matched_names.str.replace(
            r"[^a-z0-9]", "", regex=True
        )
        df["_search_rank"] = np.select(
            [
                matched_numbers.eq(search),
                matched_numbers.str.startswith(search),
                matched_names.str.startswith(search_lower),
                matched_compact_names.str.startswith(compact_search),
            ],
            [0, 1, 2, 2],
            default=3,
        )
        df = df.sort_values(["_search_rank", "train_number"])

    limit = max(
        1,
        min(
            int(limit),
            500
        )
    )

    ml_numbers = set(
        stations_df[
            "train_number"
        ]
        .astype(str)
        .unique()
    )

    results = []

    for _, row in df.head(limit).iterrows():

        number = normalize_train_number(
            row["train_number"]
        )

        results.append({

            "train_number":
                number,

            "train_name":
                str(
                    row.get(
                        "train_name",
                        ""
                    )
                ).strip(),

            "ml_supported":
                number in ml_numbers
        })

    return results


# ===============================================================
# TRAIN ROUTE
# ===============================================================

@app.get("/trains/{train_number}/route")
def get_route(
    train_number: str,
    journey_date: Optional[str] = Query(None, alias="date"),
):
    train_number = normalize_train_number(train_number)
    journey_date = _journey_date(journey_date)

    route = (
        stations_df[stations_df["train_number"] == train_number]
        .sort_values("seq")
        .copy()
    )
    if not route.empty:
        return route.to_dict(orient="records")

    # For catalogue-only trains, derive scheduled halts from one live request.
    # Avoid a second upstream call because RailRadar rate-limits live status.
    try:
        live = get_live_data_or_fallback(train_number, journey_date)
        full_route = live.get("route", []) or []
        radar_route = _halts_from_route(full_route) or full_route
        if radar_route:
            return radar_route
    except Exception:
        pass

    raise HTTPException(status_code=404, detail="Route is not currently available for this train.")


# ===============================================================
# TRAIN DETAILS
# ===============================================================

@app.get("/trains/{train_number}/details")
def get_train_details(
    train_number: str,
    journey_date: Optional[str] = Query(None, alias="date"),
):
    train_number = normalize_train_number(train_number)
    journey_date = _journey_date(journey_date)

    catalogue_rows = railradar_trains_df[railradar_trains_df["train_number"] == train_number]
    route = stations_df[stations_df["train_number"] == train_number].sort_values("seq").copy()

    if catalogue_rows.empty and route.empty:
        raise HTTPException(status_code=404, detail="Train not found")

    train_name = "" if catalogue_rows.empty else str(catalogue_rows.iloc[0].get("train_name", "")).strip()
    details = {"train_number": train_number, "train_name": train_name, "ml_supported": not route.empty,
               "origin": None, "destination": None, "n_stations": 0, "total_distance_km": 0}

    if not route.empty:
        details.update({"origin": str(route.iloc[0]["station_name"]).strip(),
                        "destination": str(route.iloc[-1]["station_name"]).strip(),
                        "n_stations": int(len(route)),
                        "total_distance_km": round(float(route["cum_distance_km"].max()), 1)})
    else:
        try:
            live = get_live_data_or_fallback(train_number, journey_date)
            rr = live.get("route", []) or []
            if rr:
                details["origin"] = rr[0].get("stationName") or rr[0].get("station_name")
                details["destination"] = rr[-1].get("stationName") or rr[-1].get("station_name")
                details["n_stations"] = len(rr)
                details["total_distance_km"] = round(float(rr[-1].get("distanceFromOriginKm", 0) or 0), 1)
        except Exception:
            pass

    return details


# ===============================================================
# SIMULATION / ML ETA
# ===============================================================

@app.post(
    "/trains/{train_number}/simulate"
)
def simulate(
    train_number: str,
    req: SimulateRequest
):

    train_number = normalize_train_number(
        train_number
    )

    # Ensure this train has ML route data
    route = stations_df[
        stations_df["train_number"]
        == train_number
    ]

    if route.empty:

        raise HTTPException(
            status_code=400,
            detail=(
                f"AI ETA is not available "
                f"for train {train_number}. "
                "This train is not part of "
                "the trained ML route dataset."
            )
        )

    # Validate weather/congestion
    if req.weather not in WEATHER_STATES:

        raise HTTPException(
            status_code=400,
            detail=(
                "Invalid weather. Choose: "
                + ", ".join(WEATHER_STATES)
            )
        )

    if req.congestion not in CONGESTION_LEVELS:

        raise HTTPException(
            status_code=400,
            detail=(
                "Invalid congestion. Choose: "
                + ", ".join(CONGESTION_LEVELS)
            )
        )

    key = (
        f"{train_number}_"
        f"{req.run_id}"
    )

    LIVE_RUNS[key] = {

        "current_seq":
            req.current_seq,

        "delay_so_far_min":
            req.live_delay_observed_min,

        "weather":
            req.weather,

        "congestion":
            req.congestion,

        "hour_of_day":
            req.hour_of_day,

        "day_of_week":
            req.day_of_week
    }

    predictions = run_inference(

        train_number=
            train_number,

        delay_so_far=
            req.live_delay_observed_min,

        weather=
            req.weather,

        congestion=
            req.congestion,

        start_hour=
            req.hour_of_day,

        day=
            req.day_of_week,

        from_seq=
            req.current_seq
    )

    return {

        "train_number":
            train_number,

        "run_id":
            req.run_id,

        "as_of_seq":
            req.current_seq,

        "live_conditions":
            req.dict(),

        "eta_forecast":
            predictions
    }


# ===============================================================
# UNIVERSAL LIVE FORECAST
# ===============================================================

@app.post("/trains/{train_number}/forecast")
def forecast_all_trains(
    train_number: str,
    req: SimulateRequest,
    journey_date: Optional[str] = Query(None, alias="date"),
):
    train_number = normalize_train_number(train_number)
    journey_date = _journey_date(journey_date)
    if req.weather not in WEATHER_STATES:
        raise HTTPException(status_code=400, detail="Invalid weather. Choose: " + ", ".join(WEATHER_STATES))
    if req.congestion not in CONGESTION_LEVELS:
        raise HTTPException(status_code=400, detail="Invalid congestion. Choose: " + ", ".join(CONGESTION_LEVELS))

    route = stations_df[stations_df["train_number"] == train_number].sort_values("seq").copy()
    ml_supported = not route.empty
    weather_status = None

    if ml_supported:
        max_seq = max(0, len(route) - 1)
        current_seq = min(max(int(req.current_seq), 0), max_seq)
        weather_status = get_weather_status(str(route.iloc[current_seq]["station_name"]))
        predictions = run_inference(train_number, req.live_delay_observed_min, weather_status["model_state"], req.congestion,
                                    req.hour_of_day, req.day_of_week, current_seq)
        live_snapshot = None
    else:
        try:
            live_snapshot = get_live_data_or_fallback(train_number, journey_date)
        except Exception as e:
            raise HTTPException(status_code=502, detail=f"Unable to fetch live train data: {e}")
        full_route = live_snapshot.get("route", []) or []
        radar_route = _halts_from_route(full_route) or full_route
        if not radar_route:
            raise HTTPException(status_code=404, detail="Live route data is unavailable for this train right now.")
        selected_index = min(max(int(req.current_seq), 0), len(radar_route)-1)
        current_seq = _stop_sequence(radar_route[selected_index], selected_index)
        current_location = live_snapshot.get("currentLocation") or {}
        weather_status = get_weather_status(
            _route_value(radar_route[selected_index], "stationName", "station_name", "name"),
            current_location.get("latitude", current_location.get("lat")),
            current_location.get("longitude", current_location.get("lng")),
        )
        predictions = run_universal_inference(train_number, radar_route, req.live_delay_observed_min,
                                              weather_status["model_state"], req.congestion, req.hour_of_day, req.day_of_week, current_seq)

    add_arrival_estimates(
        predictions,
        live_snapshot.get("route", []) if live_snapshot else None,
        req.live_delay_observed_min,
    )

    snapshot = ({**live_snapshot, "halts": radar_route} if live_snapshot else None)
    return {"train_number": train_number, "ml_supported": ml_supported,
            "prediction_mode": "LSTM - trained route" if ml_supported else "LSTM - generalized route",
            "as_of_seq": current_seq, "live_conditions": {**req.dict(), "weather": weather_status}, "eta_forecast": predictions,
            "live_snapshot": snapshot}


# ===============================================================
# CURRENT ML ETA
# ===============================================================

@app.get(
    "/trains/{train_number}/eta"
)
def get_current_eta(
    train_number: str,
    run_id: str = "demo"
):

    train_number = normalize_train_number(
        train_number
    )

    key = (
        f"{train_number}_"
        f"{run_id}"
    )

    if key not in LIVE_RUNS:

        raise HTTPException(
            status_code=404,
            detail=(
                "No active run - "
                "POST to /simulate first"
            )
        )

    state = LIVE_RUNS[key]

    predictions = run_inference(

        train_number=
            train_number,

        delay_so_far=
            state["delay_so_far_min"],

        weather=
            state["weather"],

        congestion=
            state["congestion"],

        start_hour=
            state["hour_of_day"],

        day=
            state["day_of_week"],

        from_seq=
            state["current_seq"]
    )

    return {

        "train_number":
            train_number,

        "run_id":
            run_id,

        "state":
            state,

        "eta_forecast":
            predictions
    }


# ===============================================================
# LIVE RAILRADAR STATUS
# ===============================================================

@app.get(
    "/trains/{train_number}/live"
)
def get_live_status(
    train_number: str,
    journey_date: Optional[str] = Query(None, alias="date"),
):
    train_number = normalize_train_number(train_number)
    journey_date = _journey_date(journey_date)

    try:

        # -------------------------------------------------------
        # 1. GET LIVE DATA
        # -------------------------------------------------------

        live_data = get_live_data_or_fallback(train_number, journey_date)

        current_location = live_data.get("currentLocation") or {}
        if not isinstance(current_location, dict):
            current_location = {}

        current_station_code = (
            current_location.get(
                "stationCode"
            )
        )

        current_station_name = (
            current_location.get(
                "stationName"
            )
        )

        # -------------------------------------------------------
        # 2. LOCAL ML ROUTE
        # -------------------------------------------------------

        local_route = (
            stations_df[
                stations_df["train_number"]
                == train_number
            ]
            .sort_values("seq")
            .copy()
        )

        ml_supported = not local_route.empty

        predictions = []
        current_seq = 0
        live_delay = (
            None if live_data.get("_fallback") else _number(
                current_location.get(
                    "delayMinutes",
                    live_data.get("delayMinutes", live_data.get("overallDelayMinutes", 0))
                )
            )
        )
        model_delay = live_delay if live_delay is not None else 0.0

        # The provider's currentLocation normally contains only a code. Use
        # the route record to supply the readable station name and sequence.
        live_route = live_data.get("route", []) or []
        halted_route = []
        current_route_stop = next(
            (stop for stop in live_route if _stop_code(stop) == str(current_station_code or "").strip().upper()),
            None,
        )
        if not ml_supported:
            halted_route = _halts_from_route(live_route) or live_route
        for stop in live_route:
            if str(stop.get("stationCode", "")) == str(current_station_code):
                current_station_name = current_station_name or stop.get("stationName")
                break

        distance_covered_km = _number(
            _route_value(
                current_location,
                "distanceCoveredKm", "distance_covered_km",
                "distanceFromOriginKm", "distance",
            ),
            None,
        )
        distance_source = "RailRadar reported distance" if distance_covered_km is not None else None
        if distance_covered_km is None and current_station_code:
            reported_stop = next(
                (stop for stop in live_route if _stop_code(stop) == str(current_station_code).strip().upper()),
                None,
            )
            if reported_stop:
                distance_covered_km = _number(
                    _route_value(reported_stop, "distanceFromOriginKm", "cum_distance_km", "distance"),
                    None,
                )
                if distance_covered_km is not None:
                    distance_source = "RailRadar reported station distance"

        if distance_covered_km is None:
            previous_halt = live_data.get("previousHalt") or live_data.get("previous_halt") or {}
            next_halt_for_distance = live_data.get("nextHalt") or live_data.get("next_halt") or {}
            previous_distance = _number(
                _route_value(previous_halt, "distanceFromOriginKm", "cum_distance_km", "distance"),
                None,
            )
            next_distance = _number(
                _route_value(next_halt_for_distance, "distanceFromOriginKm", "cum_distance_km", "distance"),
                None,
            )
            segment_progress = _number(
                _route_value(current_location, "segmentProgress", "segment_progress"),
                None,
            )
            if (
                previous_distance is not None
                and next_distance is not None
                and next_distance >= previous_distance
                and segment_progress is not None
                and 0 <= segment_progress <= 1
            ):
                distance_covered_km = previous_distance + (
                    (next_distance - previous_distance) * segment_progress
                )
                distance_source = "Approx. distance from reported segment progress"

        weather_lat = current_location.get("latitude", current_location.get("lat"))
        weather_lon = current_location.get("longitude", current_location.get("lng"))
        if weather_lat is None or weather_lon is None:
            weather_lat = _route_value(current_route_stop or {}, "latitude", "lat")
            weather_lon = _route_value(current_route_stop or {}, "longitude", "lng", "lon")
        weather_status = get_weather_status(current_station_name, weather_lat, weather_lon)
        weather = weather_status["model_state"]

        # -------------------------------------------------------
        # 3. ML TRAIN
        # -------------------------------------------------------

        if ml_supported:

            current_rows = local_route[
                local_route["station_code"]
                .astype(str)
                ==
                str(current_station_code)
            ]

            if not current_rows.empty:

                current_seq = int(
                    current_rows.iloc[0]["seq"]
                )
            else:
                # RailRadar can report an intermediate waypoint (for example,
                # Old Malda) that is not one of the LSTM's scheduled main
                # stations. Map that location to the next scheduled main halt
                # in the provider route instead of incorrectly starting the
                # LSTM forecast from the journey origin.
                live_index = next(
                    (
                        index
                        for index, stop in enumerate(live_route)
                        if str(stop.get("stationCode", "")).strip().upper()
                        == str(current_station_code or "").strip().upper()
                    ),
                    None,
                )

                if live_index is not None:
                    live_positions = {
                        str(stop.get("stationCode", "")).strip().upper(): index
                        for index, stop in enumerate(live_route)
                    }
                    upcoming_main_stops = local_route.assign(
                        _live_index=local_route["station_code"]
                        .astype(str)
                        .str.strip()
                        .str.upper()
                        .map(live_positions)
                    )
                    upcoming_main_stops = upcoming_main_stops[
                        upcoming_main_stops["_live_index"].notna()
                        & (upcoming_main_stops["_live_index"] >= live_index)
                    ]
                    if not upcoming_main_stops.empty:
                        current_seq = int(upcoming_main_stops.iloc[0]["seq"])

            predictions = run_inference(
                train_number=train_number,
                delay_so_far=model_delay,
                weather=weather,
                congestion="Low",
                start_hour=datetime.now().hour,
                day=datetime.now().weekday(),
                from_seq=current_seq
            )

            prediction_mode = (
                "LSTM - trained route"
            )

        # -------------------------------------------------------
        # 4. NON-ML TRAIN
        # -------------------------------------------------------

        else:

            radar_route = halted_route or _halts_from_route(live_route)

            if radar_route:

                # The train may be between halts. Use the next scheduled halt
                # rather than an intermediate waypoint or the route origin.
                current_full_seq = next(
                    (
                        _stop_sequence(stop, index)
                        for index, stop in enumerate(live_route)
                        if _stop_code(stop) == str(current_station_code or "").strip().upper()
                    ),
                    None,
                )
                if current_full_seq is not None:
                    current_seq = next(
                        (
                            _stop_sequence(stop, index)
                            for index, stop in enumerate(radar_route)
                            if _stop_sequence(stop, index) >= current_full_seq
                        ),
                        _stop_sequence(radar_route[-1], len(radar_route) - 1),
                    )
                else:
                    current_seq = _stop_sequence(radar_route[0], 0)

                predictions = (
                    run_universal_inference(
                        train_number=train_number,
                        route=radar_route,
                        delay_so_far=model_delay,
                        weather=weather,
                        congestion="Low",
                        start_hour=datetime.now().hour,
                        day=datetime.now().weekday(),
                        from_seq=current_seq
                    )
                )

                prediction_mode = (
                    "LSTM - generalized route"
                )

            else:

                prediction_mode = (
                    "No route available"
                )

        # -------------------------------------------------------
        # 5. RESPONSE
        # -------------------------------------------------------

        add_arrival_estimates(predictions, live_route, model_delay)

        current_live_index = next(
            (
                index
                for index, stop in enumerate(live_route)
                if str(stop.get("stationCode", "")).strip().upper()
                == str(current_station_code or "").strip().upper()
            ),
            None,
        )
        next_halt = live_data.get("nextHalt") or live_data.get("next_halt") or {}
        if not isinstance(next_halt, dict):
            next_halt = {}
        if not next_halt and current_live_index is not None:
            next_halt = live_route[current_live_index + 1] if current_live_index + 1 < len(live_route) else {}

        next_station_code = _route_value(next_halt, "stationCode", "station_code", "code")
        next_station_name = _route_value(next_halt, "stationName", "station_name", "name")
        # RailRadar sometimes returns only the code/name in ``nextHalt`` and
        # keeps the platform and timing details on the corresponding route
        # item. Merge the useful fields so the upcoming-stop card can use
        # both shapes of the provider payload.
        next_route_stop = next(
            (
                stop for stop in live_route
                if _stop_code(stop) == str(next_station_code or "").strip().upper()
            ),
            {},
        )
        if next_route_stop:
            next_halt = {
                **next_route_stop,
                **{
                    key: value for key, value in next_halt.items()
                    if value not in (None, "")
                },
            }
            next_station_code = _route_value(next_halt, "stationCode", "station_code", "code")
            next_station_name = _route_value(next_halt, "stationName", "station_name", "name")
        live_reference_time = _live_eta_datetime(
            live_data.get("lastUpdatedAt") or live_data.get("last_updated_at")
        )
        if not live_reference_time and journey_date:
            live_reference_time = datetime.strptime(journey_date, "%Y-%m-%d").replace(
                tzinfo=datetime.now().astimezone().tzinfo
            )
        # Count down against the current clock, not the provider snapshot time;
        # otherwise a delayed refresh can leave the ETA frozen or already expired.
        live_reference_time = datetime.now().astimezone()

        live_eta_value = _route_value(
            next_halt,
            "expectedArrival", "expected_arrival", "etaAt", "eta_at",
            "arrivalTime", "arrival_time",
        )
        scheduled_eta_value = _route_value(
            next_halt, "scheduledArrival", "scheduled_arrival",
        )
        next_eta_at = _live_eta_datetime(live_eta_value, now=live_reference_time)
        next_eta_source = "RailRadar live ETA" if next_eta_at else None
        if not next_eta_at:
            next_eta_at = _live_eta_datetime(scheduled_eta_value, now=live_reference_time)
            next_eta_source = "RailRadar scheduled arrival" if next_eta_at else None

        next_eta_minutes = _number(
            _route_value(
                next_halt,
                "etaMinutes", "eta_minutes", "minutesToArrival", "minutes_to_arrival",
            ),
            None,
        )
        if next_eta_minutes is not None:
            next_eta_minutes = max(0, round(next_eta_minutes))
            next_eta_source = "RailRadar live ETA"
        elif next_eta_at:
            eta_delta = round((next_eta_at - live_reference_time).total_seconds() / 60)
            # A scheduled arrival that has already passed is not a live ETA.
            # Keep it out of the card so we can use the model/route estimate below.
            if eta_delta >= 1:
                next_eta_minutes = eta_delta
            else:
                next_eta_at = None
                next_eta_source = None

        next_prediction = next(
            (
                prediction
                for prediction in predictions
                if str(prediction.get("station_code", "")).strip().upper()
                == str(next_station_code or "").strip().upper()
            ),
            None,
        )
        next_halt_distance_for_eta = _number(
            _route_value(next_halt, "distanceFromOriginKm", "cum_distance_km", "distance_km", "distance"),
            None,
        )
        if (
            next_eta_minutes == 0
            and distance_covered_km is not None
            and next_halt_distance_for_eta is not None
            and next_halt_distance_for_eta - distance_covered_km > 1
        ):
            next_eta_minutes = None
            next_eta_at = None
            next_eta_source = None
        if next_eta_minutes is None and next_prediction and (distance_covered_km is None or _number(_route_value(next_halt, "distanceFromOriginKm", "cum_distance_km", "distance_km", "distance"), None) is None):
            # Keep the card useful when the upstream feed provides the next
            # stop but no ETA. This is a forecast, not live telemetry.
            next_eta_at = _live_eta_datetime(
                next_prediction.get("estimated_arrival_at"),
                now=live_reference_time,
            )
            if next_eta_at:
                prediction_minutes = round((next_eta_at - live_reference_time).total_seconds() / 60)
                if prediction_minutes > 0:
                    next_eta_minutes = prediction_minutes
                    next_eta_source = next_prediction.get(
                        "arrival_source", "AI travel-time estimate"
                    )
                else:
                    next_eta_at = None
        if next_eta_minutes is None:
            next_halt_distance = _number(
                _route_value(next_halt, "distanceFromOriginKm", "cum_distance_km", "distance_km", "distance"),
                None,
            )
            speed_kmh = _number(
                _route_value(current_location, "speedKmh", "speed_kmh", "speed"),
                None,
            )
            if distance_covered_km is not None and next_halt_distance is not None:
                remaining_km = next_halt_distance - distance_covered_km
                if remaining_km >= 0:
                    travel_speed = speed_kmh if speed_kmh is not None and speed_kmh > 5 else 55.0
                    travel_minutes = remaining_km * 60 / travel_speed
                    model_delay_at_halt = _number(
                        next_prediction.get("predicted_delay_min") if next_prediction else None,
                        None,
                    )
                    if model_delay_at_halt is not None and live_delay is not None:
                        travel_minutes += model_delay_at_halt - live_delay
                    next_eta_minutes = max(1, round(travel_minutes)) if remaining_km > 0 else 0
                    next_eta_source = (
                        "AI travel-time + LSTM delay estimate"
                        if model_delay_at_halt is not None
                        else "AI travel-time estimate from current location"
                    )
                    next_eta_at = live_reference_time + timedelta(minutes=next_eta_minutes)

        next_platform = _route_value(
            next_halt,
            "platform", "platformNumber", "platform_number", "platformNo",
            "arrivalPlatform", "arrival_platform",
        )
        if next_platform in (None, "") and next_prediction:
            next_platform = next_prediction.get("platform")

        return {

            "train_number":
                train_number,

            "source": "Local fallback" if live_data.get("_fallback") else "RailRadar",

            "journey_date": journey_date or live_data.get("journeyDate"),

            "ml_supported":
                ml_supported,

            "prediction_available":
                len(predictions) > 0,

            "prediction_mode":
                prediction_mode,

            "message": live_data.get("fallbackMessage") or (
                "Live location and ETA prediction available."
                if predictions else
                "Live location available, but route data is unavailable."
            ),

            "current_location":
                current_location,

            "current_station": {

                "code":
                    current_station_code,

                "name":
                    current_station_name,

                "sequence":
                    current_seq
            },

            "live_delay_min": live_delay,

            "weather": weather_status,

            # Live railway values: travel time comes only from the provider's
            # ETA/schedule; the LSTM is used only for the delay prediction.
            "current_delay_min": live_delay,

            "distance_covered_km": distance_covered_km,

            "distance_source": distance_source,

            "next_station": {
                "code": next_station_code,
                "name": next_station_name,
                "platform": str(next_platform) if next_platform not in (None, "") else None,
            },

            "next_station_eta_at": next_eta_at.isoformat() if next_eta_at else None,

            "next_station_eta_minutes": next_eta_minutes,

            "next_station_eta_source": next_eta_source,

            "next_station_predicted_delay_min": (
                _number(next_prediction.get("predicted_delay_min"))
                if next_prediction else None
            ),

            "next_station_prediction_station": (
                next_prediction.get("station_name") if next_prediction else None
            ),

            "prediction_time":
                datetime.now().isoformat(),

            "route": live_route,

            # Scheduled halts are separately returned for non-LSTM trains.
            # The full route stays available above for live positioning.
            "halts": halted_route if not ml_supported else [],

            "eta_forecast":
                predictions,

            "live_status":
                live_data
        }

    except HTTPException:
        raise

    except Exception as e:

        raise HTTPException(
            status_code=502,
            detail=(
                "Unable to fetch live "
                "train data: "
                f"{str(e)}"
            )
        )



# Serve CSS/assets and also allow /dashboard/ access.
if os.path.isdir(DASHBOARD_DIR):
    app.mount("/dashboard", StaticFiles(directory=DASHBOARD_DIR, html=True), name="dashboard")
