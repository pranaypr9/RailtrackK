import os
import time
from typing import Optional

import requests
from dotenv import load_dotenv

load_dotenv()

RAILRADAR_API_KEY = os.getenv("RAILRADAR_API_KEY")

BASE_URL = "https://api.railradar.in/v1"
LIVE_CACHE_TTL_SECONDS = 28
_LIVE_CACHE = {}


def get_live_train_status(
    train_number: str,
    journey_date: Optional[str] = None,
    halts_only: bool = False,
    include_coordinates: bool = True,
    include_geometry: bool = True,
):
    # Reload on every request so replacing .env credentials takes effect in a
    # running dashboard without waiting for the server reloader to notice it.
    load_dotenv(override=True)
    api_key = os.getenv("RAILRADAR_API_KEY")

    if not api_key:
        raise RuntimeError("RAILRADAR_API_KEY is missing from .env")

    cache_key = (
        str(train_number), journey_date or "", bool(halts_only),
        bool(include_coordinates), bool(include_geometry),
    )
    cached = _LIVE_CACHE.get(cache_key)
    now = time.monotonic()
    if cached and now - cached[0] < LIVE_CACHE_TTL_SECONDS:
        return cached[1]

    url = f"{BASE_URL}/trains/{train_number}/live"

    headers = {
        "Authorization": f"Bearer {api_key}"
    }

    params = {}
    if journey_date:
        params["date"] = journey_date
    if halts_only:
        params["haltsOnly"] = "true"
    if include_coordinates:
        params["includeCoordinates"] = "true"
    if include_geometry:
        params["geometry"] = "true"
        params["format"] = "geojson"

    response = requests.get(
        url,
        headers=headers,
        params=params or None,
        timeout=15
    )

    response.raise_for_status()

    data = response.json()

    if not data.get("success"):
        raise RuntimeError(
            data.get("message", "RailRadar API returned an error")
        )

    payload = data["data"]
    _LIVE_CACHE[cache_key] = (now, payload)
    return payload
