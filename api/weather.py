"""Visual Crossing-backed conditions used by ETA forecasts."""

import os
import time
from typing import Optional

import requests
from dotenv import load_dotenv


VISUAL_CROSSING_TIMELINE_URL = (
    "https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline"
)
WEATHER_CACHE_TTL_SECONDS = 600
_WEATHER_CACHE = {}


def weather_state(condition: str) -> str:
    """Map provider conditions to the ETA model's categories."""
    value = (condition or "").strip().lower()
    if any(term in value for term in ("thunder", "tornado", "squall", "hail")):
        return "Storm"
    if any(term in value for term in ("rain", "drizzle", "shower")):
        return "Rain"
    if any(term in value for term in ("mist", "smoke", "haze", "dust", "fog", "sand", "ash")):
        return "Fog"
    return "Clear"


def get_weather_status(station_name: Optional[str] = None, latitude=None, longitude=None) -> dict:
    """Return current weather without exposing the provider key."""
    load_dotenv(override=True)
    api_key = os.getenv("VISUAL_CROSSING_API_KEY")
    if not api_key:
        return {"available": False, "model_state": "Clear", "message": "Weather API key is not configured."}

    try:
        lat, lon = float(latitude), float(longitude)
        location, cache_key = f"{lat:.4f},{lon:.4f}", (round(lat, 2), round(lon, 2))
    except (TypeError, ValueError):
        location = (station_name or "").strip()
        if not location:
            return {"available": False, "model_state": "Clear", "message": "Train location is not available for a weather lookup."}
        location, cache_key = f"{location}, India", (location.lower(), "india")

    cached, now = _WEATHER_CACHE.get(cache_key), time.monotonic()
    if cached and now - cached[0] < WEATHER_CACHE_TTL_SECONDS:
        return cached[1]
    try:
        response = requests.get(
            f"{VISUAL_CROSSING_TIMELINE_URL}/{location}/today",
            params={"unitGroup": "metric", "include": "current", "key": api_key, "contentType": "json"},
            timeout=10,
        )
        response.raise_for_status()
        data = response.json()
        current = data.get("currentConditions") or {}
        condition = current.get("conditions", "Clear")
        result = {"available": True, "provider": "Visual Crossing", "location": data.get("resolvedAddress") or station_name,
                  "condition": condition, "description": condition,
                  "temperature_c": current.get("temp"), "model_state": weather_state(condition),
                  "observed_at": current.get("datetimeEpoch")}
    except (requests.RequestException, ValueError):
        # Do not return provider URLs or request details: those can contain
        # the API credential as a query parameter.
        result = {"available": False, "model_state": "Clear", "message": "Weather status is temporarily unavailable."}
    _WEATHER_CACHE[cache_key] = (now, result)
    return result
