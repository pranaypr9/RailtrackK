"""Compatibility entry point for the RailConnect FastAPI application.

Run the API with:
    python -m uvicorn api.main:app --reload --port 8001
"""

from .main_ALL_TRAINS import app

__all__ = ["app"]


