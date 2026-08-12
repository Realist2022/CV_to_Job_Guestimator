"""Backwards-compatible entry point. The app now lives in src.api.app.

Keeps `uvicorn src.web_app:app --reload` working.
"""

from src.api.app import app

__all__ = ["app"]