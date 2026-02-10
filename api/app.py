"""
FastAPI application entrypoint for Guideline IQ.

Run with:
    uvicorn api.app:app --reload
"""

from __future__ import annotations

import logging

from fastapi import FastAPI

from .routes import lifespan, router


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("guideline_iq")


app = FastAPI(lifespan=lifespan)
app.include_router(router)


__all__ = ["app"]

