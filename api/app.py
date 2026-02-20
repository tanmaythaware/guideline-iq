"""
FastAPI application entrypoint for Guideline IQ.

Run with:
    uvicorn api.app:app --reload
"""

from __future__ import annotations

import logging

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address


# Limiter is created in app module and reused by routes.
limiter = Limiter(key_func=get_remote_address)

from .routes import lifespan, router


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("guideline_iq")


app = FastAPI(lifespan=lifespan)
app.state.limiter = limiter
app.include_router(router)


@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(_, __):
    return JSONResponse(
        status_code=429,
        content={"detail": "Rate limit exceeded. Please try again later."},
    )


__all__ = ["app"]

