# api/main.py
"""FastAPI application entrypoint for the PPE safety-detection service."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.dependencies import load_detector
from api.middleware.logging import LoggingMiddleware
from api.routers.predict import router as predict_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Warm-loading PPE detector...")
    load_detector()
    logger.info("PPE detector loaded, API ready.")
    yield


app = FastAPI(
    title="PPE Safety Detection API",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(LoggingMiddleware)

app.include_router(predict_router)
