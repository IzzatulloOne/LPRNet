"""
ANPR FastAPI Application
"""
import time
from contextlib import asynccontextmanager
from fastapi import FastAPI
from loguru import logger
import sys

from .routes import router
from .dependencies import init_pipeline

# Логирование в файл
logger.add("logs/anpr.log", rotation="100 MB", level="INFO")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting ANPR System...")
    init_pipeline()
    logger.info("Pipeline ready.")
    yield
    logger.info("Shutdown.")


app = FastAPI(
    title="ANPR System",
    description="Multi-region license plate recognition API",
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(router)
