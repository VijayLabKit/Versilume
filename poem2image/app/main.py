from __future__ import annotations

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

app = FastAPI(
    title="Versilume API",
    description="Poem to image generation pipeline using multi-agent NLP analysis and diffusion models.",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api/v1")


@app.api_route("/", methods=["GET", "HEAD"])
async def root() -> dict:
    return {"service": "versilume", "status": "online", "docs": "/docs"}
