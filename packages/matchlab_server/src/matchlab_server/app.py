from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from matchlab_server.api import benchmark, configs, datasets, identity_qa, qa, runs, videos
from matchlab_server.db import init_db


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield


app = FastAPI(title="MatchLab API", version="0.1.0", lifespan=lifespan)

# Local dev: the Vite dev server proxies /api, but allow direct calls too.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(videos.router)
app.include_router(configs.router)
app.include_router(datasets.router)
app.include_router(runs.router)
app.include_router(qa.router)
app.include_router(identity_qa.router)
app.include_router(benchmark.router)


@app.get("/api/health")
def health():
    return {"status": "ok"}
