"""FastAPI server exposing the RentWise agent to the Svelte frontend.

Contract is fixed by src/lib/types.ts:
    POST /api/ask {"query": "..."} -> {answer, locations[], sources[]}

`citations` and `trace` are additive -- the current frontend ignores unknown fields,
and they are what makes the architecture visible during a demo.
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Load backend/.env before the backend reads RENTWISE_* configuration.
load_dotenv(Path(__file__).resolve().parent / ".env")

from .agent import RentWiseAgent  # noqa: E402 - must follow load_dotenv
from .llm import build_backend  # noqa: E402
from .retrieval import RentWiseIndex  # noqa: E402

logging.basicConfig(
    level=os.environ.get("RENTWISE_LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("rentwise")

state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("opening index...")
    index = RentWiseIndex()
    backend = build_backend()
    state["agent"] = RentWiseAgent(index, backend)
    state["index"] = index
    log.info(
        "ready: backend=%s dense=%s", backend.name, index.dense_enabled
    )
    yield
    index.close()


app = FastAPI(title="RentWise Boston", version="0.1.0", lifespan=lifespan)

# The Vite dev server proxies /api, so same-origin in dev. CORS is here for the case
# where the frontend is served from a different origin (preview build, phone testing).
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("RENTWISE_CORS_ORIGINS", "*").split(","),
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


class AskRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)


@app.get("/api/health")
def health() -> dict:
    index: RentWiseIndex = state["index"]
    agent: RentWiseAgent = state["agent"]
    counts = {
        table: index.con.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
        for table in ("rentsmart", "str_eligibility", "property_cards")
    }
    return {
        "status": "ok",
        "llm_backend": agent.backend.name,
        "dense_retrieval": index.dense_enabled,
        "tables": counts,
    }


@app.post("/api/ask")
def ask(req: AskRequest) -> dict:
    agent: RentWiseAgent = state["agent"]
    question = req.query.strip()
    if not question:
        raise HTTPException(status_code=400, detail="query must not be empty")
    log.info("ask: %s", question)
    try:
        result = agent.ask(question)
    except Exception as exc:  # noqa: BLE001 - return a usable message to the UI
        log.exception("ask failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    log.info(
        "answered in %sms via %s tool call(s)",
        result.trace.get("elapsed_ms"),
        len(result.trace.get("tool_calls", [])),
    )
    return result.to_response()
