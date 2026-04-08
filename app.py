"""
app.py — FastAPI server exposing CodeReviewEnv as HTTP endpoints.
Required for Hugging Face Spaces deployment.

Endpoints:
  POST /reset        → Observation
  POST /step         → {observation, reward, done, info}
  GET  /state        → state dict
  GET  /health       → {"status": "ok"}
  GET  /tasks        → list of available tasks
"""
from __future__ import annotations

import logging
import os
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from env import CodeReviewEnv, Action, Observation

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="CodeReviewEnv",
    description="OpenEnv-compliant code review environment for AI agents",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# In-memory session store (single-session; fine for evaluation)
# ---------------------------------------------------------------------------

_envs: dict[str, CodeReviewEnv] = {}


def _get_env(session_id: str, task: str = "easy") -> CodeReviewEnv:
    if session_id not in _envs:
        _envs[session_id] = CodeReviewEnv(task=task)
    return _envs[session_id]


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------

class ResetRequest(BaseModel):
    task: str = "easy"
    session_id: str = "default"


class StepRequest(BaseModel):
    action: Action
    session_id: str = "default"


class StepResponse(BaseModel):
    observation: Observation
    reward: float
    done: bool
    info: dict[str, Any]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health() -> dict:
    return {"status": "ok", "environment": "CodeReviewEnv", "version": "1.0.0"}


@app.get("/tasks")
def list_tasks() -> dict:
    return {
        "tasks": ["easy", "medium", "hard"],
        "description": {
            "easy": "Python service with obvious SQL injection and broken crypto",
            "medium": "Express.js API with auth bypass and hardcoded secrets",
            "hard": "Concurrency utilities with subtle race conditions and RCE",
        },
    }


@app.post("/reset", response_model=Observation)
def reset(req: ResetRequest = None) -> Observation:
    """Reset the environment for a given task and return initial observation."""
    if req is None:
        req = ResetRequest()
    if req.task not in ("easy", "medium", "hard"):
        raise HTTPException(status_code=400, detail=f"Unknown task: {req.task}")
    env = CodeReviewEnv(task=req.task)
    _envs[req.session_id] = env
    obs = env.reset()
    logger.info("Reset: session=%s task=%s", req.session_id, req.task)
    return obs


@app.post("/step", response_model=StepResponse)
def step(req: StepRequest) -> StepResponse:
    """Apply an action and return (observation, reward, done, info)."""
    if req.session_id not in _envs:
        raise HTTPException(
            status_code=400,
            detail="Session not found. Call /reset first.",
        )
    env = _envs[req.session_id]
    try:
        obs, reward, done, info = env.step(req.action)
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return StepResponse(observation=obs, reward=reward, done=done, info=info)


@app.get("/state")
def state(session_id: str = "default") -> dict:
    """Return current internal state of the environment."""
    if session_id not in _envs:
        raise HTTPException(status_code=400, detail="Session not found.")
    return _envs[session_id].state()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)
