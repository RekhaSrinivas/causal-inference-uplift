"""
FastAPI backend for the Causal Inference Dashboard.

Run from app/ directory:  uvicorn main:app --reload
Or from project root:     uvicorn app.main:app --reload
"""
import os
import sys
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(Path(__file__).parent))

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from data_loader import load_data
from causal_engine import run_full_analysis
from llm_engine import generate_causal_summary

app = FastAPI(title="Causal Inference Dashboard", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# cache analysis results in memory (recomputed on restart)
_cache = {}

def _get_results():
    if "results" not in _cache:
        df = load_data()
        _cache["results"] = run_full_analysis(df)
    return _cache["results"]


@app.get("/api/overview")
async def overview():
    return JSONResponse(_get_results()["overview"])


@app.get("/api/propensity")
async def propensity():
    r = _get_results()
    return JSONResponse({
        "ps_treated": r["ps_treated"],
        "ps_control": r["ps_control"],
        "smd_before": {k: round(v, 3) for k, v in r["psm"]["smd_before"].items()},
        "smd_after": {k: round(v, 3) for k, v in r["psm"]["smd_after"].items()},
    })


@app.get("/api/psm")
async def psm():
    r = _get_results()
    return JSONResponse({k: v for k, v in r["psm"].items()
                         if k not in ("smd_before", "smd_after")})


@app.get("/api/uplift")
async def uplift():
    return JSONResponse(_get_results()["uplift"])


@app.get("/api/summary")
async def summary():
    """LLM-generated summary. Cached after first call."""
    if "summary" not in _cache:
        try:
            _cache["summary"] = generate_causal_summary(_get_results())
        except RuntimeError as e:
            raise HTTPException(status_code=503, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    return JSONResponse({"summary": _cache["summary"]})


# serve frontend
FRONTEND = ROOT / "frontend"

@app.get("/")
async def index():
    return FileResponse(FRONTEND / "index.html")

if FRONTEND.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND)), name="static")
