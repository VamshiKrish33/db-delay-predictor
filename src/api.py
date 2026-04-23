"""
FastAPI prediction endpoint for DB Delay Predictor.

Endpoints
---------
POST /predict      — predict delay probability for a given departure
GET  /health       — liveness check
GET  /model/info   — loaded model name, metrics, feature list
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

from src.features import apply_features, load_encoders

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT       = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "best_model.joblib"
ENC_PATH   = ROOT / "models" / "encoders.joblib"
META_PATH  = ROOT / "models" / "best_model_meta.joblib"

# ---------------------------------------------------------------------------
# Shared state (populated at startup)
# ---------------------------------------------------------------------------
_state: dict[str, Any] = {}


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    _state["model"]    = joblib.load(MODEL_PATH)
    _state["encoders"] = load_encoders(ENC_PATH)
    _state["meta"]     = joblib.load(META_PATH)
    yield
    _state.clear()


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="DB Delay Predictor",
    description="Predicts whether a Deutsche Bahn departure will be delayed > 5 min.",
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class PredictRequest(BaseModel):
    station:     str = Field(..., examples=["Köln Hbf"])
    train_type:  str = Field(..., examples=["ICE"])
    hour:        int = Field(..., ge=0, le=23)
    day_of_week: int = Field(..., ge=0, le=6, description="0=Monday … 6=Sunday")

    @field_validator("train_type")
    @classmethod
    def normalise_train_type(cls, v: str) -> str:
        return v.strip().upper()

    @field_validator("station")
    @classmethod
    def normalise_station(cls, v: str) -> str:
        return v.strip().title()


class PredictResponse(BaseModel):
    station:           str
    train_type:        str
    hour:              int
    day_of_week:       int
    delay_probability: float
    prediction:        str   # "on_time" | "delayed"


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    model    = _state.get("model")
    encoders = _state.get("encoders")
    if model is None or encoders is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # features.py expects "station_name" column
    row = pd.DataFrame([{
        "station_name": req.station,
        "train_type":   req.train_type,
        "hour":         req.hour,
        "day_of_week":  req.day_of_week,
    }])
    X = apply_features(row, encoders)

    prob       = float(model.predict_proba(X)[0, 1])
    prediction = "delayed" if int(model.predict(X)[0]) == 1 else "on_time"

    return PredictResponse(
        station=req.station,
        train_type=req.train_type,
        hour=req.hour,
        day_of_week=req.day_of_week,
        delay_probability=round(prob, 4),
        prediction=prediction,
    )


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/model/info")
def model_info() -> dict[str, Any]:
    meta = _state.get("meta")
    if meta is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {
        "model_name": meta.get("name"),
        "metrics":    meta.get("metrics"),
        "features":   meta.get("features"),
    }
