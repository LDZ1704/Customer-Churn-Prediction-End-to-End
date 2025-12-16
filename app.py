"""
FastAPI app for serving churn predictions.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

MODELS_DIR = Path("models")
MODEL_PATH = MODELS_DIR / "best_model.joblib"
META_PATH = MODELS_DIR / "metadata.json"


class PredictRequest(BaseModel):
    inputs: List[Dict[str, Any]]


def load_artifacts():
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Model file not found. Train the model first.")
    pipeline = joblib.load(MODEL_PATH)
    metadata = json.loads(META_PATH.read_text()) if META_PATH.exists() else {}
    return pipeline, metadata


app = FastAPI(title="Churn Prediction API", version="1.0.0")


@app.on_event("startup")
def _load_model():
    global MODEL_PIPELINE, MODEL_META
    MODEL_PIPELINE, MODEL_META = load_artifacts()


@app.get("/health")
async def health():
    return {"status": "ok", "model": MODEL_META.get("best_model", "unknown")}


@app.post("/predict")
async def predict(payload: PredictRequest):
    if not payload.inputs:
        raise HTTPException(status_code=400, detail="No inputs provided.")
    df = pd.DataFrame(payload.inputs)
    proba = MODEL_PIPELINE.predict_proba(df)[:, 1]
    threshold = MODEL_META.get("threshold", 0.5)
    labels = (proba >= threshold).astype(int).tolist()
    return {
        "predictions": labels,
        "probabilities": proba.tolist(),
        "model": MODEL_META.get("best_model"),
        "threshold": threshold,
    }

