"""
CLI utility for churn prediction using the saved model pipeline.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import joblib
import pandas as pd
import typer

MODELS_DIR = Path("models")
MODEL_PATH = MODELS_DIR / "best_model.joblib"
META_PATH = MODELS_DIR / "metadata.json"

app = typer.Typer(help="Predict churn from JSON/CSV using trained model.")


def load_artifacts() -> tuple[Any, Dict[str, Any]]:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Run train.py first.")
    pipeline = joblib.load(MODEL_PATH)
    metadata: Dict[str, Any] = json.loads(META_PATH.read_text()) if META_PATH.exists() else {}
    return pipeline, metadata


def predict_dataframe(df: pd.DataFrame, threshold: float = 0.5) -> List[Dict[str, Any]]:
    pipeline, metadata = load_artifacts()
    proba = pipeline.predict_proba(df)[:, 1]
    labels = (proba >= threshold).astype(int)
    return [
        {
            "churn_probability": float(p),
            "churn_label": int(label),
            "model": metadata.get("best_model", "unknown"),
        }
        for p, label in zip(proba, labels)
    ]


@app.command()
def json(json_path: str, threshold: float = 0.5) -> None:
    """
    Predict from a JSON file.
    The JSON can be an object (single row) or a list of objects (multiple rows).
    """
    data = json.loads(Path(json_path).read_text())
    records = data if isinstance(data, list) else [data]
    df = pd.DataFrame(records)
    results = predict_dataframe(df, threshold=threshold)
    typer.echo(json.dumps(results, indent=2))


@app.command()
def csv(csv_path: str, threshold: float = 0.5) -> None:
    """
    Predict from a CSV file containing the same columns as training data (except target/id).
    """
    df = pd.read_csv(csv_path)
    results = predict_dataframe(df, threshold=threshold)
    typer.echo(json.dumps(results, indent=2))


if __name__ == "__main__":
    app()

