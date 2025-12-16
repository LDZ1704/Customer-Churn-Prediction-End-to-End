"""
Train churn prediction models on Telco dataset and save the best pipeline.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

try:
    from xgboost import XGBClassifier
except ImportError:  # optional dependency
    XGBClassifier = None  # type: ignore


DATA_PATH = Path("WA_Fn-UseC_-Telco-Customer-Churn.csv")
TARGET_COL = "Churn"
ID_COLS = ["customerID"]


def load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {path}")
    df = pd.read_csv(path)
    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' missing in dataset")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"].replace(" ", np.nan), errors="coerce")
    df = df.dropna(subset=[TARGET_COL])
    df[TARGET_COL] = df[TARGET_COL].map({"Yes": 1, "No": 0})
    return df


def split_features_labels(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    feature_cols = [c for c in df.columns if c not in ID_COLS + [TARGET_COL]]
    X = df[feature_cols]
    y = df[TARGET_COL].astype(int)
    return X, y


def build_preprocess(X: pd.DataFrame) -> Tuple[ColumnTransformer, List[str], List[str]]:
    num_cols = X.select_dtypes(include=["int64", "float64", "int32", "float32"]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", MinMaxScaler()),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocess = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, num_cols),
            ("cat", categorical_transformer, cat_cols),
        ]
    )
    return preprocess, num_cols, cat_cols


def build_models(random_state: int) -> List[Tuple[str, object]]:
    models: List[Tuple[str, object]] = [
        (
            "log_reg",
            LogisticRegression(
                max_iter=600,
                class_weight="balanced",
                solver="lbfgs",
            ),
        ),
        (
            "random_forest",
            RandomForestClassifier(
                n_estimators=400,
                max_depth=None,
                min_samples_split=2,
                min_samples_leaf=1,
                n_jobs=-1,
                random_state=random_state,
                class_weight="balanced",
            ),
        ),
    ]

    if XGBClassifier is not None:
        models.append(
            (
                "xgboost",
                XGBClassifier(
                    n_estimators=500,
                    learning_rate=0.05,
                    max_depth=5,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    eval_metric="logloss",
                    random_state=random_state,
                    n_jobs=-1,
                    scale_pos_weight=1.0,
                ),
            )
        )
    return models


def evaluate_model(pipeline: Pipeline, X_val: pd.DataFrame, y_val: pd.Series) -> Dict[str, float | str]:
    proba = pipeline.predict_proba(X_val)[:, 1]
    preds = (proba >= 0.5).astype(int)
    return {
        "accuracy": float(accuracy_score(y_val, preds)),
        "f1": float(f1_score(y_val, preds)),
        "roc_auc": float(roc_auc_score(y_val, proba)),
        "classification_report": classification_report(y_val, preds),
    }


def save_artifacts(
    pipeline: Pipeline,
    metrics: Dict[str, float | str],
    num_cols: List[str],
    cat_cols: List[str],
    best_model_name: str,
    models_dir: Path,
) -> None:
    models_dir.mkdir(exist_ok=True)
    model_path = models_dir / "best_model.joblib"
    meta_path = models_dir / "metadata.json"

    joblib.dump(pipeline, model_path)
    metadata = {
        "best_model": best_model_name,
        "metrics": {k: v for k, v in metrics.items() if k != "classification_report"},
        "columns": {"numerical": num_cols, "categorical": cat_cols},
        "target": TARGET_COL,
        "threshold": 0.5,
    }
    meta_path.write_text(json.dumps(metadata, indent=2))
    (models_dir / "classification_report.txt").write_text(str(metrics["classification_report"]))

    print(f"\nSaved model to {model_path}")
    print(f"Saved metadata to {meta_path}")


def main(args: argparse.Namespace) -> None:
    df = load_data(Path(args.data))
    df = clean_data(df)
    X, y = split_features_labels(df)
    preprocess, num_cols, cat_cols = build_preprocess(X)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state, stratify=y
    )

    candidates = build_models(args.random_state)
    print(f"Training {len(candidates)} candidate models...")

    best_result: Dict[str, object] = {"roc_auc": -np.inf}
    for name, estimator in candidates:
        model = Pipeline(steps=[("preprocess", preprocess), ("model", estimator)])
        model.fit(X_train, y_train)
        metrics = evaluate_model(model, X_val, y_val)

        print(f"\n== {name} ==")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"F1      : {metrics['f1']:.4f}")
        print(f"ROC-AUC : {metrics['roc_auc']:.4f}")

        if metrics["roc_auc"] > best_result["roc_auc"]:
            best_result = {
                "name": name,
                "pipeline": model,
                "metrics": metrics,
                "roc_auc": metrics["roc_auc"],
            }

    if "pipeline" not in best_result:
        raise RuntimeError("No models were trained successfully.")

    save_artifacts(
        pipeline=best_result["pipeline"],  # type: ignore[arg-type]
        metrics=best_result["metrics"],  # type: ignore[arg-type]
        num_cols=num_cols,
        cat_cols=cat_cols,
        best_model_name=best_result["name"],  # type: ignore[arg-type]
        models_dir=Path("models"),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train churn prediction models.")
    parser.add_argument("--data", type=str, default=str(DATA_PATH), help="Path to CSV dataset.")
    parser.add_argument("--test-size", type=float, default=0.2, help="Validation split size.")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed.")
    parsed_args = parser.parse_args()
    main(parsed_args)

