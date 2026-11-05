"""Inference handlers for the SageMaker scikit-learn serving container.

The container calls: model_fn -> input_fn -> predict_fn -> output_fn.
Clients send the 8 RAW features; feature engineering is applied here so it
matches training exactly.
"""
import io
import json
import os

import joblib
import pandas as pd

RAW_FEATURES = [
    "MedInc", "HouseAge", "AveRooms", "AveBedrms",
    "Population", "AveOccup", "Latitude", "Longitude",
]
DOLLARS_PER_UNIT = 100_000  # the model predicts in units of $100,000


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Must match preprocessing.add_features (a test enforces this)."""
    df = df.copy()
    df["RoomsPerPerson"] = df["AveRooms"] / df["AveOccup"]
    df["BedroomRatio"] = df["AveBedrms"] / df["AveRooms"]
    return df


def model_fn(model_dir):
    model = joblib.load(os.path.join(model_dir, "model.joblib"))
    with open(os.path.join(model_dir, "features.json")) as f:
        features = json.load(f)
    return {"model": model, "features": features}


def input_fn(request_body, content_type="application/json"):
    if isinstance(request_body, (bytes, bytearray)):
        request_body = request_body.decode("utf-8")

    if content_type == "application/json":
        payload = json.loads(request_body)
        if isinstance(payload, dict) and "instances" in payload:
            records = payload["instances"]
        else:
            records = payload
        if isinstance(records, dict):
            records = [records]
        df = pd.DataFrame(records)
    elif content_type == "text/csv":
        df = pd.read_csv(io.StringIO(request_body), header=None, names=RAW_FEATURES)
    else:
        raise ValueError(f"Unsupported content type: {content_type}")

    missing = [c for c in RAW_FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required features: {missing}")

    df = df[RAW_FEATURES].astype(float)
    return add_features(df)


def predict_fn(input_data, model_bundle):
    return model_bundle["model"].predict(input_data[model_bundle["features"]])


def output_fn(prediction, accept="application/json"):
    prices = [round(float(p) * DOLLARS_PER_UNIT) for p in prediction]
    return json.dumps({"predictions_usd": prices}), "application/json"
