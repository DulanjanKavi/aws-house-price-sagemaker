"""SageMaker Training job (script mode, scikit-learn container).

SageMaker passes hyperparameters as command-line arguments, exposes input data
channels via SM_CHANNEL_<NAME> and expects the model in SM_MODEL_DIR.
"""
import argparse
import glob
import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

TARGET = "MedHouseVal"


def load_csv_dir(path: str) -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(path, "*.csv")))
    if not files:
        raise FileNotFoundError(f"No CSV files found in {path}")
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    # Hyperparameters (names match the dict passed to the estimator)
    parser.add_argument("--n_estimators", type=int, default=300)
    parser.add_argument("--max_depth", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=0.1)
    parser.add_argument("--validation_size", type=float, default=0.2)
    # SageMaker environment
    parser.add_argument("--model-dir", default=os.environ.get("SM_MODEL_DIR", "model"))
    parser.add_argument("--train", default=os.environ.get("SM_CHANNEL_TRAIN", "data/train"))
    parser.add_argument("--test", default=os.environ.get("SM_CHANNEL_TEST", "data/test"))
    args, _ = parser.parse_known_args(argv)
    return args


def log_to_mlflow(args, features, metrics, model):
    """Log params, metrics and the model to MLflow. Skipped unless MLFLOW_TRACKING_ARN
    is set, and a tracking failure never stops training."""
    tracking_uri = os.environ.get("MLFLOW_TRACKING_ARN")
    if not tracking_uri:
        return
    try:
        import mlflow
        import mlflow.sklearn

        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(os.environ.get("MLFLOW_EXPERIMENT_NAME", "house-price"))
        with mlflow.start_run(run_name=os.environ.get("TRAINING_JOB_NAME")):
            mlflow.log_params(
                {
                    "n_estimators": args.n_estimators,
                    "max_depth": args.max_depth,
                    "learning_rate": args.learning_rate,
                    "validation_size": args.validation_size,
                    "n_features": len(features),
                }
            )
            mlflow.log_metrics({f"test_{k}" if k != "validation_rmse" else k: v
                                for k, v in metrics.items()})
            mlflow.sklearn.log_model(model, "model")
        print("Logged run to MLflow")
    except Exception as exc:  # tracking must not break training
        print(f"WARNING: MLflow logging failed: {exc}")


def main(argv=None):
    args = parse_args(argv)

    train_df = load_csv_dir(args.train)
    test_df = load_csv_dir(args.test)
    features = [c for c in train_df.columns if c != TARGET]

    # Hold out part of the training data for validation. Tuning optimizes this metric,
    # so the test set stays untouched and gives an honest final score.
    fit_df, val_df = train_test_split(
        train_df, test_size=args.validation_size, random_state=42
    )

    model = GradientBoostingRegressor(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        random_state=42,
    )
    model.fit(fit_df[features], fit_df[TARGET])

    val_preds = model.predict(val_df[features])
    validation_rmse = float(np.sqrt(mean_squared_error(val_df[TARGET], val_preds)))
    print(f"validation_rmse={validation_rmse:.4f}")

    preds = model.predict(test_df[features])
    metrics = {
        "validation_rmse": validation_rmse,
        "rmse": float(np.sqrt(mean_squared_error(test_df[TARGET], preds))),
        "mae": float(mean_absolute_error(test_df[TARGET], preds)),
        "r2": float(r2_score(test_df[TARGET], preds)),
    }
    
    for name, value in metrics.items():
        if name != "validation_rmse":
            print(f"test_{name}={value:.4f}")

    log_to_mlflow(args, features, metrics, model)

    os.makedirs(args.model_dir, exist_ok=True)
    joblib.dump(model, os.path.join(args.model_dir, "model.joblib"))
    with open(os.path.join(args.model_dir, "features.json"), "w") as f:
        json.dump(features, f)
    with open(os.path.join(args.model_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    main()