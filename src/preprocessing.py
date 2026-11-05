"""SageMaker Processing job: clean raw housing data, engineer features, split train/test.

Inside a Processing job, SageMaker copies the S3 input to /opt/ml/processing/input,
runs this script, then uploads /opt/ml/processing/output/* back to S3.
All paths are arguments so the script can also be run locally (see tests).
"""
import argparse
import glob
import os

import pandas as pd
from sklearn.model_selection import train_test_split

TARGET = "MedHouseVal"
RAW_FEATURES = [
    "MedInc", "HouseAge", "AveRooms", "AveBedrms",
    "Population", "AveOccup", "Latitude", "Longitude",
]
ENGINEERED = ["RoomsPerPerson", "BedroomRatio"]
FEATURES = RAW_FEATURES + ENGINEERED


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Drop missing values, duplicates and unrealistic rows."""
    missing = [c for c in RAW_FEATURES + [TARGET] if c not in df.columns]
    if missing:
        raise ValueError(f"Input data is missing columns: {missing}")

    df = df[RAW_FEATURES + [TARGET]].drop_duplicates().dropna()
    df = df[
        df["AveRooms"].between(1, 20)
        & df["AveBedrms"].between(0.5, 5)
        & df["AveOccup"].between(0.5, 10)
    ]
    
    df = df[df[TARGET] < 5.0]
    return df.reset_index(drop=True)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Feature engineering. NOTE: inference.py has the same function.
    Keep both in sync (tests/test_inference.py checks they match)."""
    df = df.copy()
    df["RoomsPerPerson"] = df["AveRooms"] / df["AveOccup"]
    df["BedroomRatio"] = df["AveBedrms"] / df["AveRooms"]
    return df


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="/opt/ml/processing/input")
    parser.add_argument("--train-dir", default="/opt/ml/processing/output/train")
    parser.add_argument("--test-dir", default="/opt/ml/processing/output/test")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args(argv)

    files = sorted(glob.glob(os.path.join(args.input_dir, "*.csv")))
    if not files:
        raise FileNotFoundError(f"No CSV files found in {args.input_dir}")

    raw = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df = add_features(clean(raw))[FEATURES + [TARGET]]
    print(f"Rows: raw={len(raw)} after cleaning={len(df)}")

    train, test = train_test_split(
        df, test_size=args.test_size, random_state=args.random_state
    )

    os.makedirs(args.train_dir, exist_ok=True)
    os.makedirs(args.test_dir, exist_ok=True)
    train.to_csv(os.path.join(args.train_dir, "train.csv"), index=False)
    test.to_csv(os.path.join(args.test_dir, "test.csv"), index=False)
    print(f"Saved train={len(train)} rows, test={len(test)} rows")


if __name__ == "__main__":
    main()
