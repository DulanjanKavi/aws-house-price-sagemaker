"""Step 1: download the California Housing dataset and upload it to S3."""
import boto3
from sklearn.datasets import fetch_california_housing

import config


def main():
    df = fetch_california_housing(as_frame=True).frame
    raw_dir = config.DATA_DIR / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    local_path = raw_dir / "housing.csv"
    df.to_csv(local_path, index=False)
    print(f"Saved {len(df)} rows to {local_path}")

    key = f"{config.PREFIX}/raw/housing.csv"
    boto3.client("s3", region_name=config.REGION).upload_file(
        str(local_path), config.BUCKET, key
    )
    print(f"Uploaded to s3://{config.BUCKET}/{key}")


if __name__ == "__main__":
    main()
