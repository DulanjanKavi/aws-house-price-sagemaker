import os
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  
    pass

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
DATA_DIR = ROOT / "data"
STATE_FILE = ROOT / ".last_model_uri"  

REGION = os.environ.get("AWS_REGION", "us-east-1")
BUCKET = os.environ.get("S3_BUCKET", "")
ROLE_ARN = os.environ.get("SAGEMAKER_ROLE_ARN", "")
PREFIX = os.environ.get("S3_PREFIX", "house-price").strip("/")
ENDPOINT_NAME = os.environ.get("ENDPOINT_NAME", "house-price-endpoint")

# Default instance type, used when a stage-specific one is not set
INSTANCE_TYPE = os.environ.get("INSTANCE_TYPE", "ml.m5.large")
# Per-stage instance types 
PROCESSING_INSTANCE_TYPE = os.environ.get("PROCESSING_INSTANCE_TYPE", INSTANCE_TYPE)
TRAINING_INSTANCE_TYPE = os.environ.get("TRAINING_INSTANCE_TYPE", INSTANCE_TYPE)
ENDPOINT_INSTANCE_TYPE = os.environ.get("ENDPOINT_INSTANCE_TYPE", INSTANCE_TYPE)


FRAMEWORK_VERSION = os.environ.get("SKLEARN_FRAMEWORK_VERSION", "1.2-1")

MLFLOW_TRACKING_ARN = os.environ.get("MLFLOW_TRACKING_ARN", "")
MLFLOW_EXPERIMENT_NAME = os.environ.get("MLFLOW_EXPERIMENT_NAME", "house-price")


def s3_uri(*parts: str) -> str:
    if not BUCKET:
        raise RuntimeError("Set S3_BUCKET (see .env.example).")
    return "s3://" + "/".join([BUCKET, PREFIX, *[p.strip("/") for p in parts]])


def get_role() -> str:
    if ROLE_ARN:
        return ROLE_ARN
    try:  
        import sagemaker

        return sagemaker.get_execution_role()
    except Exception as exc:
        raise RuntimeError("Set SAGEMAKER_ROLE_ARN (see .env.example).") from exc


def get_session():
    import boto3
    import sagemaker

    return sagemaker.Session(
        boto_session=boto3.Session(region_name=REGION),
        default_bucket=BUCKET or None,
    )
