"""Step 4: deploy the trained model as a SageMaker endpoint (uses src/inference.py)."""
import argparse

from sagemaker.serverless import ServerlessInferenceConfig
from sagemaker.sklearn.model import SKLearnModel

import config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-data", 
    )
    parser.add_argument("--serverless", action="store_true", )
    args = parser.parse_args()

    model_data = args.model_data
    if not model_data and config.STATE_FILE.exists():
        model_data = config.STATE_FILE.read_text().strip()
    if not model_data:
        raise SystemExit("No model found. Run run_training.py or pass --model-data.")

    model = SKLearnModel(
        model_data=model_data,
        role=config.get_role(),
        entry_point="inference.py",
        source_dir=str(config.SRC_DIR),
        framework_version=config.FRAMEWORK_VERSION,
        py_version="py3",
        sagemaker_session=config.get_session(),
    )

    options = {"endpoint_name": config.ENDPOINT_NAME}
    if args.serverless:
        options["serverless_inference_config"] = ServerlessInferenceConfig(
            memory_size_in_mb=2048, max_concurrency=5
        )
    else:
        options.update(initial_instance_count=1, instance_type=config.INSTANCE_TYPE)

    model.deploy(**options)
    print(f"Endpoint '{config.ENDPOINT_NAME}' is InService.")


if __name__ == "__main__":
    main()
