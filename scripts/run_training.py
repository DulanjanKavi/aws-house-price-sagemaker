"""Step 3: train the model with src/train.py as a SageMaker Training job."""
from sagemaker.inputs import TrainingInput
from sagemaker.sklearn.estimator import SKLearn

import config


def mlflow_environment():
    """Pass the MLflow settings to the training container (empty = MLflow off)."""
    if not config.MLFLOW_TRACKING_ARN:
        return None
    return {
        "MLFLOW_TRACKING_ARN": config.MLFLOW_TRACKING_ARN,
        "MLFLOW_EXPERIMENT_NAME": config.MLFLOW_EXPERIMENT_NAME,
    }


def main():
    estimator = SKLearn(
        entry_point="train.py",
        source_dir=str(config.SRC_DIR),
        dependencies=["requirements.txt"],
        framework_version=config.FRAMEWORK_VERSION,
        py_version="py3",
        role=config.get_role(),
        instance_type=config.TRAINING_INSTANCE_TYPE,
        instance_count=1,
        hyperparameters={"n_estimators": 300, "max_depth": 4, "learning_rate": 0.1},
        metric_definitions=[
            {"Name": "validation:rmse", "Regex": r"validation_rmse=([0-9\.]+)"},
            {"Name": "test:rmse", "Regex": r"test_rmse=([0-9\.]+)"},
            {"Name": "test:mae", "Regex": r"test_mae=([0-9\.]+)"},
            {"Name": "test:r2", "Regex": r"test_r2=(-?[0-9\.]+)"},
        ],
        output_path=config.s3_uri("model-artifacts"),
        base_job_name="house-price-train",
        environment={
            "MLFLOW_TRACKING_ARN": config.MLFLOW_TRACKING_ARN,
            "MLFLOW_EXPERIMENT_NAME": config.MLFLOW_EXPERIMENT_NAME,
        },
        # ----------------------------------------------------
        
        sagemaker_session=config.get_session(),
    )
    estimator.fit(
        {
            "train": TrainingInput(config.s3_uri("processed/train"), content_type="text/csv"),
            "test": TrainingInput(config.s3_uri("processed/test"), content_type="text/csv"),
        }
    )
    config.STATE_FILE.write_text(estimator.model_data)
    print(f"Model artifact: {estimator.model_data}")


if __name__ == "__main__":
    main()