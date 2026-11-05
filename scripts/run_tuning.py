""" hyperparameter tuning with a SageMaker automatic model tuning job.

Runs several training jobs with different hyperparameters, minimizes validation RMSE,
and saves the best model's S3 path so scripts/deploy.py deploys it.
"""
from sagemaker.inputs import TrainingInput
from sagemaker.sklearn.estimator import SKLearn
from sagemaker.tuner import ContinuousParameter, HyperparameterTuner, IntegerParameter

import config


def mlflow_environment():
    """Pass the MLflow settings to the training container (empty = MLflow off)."""
    if not config.MLFLOW_TRACKING_ARN:
        return None
    return {
        "MLFLOW_TRACKING_ARN": config.MLFLOW_TRACKING_ARN,
        "MLFLOW_EXPERIMENT_NAME": config.MLFLOW_EXPERIMENT_NAME,
    }


MAX_JOBS = 25  # number of trials
MAX_PARALLEL_JOBS = 1  


def main():
    estimator = SKLearn(
        entry_point="train.py",
        source_dir=str(config.SRC_DIR),
        framework_version=config.FRAMEWORK_VERSION,
        py_version="py3",
        role=config.get_role(),
        instance_type=config.TRAINING_INSTANCE_TYPE,
        instance_count=1,
        output_path=config.s3_uri("tuning-artifacts"),
        sagemaker_session=config.get_session(),
        environment={
            "MLFLOW_TRACKING_ARN": config.MLFLOW_TRACKING_ARN,
            "MLFLOW_EXPERIMENT_NAME": config.MLFLOW_EXPERIMENT_NAME,
        },
    )

    tuner = HyperparameterTuner(
        estimator=estimator,
        objective_metric_name="validation:rmse",
        objective_type="Minimize",
        metric_definitions=[
            {"Name": "validation:rmse", "Regex": r"validation_rmse=([0-9\.]+)"},
            {"Name": "test:rmse", "Regex": r"test_rmse=([0-9\.]+)"},
        ],
        hyperparameter_ranges={
            "n_estimators": IntegerParameter(100, 500),
            "max_depth": IntegerParameter(2, 8),
            "learning_rate": ContinuousParameter(0.01, 0.3),
        },
        strategy="Bayesian",
        max_jobs=MAX_JOBS,
        max_parallel_jobs=MAX_PARALLEL_JOBS,
        base_tuning_job_name="house-price-tune",
    )

    tuner.fit(
        {
            "train": TrainingInput(config.s3_uri("processed/train"), content_type="text/csv"),
            "test": TrainingInput(config.s3_uri("processed/test"), content_type="text/csv"),
        }
    )

    best = tuner.describe()["BestTrainingJob"]
    print("Best training job:", best["TrainingJobName"])
    print("Best hyperparameters:", best["TunedHyperParameters"])
    print("Best validation RMSE:", best["FinalHyperParameterTuningJobObjectiveMetric"]["Value"])

    results = tuner.analytics().dataframe()
    results_path = config.ROOT / "docs" / "tuning_results.csv"
    results.to_csv(results_path, index=False)
    print(f"All trials saved to {results_path}")

    config.STATE_FILE.write_text(tuner.best_estimator().model_data)
    print("Best model path saved. Run scripts/deploy.py to deploy it.")


if __name__ == "__main__":
    main()