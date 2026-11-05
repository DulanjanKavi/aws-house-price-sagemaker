"""Step 2: run src/preprocessing.py as a SageMaker Processing job."""
from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.sklearn.processing import SKLearnProcessor

import config


def main():
    processor = SKLearnProcessor(
        framework_version=config.FRAMEWORK_VERSION,
        role=config.get_role(),
        instance_type=config.INSTANCE_TYPE,
        instance_count=1,
        base_job_name="house-price-preprocess",
        sagemaker_session=config.get_session(),
    )
    processor.run(
        code=str(config.SRC_DIR / "preprocessing.py"),
        inputs=[
            ProcessingInput(
                source=config.s3_uri("raw/housing.csv"),
                destination="/opt/ml/processing/input",
            )
        ],
        outputs=[
            ProcessingOutput(
                output_name="train",
                source="/opt/ml/processing/output/train",
                destination=config.s3_uri("processed/train"),
            ),
            ProcessingOutput(
                output_name="test",
                source="/opt/ml/processing/output/test",
                destination=config.s3_uri("processed/test"),
            ),
        ],
        arguments=["--test-size", "0.2"],
    )


if __name__ == "__main__":
    main()
