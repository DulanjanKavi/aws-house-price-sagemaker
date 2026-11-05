"""Step 5: send a prediction request to the live endpoint with plain boto3."""
import json

import boto3

import config

# First row of the California Housing dataset (true value is about $452,600)
SAMPLE_HOUSE = {
    "MedInc": 8.3252, "HouseAge": 41, "AveRooms": 6.984127, "AveBedrms": 1.02381,
    "Population": 322, "AveOccup": 2.555556, "Latitude": 37.88, "Longitude": -122.23,
}


def main():
    runtime = boto3.client("sagemaker-runtime", region_name=config.REGION)
    response = runtime.invoke_endpoint(
        EndpointName=config.ENDPOINT_NAME,
        ContentType="application/json",
        Accept="application/json",
        Body=json.dumps({"instances": [SAMPLE_HOUSE]}),
    )
    print(json.loads(response["Body"].read()))


if __name__ == "__main__":
    main()
