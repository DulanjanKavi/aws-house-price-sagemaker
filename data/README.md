# Data

The project uses the **California Housing** dataset bundled with scikit-learn
(`sklearn.datasets.fetch_california_housing`). `scripts/upload_data.py` downloads it
and uploads it to S3, so no manual download or Kaggle account is needed.

| Column | Meaning |
|---|---|
| MedInc | Median income in the block group (tens of thousands of $) |
| HouseAge | Median house age |
| AveRooms / AveBedrms | Average rooms / bedrooms per household |
| Population | Block group population |
| AveOccup | Average household members |
| Latitude / Longitude | Location |
| MedHouseVal (target) | Median house value in units of $100,000 |

`data/raw/` is git-ignored. `data/sample/sample.csv` is **synthetic** data with the same
columns, included only so the code can be smoke-tested without AWS. Do not use it to
report model results.
