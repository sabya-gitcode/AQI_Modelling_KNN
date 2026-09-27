
# Air Quality Index Estimation – Applied Machine Learning Workflow

This repository explores estimating the Air Quality Index (AQI) from measured pollutant concentrations and city information using K-nearest neighbors (KNN) regression. The focus is a clear, reproducible workflow: handling missing values, encoding categorical data, scaling numerical features, and evaluating the model on held-out records.

The model estimates AQI from readings in the same record. It is **not a forecast of future air quality**.

## Project Overview

The notebook implements a regression workflow that includes:

- Inspecting the dataset and missing-value percentages
- Removing records without a known AQI target
- Splitting the remaining records into training and test sets
- Encoding `City` with one-hot encoding
- Filling missing pollutant readings using medians learned from the training set
- Standardizing pollutant features so KNN can compare distances meaningfully
- Training a KNN regressor and evaluating predictions on held-out test data
- Saving the fitted model together with its preprocessing transformer

The main learning point is that the imputer, encoder, and scaler are fitted on **training data only**. The same fitted transformations are then applied to the test data.

## Model Evaluated

This project trains one `KNeighborsRegressor` using scikit-learn's default settings. KNN estimates a record's AQI by finding nearby records in the transformed feature space and averaging their target values. The notebook does not compare multiple algorithms or tune the number of neighbors.

## Evaluation Approach

The dataset is split into 80% training and 20% test records with `random_state=42`. The held-out test set contains **4,970 records**.

| Metric | Test result |
| --- | ---: |
| Mean absolute error (MAE) | 25.22 AQI points |
| Mean squared error (MSE) | 2,497.04 |
| Root mean squared error (RMSE) | 49.97 AQI points |
| R² | 0.864 |
| Adjusted R² | 0.863 |

MAE indicates an average absolute difference of about 25 AQI points between the model's estimates and recorded values in this test split. The results describe this dataset and split; they are not a guarantee of performance on new monitoring data.

## Dataset and Preprocessing

`AQI.csv` has **29,531 daily records** across **26 Indian cities**. After removing records without a measured AQI, **24,850 records** remain. `AQI` is the numerical target, and the model uses these inputs:

`City`, `PM2.5`, `NO`, `NO2`, `NOx`, `CO`, `SO2`, `O3`, and `Benzene`.

The notebook excludes `Date`, `PM10`, `NH3`, `Toluene`, `Xylene`, and `AQI_Bucket`. Columns with high missingness were omitted for this exercise. `AQI_Bucket` represents a category derived from AQI and would disclose information about the target if used as an input.

Missing pollutant readings are filled with each feature's **training-set median**. The categorical city values are one-hot encoded, and the numeric features are standardized before fitting KNN.


## Limitations and Scope

This is an exploratory portfolio project. Its limits include:

- A single random train–test split, without cross-validation
- One untuned KNN model and no simple baseline for comparison
- A random split that can put records from the same cities and nearby dates in both sets; it does not measure forecasting performance on later dates
- Dropped pollutant columns that may contain useful information if their missing values are addressed differently

A useful next step would be a time-based holdout, followed by comparisons with a baseline and other regression models.

## Run the Notebook

1. Place `AQI.csv` alongside `AQI_Model.ipynb`, or upload it to the notebook's working directory in Colab.
2. Install dependencies: `pip install pandas numpy matplotlib seaborn scipy scikit-learn notebook`.
3. Open `AQI_Model-2.ipynb` in Jupyter or Colab and run all cells from the top.

The final cell writes `finalized_model.sav`, which contains both the fitted KNN model and fitted preprocessing transformer.

## Intended Audience and Use

This repository demonstrates practical handling of missing inputs, categorical encoding, feature scaling, and held-out evaluation in a beginner regression project. It is shared for learning and portfolio purposes.
