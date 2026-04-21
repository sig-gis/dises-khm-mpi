# DISES MPI Predictions Project

This project estimates the Multidimensional Poverty Index (MPI) in Cambodia using Small Area Estimation (SAE) and Bayesian geostatistical modeling. The workflow integrates data from the DHS and NIS, performs covariate selection, trains spatial models, and generates prediction surfaces and uncertainty maps.

## Main Files and Their Purpose

- **1. Covariates Selection.ipynb**  
  Prepares and selects geospatial covariates for the MPI model.
  - Loads and cleans georeferenced DHS and NIS data.
  - Handles missing values using interpolation.
  - Removes all DHS variables except the target indicator.
  - Generates and transforms covariates (including pairwise interactions).
  - Applies Lasso regression for feature selection.
  - Saves the processed dataset and a report for modeling.

- **2. MBG Model Training.ipynb**  
  Trains a Bayesian Model-Based Geostatistics (MBG) model using the selected covariates.
  - Loads selected features and target variable.
  - Transforms and standardizes data.
  - Fits a spatial Bayesian model using PyMC.
  - Performs posterior predictive checks, residual analysis, and uncertainty quantification.
  - Saves model outputs and predictions.

- **3. Final Results.ipynb**  
  Processes and visualizes the final prediction surfaces and uncertainty maps.
  - Reverts predictions to the original scale.
  - Merges predictions with spatial data.
  - Generates and saves surface and uncertainty maps (with option to overlay country boundaries).
  - Evaluates model precision on unseen data and exports results for reporting.

- **functions.py**  
  Contains utility functions used throughout the notebooks, such as:
  - Data transformation and reversion.
  - Plotting distributions and missing values.
  - Interpolation and PDF export helpers.

- **scaler_y.pkl**  
  Stores the fitted scaler object for the target variable, used to revert predictions to the original scale.

- **README.txt**  
  (This file) Explains the project structure and the purpose of each main file.

## Data and Output Folders

- **full_panel_merged/**  
  Contains input processed data files.

- **temp_files/**  
  Stores intermediate files, reports, and model outputs generated during the workflow.

- **predictions/**  
  Stores final prediction shapefiles for sharing and reporting.

- **compared_reports/**  
  Stores precision and comparison reports for different model runs.

## Workflow Overview

1. **Covariate Selection:**  
   Use `1. Covariates Selection.ipynb` to select and transform relevant covariates.

2. **Model Training:**  
   Train the Bayesian geostatistical model in `2. MBG Model Training.ipynb`.

3. **Results and Visualization:**  
   Generate final prediction maps and reports with `3. Final Results.ipynb`.

## Environment Setup

Create new environment by using the [environment.yml file](environment.yml):
```sh
conda env create --file environment.yml
```
The new environment will be called "dises". Activate environment using:
```sh
conda activate dises
```

** Do not create the environment "dises" if it was already created for running scripts from the [DISES SAE repo](https://github.com/sig-gis/dises-sae/tree/adjusted).


For more details, see the [PyMC installation guide](https://www.pymc.io/projects/docs/en/stable/installation.html).

---

For questions about specific functions, see `functions.py`.
