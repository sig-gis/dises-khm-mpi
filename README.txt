# DISES MPI Predictions Project

This project estimates the Multidimensional Poverty Index (MPI) in Cambodia using Small Area Estimation (SAE) and Bayesian geostatistical modeling. The workflow integrates data from the DHS and NIS, performs covariate selection, trains spatial models, and generates prediction surfaces and uncertainty maps.

## Main Files and Their Purpose

- **0-DHS-NIS-complete.ipynb**  
  Prepares and harmonizes the NIS and DHS datasets.  
  - Loads raw NIS and DHS data.
  - Merges with geographic information (village and cluster shapefiles).
  - Filters and cleans data (e.g., removes clusters with near-zero coordinates, keeps only rural clusters).
  - Exports harmonized shapefiles for further analysis.

- **1. data-download.ipynb**  
  Downloads required data from Google Drive using the Google Drive API.
  - Handles authentication and recursive folder/file downloads.
  - Ensures all necessary raw data is available locally.

- **2. Covariates Selection.ipynb**  
  Performs covariate selection for the MPI model.
  - Cleans and transforms geospatial data.
  - Handles missing values with interpolation.
  - Removes all DHS variables except the target indicator.
  - Generates and transforms covariates (including pairwise interactions).
  - Applies Lasso regression for feature selection.
  - Saves the processed dataset and a report for modeling.

- **3. MBG Model Training.ipynb**  
  Trains a Bayesian Model-Based Geostatistics (MBG) model using the selected covariates.
  - Loads selected features and target variable.
  - Transforms and standardizes data.
  - Fits a spatial Bayesian model using PyMC.
  - Performs posterior predictive checks, residual analysis, and uncertainty quantification.
  - Saves model outputs and predictions.

- **4. Final Results.ipynb**  
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

- **data/**  
  Contains raw and processed data files.

- **temp_files/**  
  Stores intermediate files, reports, and model outputs generated during the workflow.

- **predictions/**  
  Stores final prediction shapefiles for sharing and reporting.

- **compared_reports/**  
  Stores precision and comparison reports for different model runs.

## Workflow Overview

1. **Download Data:**  
   Use `1. data-download.ipynb` to fetch all required datasets from Google Drive.

2. **Data Preparation:**  
   Run `0-DHS-NIS-complete.ipynb` to harmonize and export the NIS and DHS datasets.

3. **Covariate Selection:**  
   Use `2. Covariates Selection.ipynb` to select and transform relevant covariates.

4. **Model Training:**  
   Train the Bayesian geostatistical model in `3. MBG Model Training.ipynb`.

5. **Results and Visualization:**  
   Generate final prediction maps and reports with `4. Final Results.ipynb`.

## Environment Setup

Recommended conda environment setup:
```sh
conda create -c conda-forge -n pymc_env "pymc>=5"
conda activate pymc_env
conda install geopandas rasterio fiona seaborn scikit-gstat ipywidgets
pip install openpyxl scikit-gstat
```

For more details, see the [PyMC installation guide](https://www.pymc.io/projects/docs/en/stable/installation.html).

---

For questions about specific functions, see
