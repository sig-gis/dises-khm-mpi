# DISES MPI Predictions Project

This project estimates the Multidimensional Poverty Index (MPI) or Indicators in Cambodia using Small Area Estimation (SAE) and Bayesian geostatistical modeling. The workflow integrates data from the DHS and NIS, performs covariate selection, trains spatial models, and generates prediction surfaces and uncertainty maps.

## Main Files and Their Purpose

- **0-DHS-NIS-complete.ipynb**  
  Prepares and harmonizes the NIS and DHS datasets. Loads raw data, merges with geographic information, and exports shapefiles for further analysis.

- **1. data-download.ipynb**  
  Downloads required data from Google Drive using the Google Drive API. Handles authentication and recursive folder/file downloads.

- **2. Covariates Selection.ipynb**  
  Performs covariate selection for the MPI model. Cleans and transforms geospatial data, applies Lasso regression for feature selection, and saves the processed dataset and report for modeling.

- **3. MBG Model Training.ipynb**  
  Trains a Bayesian Model-Based Geostatistics (MBG) model using the selected covariates. Handles data transformation, model fitting, posterior predictive checks, residual analysis, and saves model outputs and predictions.

- **4. Final Results.ipynb**  
  Processes and visualizes the final prediction surfaces and uncertainty maps. Reverts predictions to the original scale, evaluates model precision, and exports results for reporting and sharing.

- **functions.py**  
  Contains utility functions used throughout the notebooks, such as data transformation, plotting, interpolation, and PDF export helpers.

- **scaler_y.pkl**  
  Stores the fitted scaler object for the target variable, used to revert predictions to the original scale.

- **Comments.txt**  
  Contains project notes, decisions, and discussion points relevant to the modeling process.

- **README.txt**  
  (This file) Explains the project structure and the purpose of each main file.

## Data and Output Folders

- **data/**  
  Contains raw and processed data files.

- **temp_files/**  
  Stores intermediate files, reports, and model outputs generated during the workflow.

- **global-layers-integration/**  
  Handles the integration of additional GIS layers for covariate construction.

- **optional_notebooks/**  
  Contains supplementary or legacy notebooks for reference.

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

Follow the instructions in the original README or use the following commands:

```sh
conda create -c conda-forge -n pymc_env "pymc>=5"
conda activate pymc_env
conda install geopandas rasterio fiona seaborn scikit-gstat ipywidgets
pip install openpyxl scikit-gstat
```

For more details, see the [PyMC installation guide](https://www.pymc.io/projects/docs/en/stable/installation.html).

---

For questions about specific functions, see
