To install the conda env: 

conda create -c conda-forge -n pymc_env "pymc>=5"
conda activate pymc_env
conda install geopandas rasterio fiona seaborn scikit-gstat ipywidgets

Or follow the instructions on: https://www.pymc.io/projects/docs/en/stable/installation.html

Notebooks structure: 

- Notebook 1 downloads the necessary data from Google Drive
	- Google drive API credentials are needed in the form of a client_secrets.json file
- Notebook 2 applies the covariates selection process and dimensionality reduction
- Notebook 3 trains the Bayesian geostatistical model 
- Notebook 4 makes predictions for unknown locations in batches
- Notebook 5 analyses predictions and plots maps

The previous process of adding layers is handled in the global-layers folder:

- Notebook 1 downloads all the layers necessary to build a covariate matrix 
- Notebook 2 integrates more covariates into a single shape file

functions.py file contains functions used throughout
