To install the conda env: 

Follow the instructions on: https://www.pymc.io/projects/docs/en/stable/installation.html

Or: 

conda create -c conda-forge -n pymc_env "pymc>=5"
conda activate pymc_env
conda install geopandas rasterio fiona seaborn scikit-gstat ipywidgets
pip install openpyxl scikit-gstat



Notebooks structure: 

- Notebook 1 downloads the necessary data from Google Drive
	- Google drive API credentials are needed in the form of a client_secrets.json file
- Notebook 2 applies the covariates selection process (Lasso) and dimensionality reduction
- Notebook 3 trains the Bayesian geostatistical model and makes predictions for unknown locations in batches
- Notebook 4 analyses predictions and plots maps 

The process of adding GIS layers is handled in the global-layers-integration folder. There:

- Notebook 1 downloads all the layers necessary to build a covariate matrix 
- Notebook 2 integrates more covariates into a single shape file
- Various .py files are need and included in the folder

functions.py file contains functions used throughout

The optional_notebooks contains some old notebooks that might be handy for other processes. 
