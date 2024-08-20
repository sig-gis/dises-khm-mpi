To install the conda env: 

conda create -c conda-forge -n pymc_env "pymc>=5"
conda activate pymc_env
conda install geopandas rasterio fiona seaborn scikit-gstat ipywidgets

Or follow the instructions on: https://www.pymc.io/projects/docs/en/stable/installation.html

Notebooks structure: 

- Notebook 0 downloads all the layers necessary to build a covariate matrix 
- Notebook 1 integrates more covariates into a single shape file
- Notebook 2 selects covariates and filters out urban locations
- Notebook 3 loads the covariates, run the Bayesian model and analyses results

functions.py file contains functions used throughout
