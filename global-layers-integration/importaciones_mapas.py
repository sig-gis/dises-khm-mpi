#Se importan los paquetes basico necesarios
import pandas as pd
import geopandas as gpd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import numpy as np
from IPython.display import Markdown as md #Este paquete sirve para escribir párrafos que se modifican automáticamente
import plotly
import plotly.express as px
import plotly.graph_objects as go
from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets
import mapclassify
import contextily as ctx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib as mpl
from shapely.geometry import box
import xarray as xr

#Se centran los outputs para que después haya un mejor display
from IPython.display import display, HTML

CSS = """
.output {
    align-items: center;
}
"""

HTML('<style>{}</style>'.format(CSS))

import warnings
warnings.filterwarnings('ignore')

#Esto garantiza que se puedan ver los objetos de plotly
from plotly.offline import download_plotlyjs, init_notebook_mode, iplot
init_notebook_mode(connected=True)


#sys.prefix
#sys.path.append('/Users/Daniel/OneDrive - C- ANALISIS SAS/programacion/funciones/')

#Estilos para pandas
pd.options.display.float_format = '{:,.2f}'.format

import mapas

