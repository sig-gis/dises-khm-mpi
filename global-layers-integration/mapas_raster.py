#Program to create maps with rasters
from rasterstats import zonal_stats, point_query
import numpy as np
import os
import imageio
import matplotlib.pyplot as plt
import geopandas as gpd
import pandas as pd

import subprocess
from osgeo import gdal

#Estos paquetes son para la transformación de los archivos .nc en raster o GeoTiff
import netCDF4 as nc
import xarray as xr
import rioxarray

import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from mpl_toolkits.axes_grid1 import make_axes_locatable

import xarray
import contextily as ctx
import rasterio as rio
import fiona
from rasterio import mask
import seaborn as sns

from shapely.geometry import box
from shapely.geometry import mapping, shape
from shapely.geometry import Point, MultiPolygon
from shapely import affinity

from rasterio.plot import show
import numpy as np
from fiona.crs import from_epsg
import glob
import tempfile

from matplotlib.lines import Line2D
from matplotlib.patches import Patch 
from mpl_toolkits.axes_grid1 import make_axes_locatable
from datetime import date

import numpy as np
import rasterio
from rasterio.transform import from_origin
from rasterio.enums import Resampling
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.crs import CRS
from rasterio.io import MemoryFile
import pandas as pd

#Crear un diccionario con todas las rutas a la información necesaria

def diccionario_rutas(paises):
    """
    Retorna diccionario con las rutas a los shapes de las fronteras de los países y una lista con las llaves del diccionario.
    --------
    Parámetros: 
        Diccionario con las abreviasciones de 3 letras y los nombres de los países.
    """

    rutas = {}

    #Divipola

    for i in ['HND', 'GTM', 'SLV', 'NIC']:
        rutas[i] = {}
        for j in [1, 2]:
            path = 'MMC - General - FIDA/Data/{}_adm/{}_adm{}.shp'.format(
                i, i, j)
            gdf_name = 'divipola'+str(j)
            rutas[i][gdf_name] = path
            
    #Proyectos del fida
        rutas[i]['proyectos_fida_shape_nuevo'] = 'MMC - General - FIDA/Data/fida-proyectos-shapes/{}-proyectos.shp'.format(i)
    #Nombre del país
        rutas[i]['nom'] = paises[i]
    
    #Población
        rutas[i]['pob'] = 'MMC - General - FIDA/Data/population_{}_2018-10-01/population_{}_2018-10-01.tif'.format(i.lower(), i.lower())
        rutas[i]['pob_fixed'] = 'MMC - General - FIDA/Data/population_{}_2018-10-01/population_{}_fixed_2018-10-01.tif'.format(i.lower(), i.lower())
    
    
    #PIB
    #Archivo con las estimaciones de PIB PPP para todo el mundo 1990-2015
        rutas[i]['pib'] = 'MMC - General - FIDA/Data/PIB_per_capita/GDP_PPP_1990_2015_5arcmin_v2.nc'
        
    #Altitud y topografía
        rutas[i]['NASADEM'] = 'MMC - General - FIDA/Data/DEM/{}_output_NASADEM.tif'.format(i)
        rutas[i]['NASADEM_fixed'] = 'MMC - General - FIDA/Data/DEM/{}_output_NASADEM_fixed.tif'.format(i)
        
        #Copernicus data
        rutas[i]['COPDEM'] = 'MMC - General - FIDA/Data/DEM/{}_output_COP30.tif'.format(i)
        
    #Acceso a ciudades de más de 50k habitantes
        rutas[i]['access'] = 'MMC - General - FIDA/Data/travel_time_to_cities_11.tif'
        rutas[i]['access_fixed'] = 'MMC - General - FIDA/Data/travel_time_to_cities_11_fixed.tif'
        
    #Shape con los proyectos FIDA desde 1984 en los países del corredor seco
        rutas[i]['fida'] = 'MMC - General - FIDA/Data/fida-ni_sv_hn_gt_20230515_drycorridor/ni_sv_hn_gt_20230515_drycorridor.shp'
    
    #Shape con las áreas del corredor seco según el estudio de Quesada (2019)
        rutas[i]['corredor_seco'] = 'MMC - General - FIDA/Data/fao_corredor_seco/rlc_corredorseco-corredor_seco_fao.json'
    
    #Raster con la información de Hansen Tree Cover para los 4 países del corredor seco
        rutas[i]['hansen'] = '/Users/Daniel/Library/CloudStorage/GoogleDrive-dwiesner@veredata.co/My Drive/imagenes_earth_engine/hsns_treecover2000_corredor.tif'
        rutas[i]['hansen_fixed'] = '/Users/Daniel/Library/CloudStorage/GoogleDrive-dwiesner@veredata.co/My Drive/imagenes_earth_engine/hsns_treecover2000_corredor_fixed.tif'
    
    #Archivo excel con proyectos del FIDA
        rutas[i]['proyectos_fida'] = 'MMC - General - FIDA/Data/proyectos-fida-corredor-seco.xlsx'
    
    #Archivo de excel con proyectos fida y seleccón manual de municipos
        rutas[i]['proyectos_fida_manual'] = 'MMC - General - FIDA/Data/municipios_fida_manual.xlsx'
        
        rutas[i]['proyectos_fida_shape'] = 'MMC - General - FIDA/Data/fida-proyectos-shapes/{}-proyectos.shp'.format(i)
    
    return rutas, list(rutas.keys())

#Funcion para cambiar los valores no data de los raster
def fix_no_data_value_nasadem(input_file, output_file, no_data_value=np.nan):
    with rasterio.open(input_file, "r+") as src:
        src.nodata = no_data_value
        with rasterio.open(output_file, 'w',  **src.profile) as dst:
            for i in range(1, src.count + 1):
                band = src.read(i)
                band = np.where(band<0,no_data_value,band)
                dst.write(band,i)
                
def count_intersections(gdf1, gdf2, new_column_name):
    """
    This function takes two GeoDataFrames and counts how many polygons from the second GeoDataFrame
    intersect with each polygon in the first GeoDataFrame. 

    Parameters:
    - gdf1 (GeoDataFrame): The primary GeoDataFrame.
    - gdf2 (GeoDataFrame): The secondary GeoDataFrame from which we want to count intersections.
    - new_column_name (str): The name of the new column in gdf1 that will store the count of intersections.

    Returns:
    - GeoDataFrame: The modified gdf1 with an additional column showing the count of intersections.
    """

    def count_intersections_for_row(row):
        # Define the geometry for the current row in gdf1
        geom = row['geometry']

        # Count the number of intersections with polygons in gdf2
        return gdf2[gdf2.intersects(geom)].shape[0]

    # Apply the function to every row in gdf1 to get counts of intersections
    gdf1[new_column_name] = gdf1.apply(count_intersections_for_row, axis=1)

    return gdf1

def create_geo_dataframe(dataframe, lat_column, lon_column, correct_commas=False, crs="WGS84"):
    """
    Create a GeoDataFrame from a DataFrame with latitude and longitude columns.

    Parameters:
        dataframe (pd.DataFrame): Input pandas DataFrame.
        lat_column (str): Name of the column containing latitude values.
        lon_column (str): Name of the column containing longitude values.

    Returns:
        gpd.GeoDataFrame: GeoDataFrame with Point geometries.
    """
    # Ensure the input columns exist in the DataFrame
    if lat_column not in dataframe.columns or lon_column not in dataframe.columns:
        raise ValueError(f"Columns '{lat_column}' and '{lon_column}' must exist in the DataFrame.")
        
    if correct_commas == True: 
        dataframe[lat_column] = dataframe[lat_column].astype(str).apply(lambda x: x.replace(',','.')).astype(float)
        dataframe[lon_column] = dataframe[lon_column].astype(str).apply(lambda x: x.replace(',','.')).astype(float)

    # Create Point geometries from the latitudes and longitudes
    geometry = [Point(lon, lat) for lat, lon in zip(dataframe[lat_column], dataframe[lon_column])]

    # Create a GeoDataFrame
    geo_df = gpd.GeoDataFrame(dataframe, geometry=geometry, crs=crs)

    return geo_df

### Rasters

def check_raster_crs(raster_path):
    """
    Check the Coordinate Reference System (CRS) of a raster.

    Parameters:
    raster_path (str): Path to the raster file.

    Returns:
    crs (str): CRS of the raster.
    """
    with rio.open(raster_path) as src:
        crs = src.crs

    return crs

def raster_clipping(shape_path, raster_path, file_name):
    """
    This function clips a raster image to the bounding box of a given shapefile. 

    Parameters:
    shape_path (str): Path to the shapefile (.shp) used to clip the raster.
    raster_path (str): Path to the raster file that will be clipped.
    file_name (str): Name of the output file where the resulting clipped raster will be stored.

    Returns:
    cropped_raster_path (str): Path to the resulting clipped raster.

    The function uses fiona to open and read the shapefile. The shapefile should be
    in the form of a .shp file. The shapes from the shapefile are extracted and stored.

    rasterio is used to open the raster file and to perform the masking. The raster is
    masked according to the bounding boxes of the shapes from the shapefile, and the
    image outside these bounding boxes is cropped.

    The metadata of the original raster image is updated to reflect the changes in size
    and the transformation matrix after cropping.

    The updated raster data is then written to a new raster file with the specified 
    output file name. This file will be saved in the same format (GTiff) as the original 
    raster file. 

    Finally, the function prints and returns the path to the newly created raster file.
    """

    with fiona.open(shape_path, "r") as shapefile:
        shapes = [feature["geometry"] for feature in shapefile]

    with rio.open(raster_path) as src:
        out_image, out_transform = mask.mask(src, shapes, crop=True)
        out_meta = src.meta 
    
    out_meta.update({"driver": "GTiff",
                     "height": out_image.shape[1],
                     "width": out_image.shape[2],
                     "transform": out_transform})

    cropped_raster_path = file_name
    with rio.open(cropped_raster_path, "w", **out_meta) as dest:
        dest.write(out_image)

    print(cropped_raster_path)
    return cropped_raster_path

def raster_clipping(shape_path, raster_path, file_name):
    """
    Clipping a raster image to the shape of a given shapefile and ensuring CRS compatibility.
    """
    # Load the shapefile using GeoPandas to easily handle CRS
    gdf = gpd.read_file(shape_path)

    with rio.open(raster_path) as src:
        raster_crs = src.crs
        # Check CRS compatibility, reproject if necessary
        if gdf.crs != raster_crs:
            print(f"Reprojecting shapefile from {gdf.crs} to {raster_crs}")
            gdf = gdf.to_crs(raster_crs)

        # Extract shapes from the GeoDataFrame for masking
        shapes = [feature["geometry"] for _, feature in gdf.iterrows()]

        # Perform the masking operation
        out_image, out_transform = mask.mask(src, shapes, crop=True)
        out_meta = src.meta

    # Update metadata for the output file
    out_meta.update({"driver": "GTiff",
                     "height": out_image.shape[1],
                     "width": out_image.shape[2],
                     "transform": out_transform})

    # Define the path for the cropped raster
    cropped_raster_path = file_name
    # Write the cropped raster to file
    with rio.open(cropped_raster_path, "w", **out_meta) as dest:
        dest.write(out_image)

    #print(f"Cropped raster saved to: {cropped_raster_path}")
    return cropped_raster_path


def replace_no_data(raster_path, output_path):
    """
    Replace no data values in a raster file with numpy's null (np.nan).

    Parameters:
    raster_path (str): Path to the input raster file.
    output_path (str): Path to the output raster file where the result will be stored.

    The function uses rasterio to open the raster file. It reads the raster data and
    replaces no data values with np.nan.

    The modified raster data is then written to a new raster file with the specified 
    output file name. This file will be saved in the same format as the original 
    raster file.
    """

    # Open the raster file
    with rasterio.open(raster_path) as src:
        # Read the raster data
        raster_data = src.read(1)

        # Replace no data values with np.nan
        no_data = src.nodata
        if no_data is not None:
            raster_data[raster_data == no_data] = np.nan

        # Update the metadata for the new raster file
        out_meta = src.meta.copy()
        out_meta.update({'nodata': np.nan})

    # Write the modified raster data to the new file
    with rasterio.open(output_path, 'w', **out_meta) as dest:
        dest.write(raster_data, 1)


### Mapas

def plot_raster_with_boundaries(raster_path, 
                                shapefile_path, 
                                title, 
                                attribution, 
                                nodata_color='black', 
                                boundary_color='red', 
                                colormap='viridis', 
                                colorbar_label='Values', 
                                png_filepath=None):
    """
    Plots a raster image with shapefile boundaries overlaid.

    Parameters:
    raster_path (str): Path to the .tif raster file.
    shapefile_path (str): Path to the shapefile.
    title (str): The title of the plot.
    attribution (str): The attribution for the image.
    nodata_color (str): Color to display No Data pixels.
    boundary_color (str): Color of the shapefile boundaries.
    colormap (str): Colormap used for the raster.
    colorbar_label (str): Label for the colorbar.
    png_filepath (str): The filepath to save the .png file, if desired.
    """
    # Open the raster file
    with rio.open(raster_path) as src:
        raster = src.read(1)
        raster_extent = [src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top]
        nodata_value = src.nodata  # Get the No Data value from the raster's metadata
        crs = src.crs.to_dict()  # Convert the CRS to a dictionary

    # Open the shapefile
    gdf = gpd.read_file(shapefile_path)
    
    # Reproject the shapefile to match the raster's CRS
    if gdf.crs != crs:
        gdf = gdf.to_crs(crs)

    # Mask nodata values
    if nodata_value is not None:
        raster = np.ma.masked_where(raster == nodata_value, raster)

    # Create a figure
    fig, ax = plt.subplots(1, figsize=(12, 6))

    # Define colormap
    colors = plt.get_cmap(colormap)(np.linspace(0, 1, 256))
    under_color = mpl.colors.to_rgb(nodata_color)
    colors = mpl.colors.ListedColormap(colors)
    colors.set_under(under_color)
    
    # Normalize object to map colormap to data values correctly, excluding NaN values
    vmin, vmax = np.nanmin(raster), np.nanmax(raster)
    normalize = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

    # Plot the raster
    im = ax.imshow(raster, extent=raster_extent, cmap=colors, norm=normalize)
    im.set_clim(vmin, vmax)

    # Plot the shapefile boundaries
    gdf.boundary.plot(ax=ax, color=boundary_color, linewidth=0.1)

    # Set the title
    ax.set_title(title)

    # Hide the axes
    ax.axis('off')

    # Add the colorbar
    cbar = plt.colorbar(im, fraction=0.046, pad=0.04)  # Adjust colorbar size relative to image
    cbar.set_label(colorbar_label, rotation=270, labelpad=20)  # Add label for colorbar
    
    # Add the attribution
    fig.text(0.65, 0.04, attribution, ha='center')

    # Save the plot as a .png file if png_filepath is provided
    if png_filepath is not None:
        plt.savefig(png_filepath, dpi=300, bbox_inches='tight')

    # Show the plot
    plt.show()
    
## Nueva función


def plot_raster_with_boundaries(raster_path, 
                                shapefile_path, 
                                title, 
                                attribution, 
                                nodata_color='black', 
                                boundary_color='black',
                                treatment_color='blue',
                                colormap='viridis', 
                                colorbar_label='Values', 
                                png_filepath=None):
    """
    Plots a raster image with shapefile boundaries overlaid.

    Parameters:
    raster_path (str): Path to the .tif raster file.
    shapefile_path (str): Path to the shapefile.
    title (str): The title of the plot.
    attribution (str): The attribution for the image.
    nodata_color (str): Color to display No Data pixels.
    boundary_color (str): Color of the shapefile boundaries.
    treatment_color (str): Color of the shapefile for the column "tratado".
    colormap (str): Colormap used for the raster.
    colorbar_label (str): Label for the colorbar.
    png_filepath (str): The filepath to save the .png file, if desired.
    """
    # Open the raster file
    with rio.open(raster_path) as src:
        raster = src.read(1)
        raster_extent = [src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top]
        nodata_value = src.nodata  # Get the No Data value from the raster's metadata
        crs = src.crs.to_dict()  # Convert the CRS to a dictionary

    # Open the shapefile
    gdf = gpd.read_file(shapefile_path)
    
    # Reproject the shapefile to match the raster's CRS
    if gdf.crs != crs:
        gdf = gdf.to_crs(crs)

    # Mask nodata values
    if nodata_value is not None:
        raster = np.ma.masked_where(raster == nodata_value, raster)

    # Create a figure with white background
    fig, ax = plt.subplots(1, figsize=(12, 6))
    fig.patch.set_facecolor('white')

    # Define colormap
    colors = plt.get_cmap(colormap)(np.linspace(0, 1, 256))
    under_color = mpl.colors.to_rgb(nodata_color)
    colors = mpl.colors.ListedColormap(colors)
    colors.set_under(under_color)
    
    # Normalize object to map colormap to data values correctly, excluding NaN values
    vmin, vmax = np.nanmin(raster), np.nanmax(raster)
    normalize = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

    # Plot the raster
    im = ax.imshow(raster, extent=raster_extent, cmap=colors, norm=normalize)
    im.set_clim(vmin, vmax)

    # Plot the shapefile boundaries
    gdf.boundary.plot(ax=ax, color=boundary_color, linewidth=0.1)

    if 'tratado' in gdf.columns:
        
        gdf[gdf['tratado']==0].boundary.plot(ax=ax,color='black', linewidth=0.2)
        gdf[gdf['tratado']==1].boundary.plot(ax=ax,color=treatment_color, linewidth=0.4)
    else:
        gdf.boundary.plot(ax=ax, color=boundary_color, linewidth=0.1)
    

    # Set the title
    ax.set_title(title)

    # Hide the axes
    ax.axis('off')

    # Add the colorbar
    cbar = plt.colorbar(im, fraction=0.046, pad=0.04)  # Adjust colorbar size relative to image
    cbar.set_label(colorbar_label, rotation=270, labelpad=20)  # Add label for colorbar
    
    # Add the attribution
    fig.text(0.65, 0.04, attribution, ha='center')

    # Save the plot as a .png file if png_filepath is provided
    if png_filepath is not None:
        plt.savefig(png_filepath, dpi=300, bbox_inches='tight')

    # Show the plot
    plt.show()

    
#Ajustar la barra de color en algunos casos
def adjusted_blues_cmap(start_val):
    """
    Adjusts the Blues colormap to start with a light blue shade when the
    start value is higher than zero.

    Parameters:
    - start_val: The starting value of your data range.

    Returns:
    - Custom colormap.
    """
    from matplotlib.colors import ListedColormap

    base_blues = plt.cm.get_cmap('Blues', 256)
    new_colors = base_blues(np.linspace(0, 1, 256))
    if start_val > 0:
        new_colors[0] = (0.8, 0.9, 1, 1)  # A light shade of blue
    return ListedColormap(new_colors)

#Mapa de municipios fida
def mapa_municipios(gdf, column, title, colorbar_lab, filename, attribution, cmap='Blues', dpi=300, ruta_corredor_seco='.'):

    """
    Retorna un mapa con los polígonos coloreados según los valores de column. 
    Parameters:
    -----------
        gdf: Geopandas dataframe.
        Column: gdf['Column'] with information for the colour map. 
         colorbar_lab: label para la leyenda del colorbar. 
        dpi: float or 'figure', default: :rc:`savefig.dpi`
            The resolution in dots per inch.  If 'figure', use the figure's
            dpi value.
    """
    #legenda
    
    legend_patches = [] 
    
    # Se crea el plot general
    
    fig, ax = plt.subplots(figsize=(10,10))
    ax.set_axis_off()
    ax.set_title(title)
    
    # Before plotting, adjust the colormap if necessary:
    min_val = gdf[column].min()
    if min_val > 0:
        cmap = adjusted_blues_cmap(min_val)
    
    #Se plotea el geodataframe 
    #gdf.boundary.plot(ax=ax, color='black', linewidth=0.6, alpha=0.3)
    gdf.plot(ax=ax, column=column, cmap=cmap, alpha=0.8,)
    
    #Agergar el corredor seco en rojo

    #Agregar la leyenda
    
    def clip_to_bbox(gdf1, gdf2):
        # Calculate bounding box of the first GeoDataFrame
        bbox = box(*gdf1.total_bounds)

        # Create a new GeoDataFrame with the bounding box
        bbox_gdf = gpd.GeoDataFrame(geometry=[bbox], crs=gdf1.crs)

        # Clip the second GeoDataFrame to the bounding box of the first
        clipped_gdf2 = gpd.clip(gdf2, gdf1)

        return clipped_gdf2
    
    if ruta_corredor_seco != '.': 
        cos = gpd.read_file(ruta_corredor_seco)
        cos = clip_to_bbox(gdf, cos)
        cos['dis'] = True
        cos = cos.dissolve(by='dis')
        cos.boundary.plot(ax=ax, color='r', alpha=0.8)
        cos_legend = Line2D([0], [0], color='r', label='Corredor Seco', lw=2)
        legend_patches = legend_patches + [cos_legend]
    
    #Se intenta agregar el basemap, pero si algo falla (si no hay internet, por ejemplo), entonces se crea el mapa sin basemap.
    try: 
        ctx.add_basemap(ax=ax, source=ctx.providers.CartoDB.Positron, crs=gdf.crs.to_string(), attribution=attribution)
    except: print('No se pudo agregar el mapa base')

        #Agaregar la leyenda
    plt.legend(handles=legend_patches, 
               loc=False,  
               ncols=1,
               frameon=True)
    
    #Color bar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    im = ax.imshow(gdf[[column]], cmap=cmap)
    plt.colorbar(im, cax=cax, orientation='vertical')
    cax.set_ylabel(colorbar_lab, fontsize=12)
    
    #Se guarda el archivo
    plt.savefig(filename, dpi=dpi)
    
    
    return

def mapa_colores_poligonos(gdf, column, title, colorbar_lab, filename, attribution, cmap='Blues', dpi=300, second_gdf=False, gdf2='.', labels_col='.'):

    """
    Retorna un mapa con los polígonos coloreados según los valores de column. 
    Parameters:
    -----------
        gdf: Geopandas dataframe.
        Column: gdf['Column'] with information for the colour map. 
        colorbar_lab: label para la leyenda del colorbar. 
        dpi: float or 'figure', default: :rc:`savefig.dpi`
            The resolution in dots per inch.  If 'figure', use the figure's
            dpi value.
    """
    
    # Se crea el plot general
    
    fig, ax = plt.subplots(figsize=(10,10))
    ax.set_axis_off()
    ax.set_title(title)
    
    # Before plotting, adjust the colormap if necessary:
    min_val = gdf[column].min()
    if min_val > 0:
        cmap = adjusted_blues_cmap(min_val)
    
    #Se plotea el geodataframe 
    #gdf.boundary.plot(ax=ax, color='black', linewidth=0.6, alpha=0.3)
    gdf.plot(ax=ax, column=column, cmap=cmap, alpha=0.8,)
    
    #Agregar el segundo gdf
    if second_gdf == True: 
        # Plot polygons
        gdf2.boundary.plot(ax=ax, color='grey', alpha=0.5)
    
        #Add labels for the polygons_layer
        try:
            gdf2.apply(lambda x: ax.annotate(text=x[labels_col], 
                                   xy=x.geometry.centroid.coords[0], 
                                   ha='center', color='black', 
                                   fontsize=8),axis=1)
        except: print('No se pudo agregar labels al gdf2')

    #Se intenta agregar el basemap, pero si algo falla (si no hay internet, por ejemplo), entonces se crea el mapa sin basemap.
    try: 
        ctx.add_basemap(ax=ax, source=ctx.providers.CartoDB.Positron, crs=gdf.crs.to_string(), attribution=attribution)
    except: print('No se pudo agregar el mapa base')
     
    #Color bar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    im = ax.imshow(gdf[[column]], cmap=cmap)
    plt.colorbar(im, cax=cax, orientation='vertical')
    cax.set_ylabel(colorbar_lab, fontsize=12)
    
    #Se guarda el archivo
    plt.savefig(filename, dpi=dpi)
    
    
    return
    
#[Acá va la revisión]    

def plot_rasters(raster1_path, raster2_path, title, subtitle1, subtitle2, attribution, cmap='gray', save_as=None):
    """
    Plots two raster images side by side.

    Parameters:
    raster1_path (str): Path to the first .tif raster file.
    raster2_path (str): Path to the second .tif raster file.
    title (str): The title of the whole composition.
    subtitle1 (str): The title of the first subplot.
    subtitle2 (str): The title of the second subplot.
    attribution (str): The attribution for the images.
    cmap (str, optional): The colormap to use for the images. Defaults to 'gray'.
    save_as (str, optional): The file path to save the figure as a PNG. If None, the figure is not saved. Defaults to None.
    """

    # Open the raster files
    with rasterio.open(raster1_path) as src1:
        raster1 = src1.read(1)

    with rasterio.open(raster2_path) as src2:
        raster2 = src2.read(1)

    # Create a figure with two subplots (side by side)
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    
    # Plot the rasters
    axs[0].imshow(raster1, cmap=cmap)
    axs[1].imshow(raster2, cmap=cmap)
    
    # Set the titles
    fig.suptitle(title)
    axs[0].set_title(subtitle1)
    axs[1].set_title(subtitle2)
    
    # Hide the axes
    axs[0].axis('off')
    axs[1].axis('off')
    
    # Add the attribution
    fig.text(0.5, 0.04, attribution, ha='center')

    # If save_as parameter is provided, save the figure as a PNG
    if save_as is not None:
        plt.savefig(save_as, format='png')

    # Show the plot
    plt.show()

def cargar_gdf_fronteras(filename):
    """
    Returns a GeoDataFrame with columns filtered. 

    Parameters
    ----------
    filename : str, path object or file-like object
        Either the absolute or relative path to the file or URL to
        be opened, or any object with a read() method (such as an open file
        or StringIO)
    """
    
    #Cargar el gdf con las fronteras administrativas del país en la memoria
    gdf = gpd.read_file(filename) #Leer el gdf
    gdf = gdf[['ISO', 'NAME_0', 'ID_1', 'NAME_1', 'ID_2', 'NAME_2', 'TYPE_2', 'geometry']] #Filtrar las columnas que no son útiles
    return gdf

### Estadísticas zonales

#función para sacar estadíscticas zonales
def get_zonal_stats(gdf_path, raster_path, stat, col_name, band=1):
    """
    Retorna una serie del tamaño del gdf con la estadística requerída.

    Parameters:
    - gdf_path: str, path to the GeoDataFrame (shapefile or GeoJSON)
    - raster_path: str, path to the raster file
    - stat: str, name of the statistic to compute (e.g., 'mean', 'sum')
    - col_name: str, name for the resulting column
    - band: int, optional, raster band to use (default is 1)
    """
    
    # Check CRS match
    with rasterio.open(raster_path) as src:
        raster_crs = src.crs

    gdf1 = gpd.read_file(gdf_path)

    if gdf1.crs != raster_crs:
        gdf1 = gdf1.to_crs(raster_crs)
        gdf1.to_file(gdf_path)

    stats = zonal_stats(
        gdf_path,
        raster_path,
        stats=[stat],
        band=band
    )

    stats_df = pd.DataFrame.from_dict(stats)
    stats_df.columns = [col_name]
    return stats_df.round(2)


#Función para agregar una columna con información a un gdf
def add_column_to_gdf(gdf, s):
    df = pd.concat([gdf, s], axis=1) #Pegar las estadísticas calculadas
    gdf = gpd.GeoDataFrame(df) #Convertir de nuevo en gdf
    return gdf

def collapse_strings(df):
    
    grouped_data = df.groupby([df.columns[0]])
    df['acronym'] = df['acronym'].astype(str)
    df = grouped_data[df.columns[1]].apply(lambda x: ', '.join(x))
    df = pd.DataFrame(data=df).reset_index()
    df.rename(mapper={df.columns[1]:'proyectos'}, axis=1, inplace=True)
    
    #Proyectos en una lista
    df['proyectos'] = df['proyectos'].apply(lambda x: x.split(', '))
    
    #Funcion para el número de proyectos
    def process_series(s):
        def check_list(lst):
            if len(lst) == 1 and lst[0]=='nan':
                return 0
            else:
                return len(lst)
        return s.apply(check_list)
    
    df['n_proyectos'] = process_series(df['proyectos'])
    
    df['fida'] = np.where(df['n_proyectos']>0, True, False)
    
    return df

def save_array_as_raster(reference_raster_path, array, output_raster_path):
    """
    Saves a numpy array as a raster file using the characteristics of a reference raster file.

    Parameters:
    reference_raster_path (str): Path to the reference raster file.
    array (numpy.ndarray): The numpy array to be saved as a raster.
    output_raster_path (str): Path to the output raster file.

    The function opens the reference raster file using rasterio and reads the metadata.
    It then updates the metadata to match the dimensions of the input array.

    The numpy array is then written to the specified output path as a new raster file,
    using the updated metadata.

    """
    print('Guardando el nuevo raster')
    with rasterio.open(reference_raster_path) as src:
        meta = src.meta

    meta.update({
        "dtype": rasterio.float32,
        "count": 1,
        "height": array.shape[0],
        "width": array.shape[1]
    })

    with rasterio.open(output_raster_path, 'w', **meta) as dst:
        dst.write(array.astype(rasterio.float32), 1)

def tree_cover_year_raster(hnsn_path, year, output_raster_path):
    """
    Esta función calcula la cobertura arbórea para un año dado basándose en un conjunto de datos ráster de Hansen. 
    Tiene en cuenta la pérdida ocurrida a lo largo de los años y guarda el resultado como un nuevo archivo ráster.
    
    Parámetros:
    hnsn_path (str): La ruta del archivo del conjunto de datos ráster de Hansen.
    year (int): El año para el cual calcular la cobertura arbórea.
    output_raster_path (str): La ruta del archivo donde se debe guardar el archivo ráster de salida.
    
    Devoluciones:
    str: Un mensaje que indica que el ráster de cobertura arbórea para el año ingresado ha sido guardado.
    """
    
    # Abrir el ráster de Hansen
    hnsn = rio.open(hnsn_path)
    
    # Leer las bandas en el ráster de Hansen
    tc2 = hnsn.read(1)  # Porcentaje del pixel cubierto por árboles en el año 2000
    loss = hnsn.read(2)  # Booleano que indica si el pixel ha sido perdido a lo largo de los años
    lsy = hnsn.read(4)  # Año en el que ocurrió la pérdida (de 0 a 2022)
    
    print('Se leyeron las tres bandas')
    
    y = year - 2000
    lost_to_year = (lsy <= y) & (loss == 1)  # Píxeles perdidos hasta el año ingresado
    tcyear = tc2 * (~(lost_to_year) + 2)  # Multiplicar la inversa para conservar solo los píxeles sin pérdida
    
    print('Se construyó el ráster con la cobertura arbórea para el año')
    
    del lost_to_year, tc2, loss, lsy  # Liberar memoria
    
    save_array_as_raster(hnsn_path, tcyear, output_raster_path)  # Guardar el ráster de salida
    
    del tcyear  # Liberar memoria
    
    return print('El ráster de cobertura arbórea para el año ' + str(year) + ' se guardó en: ' + output_raster_path)

def calculate_tree_cover(hansen_raster_path, gdf, target_year):
    """
    Calculate the tree cover area in hectares for each polygon in a given GeoDataFrame for a specific year.
    
    Parameters:
    - hansen_raster_path (str): The file path to the Hansen Global Forest Change raster data.
    - gdf (GeoDataFrame): A GeoDataFrame containing polygon geometries.
    - target_year (int): The target year for which tree cover needs to be calculated, ranging from 2000 to 2022.
    
    Returns:
    - GeoDataFrame: A new GeoDataFrame with an additional column 'tree_cover_hectares' that contains the 
                    area covered by trees in hectares for each polygon.
    
    Notes:
    - The function assumes that Band 1 of the Hansen raster is the tree cover for the year 2000,
      Band 2 is a loss indicator, and Band 4 indicates the year of loss.
    - The function performs coordinate reference system (CRS) matching between the raster and the GeoDataFrame.
    - The temporary raster 'current_tree_cover.tif' will be generated, which contains the tree cover state
      for the given target year.
    - A conversion factor of 0.00095 is used to convert each pixel to hectares.
    """
    
    if target_year < 2000 or target_year > 2023:
        raise ValueError("Year must be between 2000 and 2022")
        
    target_year = target_year - 2000  # Hansen uses 2-digit years after 2000

    # Check if the CRS of the raster and the gdf match
    with rasterio.open(hansen_raster_path) as src:
        raster_crs = src.crs
        
    if gdf.crs != raster_crs:
        gdf = gdf.to_crs(raster_crs)
        
    # Read bands and get profile
    with rasterio.open(hansen_raster_path) as src:
        raster_profile = src.profile  # Store the profile for later use
        tree_cover_2000 = src.read(1)
        loss = src.read(2)
        loss_year = src.read(4)

    # Calculate current tree cover for the target year
    current_tree_cover = np.where(
        (loss==1) & (loss_year <= (target_year)),
        0,
        tree_cover_2000
    )

    # Save the current tree cover as a temporary raster
    with rasterio.open('current_tree_cover.tif', 'w', **raster_profile) as dst:
        dst.write(current_tree_cover, 1)
        
    # Calculate zonal statistics using the current tree cover
    stats = zonal_stats(
        gdf.geometry,
        'current_tree_cover.tif',
        stats="sum",
        nodata=0
    )
    
    # Convert the stats to a DataFrame and merge it with the original GeoDataFrame
    stats_df = pd.DataFrame(stats)
    stats_df.columns = ['tree_cover_{}'.format(target_year)]
    stats_df = stats_df * 0.00095  # Conversion factor for each pixel to hectares
    gdf_with_stats = pd.concat([gdf, stats_df], axis=1)
    
    return gdf_with_stats



#Listado de acrónimos de proyectos que hacen parte de la evaluación
def extract_word_in_parens(string):
    opening_index = string.find('(')
    closing_index = string.find(')')
    if opening_index == -1 or closing_index == -1:
        return None
    else:
        return string[opening_index + 1:closing_index]
    
def find_rows_by_term(df, column_name, term):
    """
    Filters rows in a DataFrame where a specified column contains a particular word or term.

    Parameters:
    df (pd.DataFrame): The input DataFrame.
    column_name (str): The name of the column to search in.
    term (str): The word or term to search for.

    Returns:
    result (pd.DataFrame): A DataFrame containing the rows where the term was found.
    """
    
    # Filter rows where the specified column contains the term
    result = df[df[column_name].astype(str).str.contains(term, case=False, na=False)]
    
    return result


def merge_rasters(input_folder, output_file):
    """
    This function merges multiple rasters into a single raster. The rasters should all have the same CRS.

    Parameters:
    input_folder (str): The folder where the raster files (.tif) to be merged are located.
    output_file (str): The filepath where the merged raster should be saved.

    Returns:
    None
    """
    # Get all the raster filepaths from the input_folder
    search_criteria = f"{input_folder}/*.tif"
    raster_files = glob.glob(search_criteria)

    # List for the data
    src_files_to_mosaic = []

    # Open raster files and add them to the list
    for file in raster_files:
        src = rio.open(file)
        src_files_to_mosaic.append(src)

    # Merge function returns a single mosaic array and the transformation info
    mosaic, out_trans = merge(src_files_to_mosaic)

    # Copy the metadata from the first raster file in the list
    out_meta = src_files_to_mosaic[0].meta.copy()

    # Update the metadata with new dimensions, transform (affine) and CRS (optional)
    out_meta.update({
        "driver": "GTiff",
        "height": mosaic.shape[1],
        "width": mosaic.shape[2],
        "transform": out_trans,
    })

    # Write the mosaic raster to disk
    with rio.open(output_file, "w", **out_meta) as dest:
        dest.write(mosaic)

    # Close the raster files
    for file in src_files_to_mosaic:
        file.close()

    print(f"Merged raster saved to: {output_file}")

def calculate_area_hectares(gdf, country_code):
    """
    Calculate the area of each polygon in a GeoDataFrame for specific countries in Central America.
    
    Parameters:
    - gdf (GeoDataFrame): Input GeoDataFrame containing polygons.
    - country_code (str): ISO code for the country (e.g., 'HND', 'GTM', 'SLV', 'NIC').
    
    Returns:
    - GeoDataFrame: A new GeoDataFrame with an additional 'area_hectares' column representing 
      the area of each polygon in hectares. The GeoDataFrame will have the same CRS as the input gdf.
    """
    
    # Dictionary mapping country ISO codes to their respective CRS for UTM zones
    crs_mapping_iso = {
        'HND': 'EPSG:32616',
        'GTM': 'EPSG:32615',
        'SLV': 'EPSG:32615',
        'NIC': 'EPSG:32616'
    }
    
    # Store the original CRS for later use
    original_crs = gdf.crs
    
    # Convert the GeoDataFrame to the appropriate CRS for the given country code to calculate the area
    #gdf_projected = gdf.to_crs(crs_mapping_iso[country_code])
    gdf_projected = gdf.to_crs("EPSG:5461")
 
    
    # Calculate the area of each polygon in hectares
    gdf_projected['area_hectares'] = gdf_projected['geometry'].area / 10000
    
    # Reproject back to the original CRS
    gdf_with_area = gdf_projected.to_crs(original_crs)
    
    return gdf_with_area

def fetch_inflation_rates(series_path='.'):
    """
    Fetch the inflation rates from the U.S. Federal Reserve (FRED).

    Returns:
    - dict: A dictionary with years as keys and inflation rates as values.
    """
    if series_path=='.':
        #Descargar nuevo CSV
        url = "https://fred.stlouisfed.org/graph/fredgraph.csv?bgcolor=%23e1e9f0&chart_type=line&drp=0&fo=open%20sans&graph_bgcolor=%23ffffff&height=450&mode=fred&recession_bars=on&txtcolor=%23444444&ts=12&tts=12&width=1138&nt=0&thu=0&trc=0&show_legend=yes&show_axis_titles=yes&show_tooltip=yes&id=FPCPITOTLZGUSA&scale=left&cosd=1960-01-01&coed=2022-01-01&line_color=%234572a7&link_values=false&line_style=solid&mark_type=none&mw=3&lw=2&ost=-99999&oet=99999&mma=0&fml=a&fq=Annual&fam=avg&fgst=lin&fgsnd=2020-02-01&line_index=1&transformation=lin&vintage_date={}-01-01&revision_date={}-01-01&nd=1960-01-01".format(date.today().year, date.today().year)
        df = pd.read_csv(url, parse_dates=["DATE"])
        df["YEAR"] = df["DATE"].dt.year
        df["INFLATION_RATE"] = df["FPCPITOTLZGUSA"]/100

        inflation_rates = df.set_index("YEAR")["INFLATION_RATE"].to_dict()
        return inflation_rates

    if series_path!='.': 
        df = pd.read_csv(series_path, parse_dates=["DATE"])
        df["YEAR"] = df["DATE"].dt.year
        df["INFLATION_RATE"] = df["FPCPITOTLZGUSA"]/100

        inflation_rates = df.set_index("YEAR")["INFLATION_RATE"].to_dict()
        return inflation_rates

def adjust_for_inflation(amount, base_year, target_year, series_path):
    """
    Adjust an amount for inflation.

    Parameters:
    - amount (float): The amount in USD to be adjusted.
    - base_year (int): The year from which the amount is given.
    - target_year (int): The year to which you want to adjust the amount.

    Returns:
    - float: The adjusted amount.
    """
    inflation_rates = fetch_inflation_rates(series_path)
    
    # Check if base_year and target_year are in inflation_rates
    if base_year not in inflation_rates or target_year not in inflation_rates:
        raise ValueError("Base year or target year not found in inflation rates data.")
    
    # Adjust the amount for each year from base_year to target_year
    for year in range(base_year + 1, target_year + 1):
        amount = amount * (1 + inflation_rates[year])
    
    return amount

def get_bounding_box_string(input_data, original_crs, output_crs=None):
    """
    Returns the bounding box coordinates of a GeoDataFrame or a raw geometry in string format separated by commas.
    
    Parameters:
    - input_data: GeoDataFrame or raw geometry whose bounding box will be computed.
    - output_crs: (Optional) EPSG code for the desired output CRS. If not provided, the original CRS of the input_data is used.
    
    Returns:
    - str: Bounding box coordinates in the format "minx, miny, maxx, maxy".
    """
    
    # Check if input_data is a GeoDataFrame
    if isinstance(input_data, gpd.GeoDataFrame):
        gdf = input_data
    # Check if input_data is a raw geometry (e.g., shapely geometry)
    elif hasattr(input_data, 'bounds'):
        gdf = gpd.GeoSeries([input_data], crs=original_crs)
    else:
        raise ValueError("The input_data should be a GeoDataFrame or a raw geometry.")
    
    # Reproject the GeoDataFrame if an output CRS is provided
    if output_crs:
        gdf = gdf.to_crs(output_crs)
    
    # Compute the bounding box
    bbox = gdf.total_bounds
    bbox_string = ','.join(map(str, bbox))
    
    return bbox_string

def jitter_geometries(gdf, gdf1, min_distance, max_distance, max_attempts=100):
    """
    Takes a GeoDataFrame with polygons (gdf), computes their centroids,
    and returns a new GeoDataFrame where polygons are centered 
    around a new point that's between 'min_distance' and 'max_distance' meters from the original centroid,
    without intersecting the original polygons and ensuring they are contained within gdf1 boundaries.

    Parameters:
        gdf: A GeoDataFrame with polygons to be jittered.
        gdf1: A GeoDataFrame with polygons that should contain the jittered geometries.
        min_distance: Minimum distance in meters from the original centroid.
        max_distance: Maximum distance in meters from the original centroid.
        max_attempts: Maximum number of attempts to find a non-intersecting polygon.

    Returns:
        A new GeoDataFrame with "jittered" polygons.
    """
    
    if min_distance > max_distance:
        raise ValueError("Minimum distance should be less than or equal to the maximum distance.")
    
    # Ensure both GeoDataFrames are in a projected coordinate system
    if gdf.crs.is_geographic:
        gdf = gdf.to_crs(epsg=3395)  # Adjust the EPSG code for your area if needed
    
    if gdf1.crs != gdf.crs:
        gdf1 = gdf1.to_crs(gdf.crs)
    
    original_geoms = gdf.geometry.values
    
    # Dissolving all polygons in gdf1 to get the combined boundaries
    boundary = gdf1.unary_union
    
    jittered_geoms = []
    
    for geom in original_geoms:
        centroid = geom.centroid
        
        for _ in range(max_attempts):
            angle = 2 * np.pi * np.random.random()
            r = min_distance + (max_distance - min_distance) * np.random.random()
            new_x = centroid.x + r * np.cos(angle)
            new_y = centroid.y + r * np.sin(angle)
            dx, dy = new_x - centroid.x, new_y - centroid.y
            new_geom = affinity.translate(geom, xoff=dx, yoff=dy)
            
            # Check for intersections with the original and jittered polygons 
            # and ensure the new geometry is within the boundary of gdf1
            if not any(new_geom.intersects(other) for other in list(original_geoms) + jittered_geoms) and boundary.contains(new_geom):
                break
        else:
            raise ValueError(f"Couldn't jitter a geometry after {max_attempts} attempts.")
        
        jittered_geoms.append(new_geom)
    
    gdf['geometry'] = jittered_geoms
    
    # Convert back to original CRS if necessary
    if gdf.crs != gdf.crs:
        gdf = gdf.to_crs(gdf.crs)
    
    return gdf

def calculate_resolution(lats, lons):
    """
    Calculate the resolution based on the average distance between consecutive points.
    
    Parameters:
        lats (array-like): Array-like containing latitude values.
        lons (array-like): Array-like containing longitude values.
    
    Returns:
        float: The inferred resolution based on the average distance between points.
    """
    # Calculate the differences between consecutive latitude and longitude values
    lat_diff = np.diff(lats)
    lon_diff = np.diff(lons)
    
    # Calculate the average absolute differences
    avg_lat_diff = np.mean(np.abs(lat_diff))
    avg_lon_diff = np.mean(np.abs(lon_diff))
    
    # Use the maximum of the two average differences as the resolution
    return max(avg_lat_diff, avg_lon_diff)

def create_raster_from_points(df, lat_col, lon_col, value_col, output_file):
    """
    Create a raster from latitude, longitude, and values in a Pandas DataFrame.
    
    Parameters:
        df (DataFrame): Pandas DataFrame containing latitude, longitude, and values.
        lat_col (str): Name of the column containing latitude values.
        lon_col (str): Name of the column containing longitude values.
        value_col (str): Name of the column containing the values to be rasterized.
        output_file (str): Path to save the output raster file.
    
    Returns:
        None
    """
    # Extract latitude, longitude, and values from the DataFrame
    lats = df[lat_col].values
    lons = df[lon_col].values
    values = df[value_col].values
    
    # Calculate the resolution
    resolution = calculate_resolution(lats, lons)
    
    # Define the extent of the raster
    min_lat, max_lat = min(lats), max(lats)
    min_lon, max_lon = min(lons), max(lons)
    num_rows = int((max_lat - min_lat) / resolution) + 1
    num_cols = int((max_lon - min_lon) / resolution) + 1
    
    # Create an empty array to store raster values
    raster_values = np.zeros((num_rows, num_cols))
    
    # Calculate the index of each point in the raster grid
    row_indices = ((max_lat - lats) / resolution).astype(int)
    col_indices = ((lons - min_lon) / resolution).astype(int)
    
    # Assign values to the corresponding grid cells
    raster_values[row_indices, col_indices] = values
    
    # Define the transformation parameters
    transform = from_origin(min_lon, max_lat, resolution, resolution)
    
    # Write the raster to a GeoTIFF file
    with rasterio.open(
        output_file,
        'w',
        driver='GTiff',
        height=num_rows,
        width=num_cols,
        count=1,
        dtype=raster_values.dtype,
        crs=CRS.from_epsg(4326),  # WGS84 CRS
        transform=transform
    ) as dst:
        dst.write(raster_values, 1)

def create_png_map_with_boundaries_and_crs_check(raster_path, gdf, names_column, base_year=2000, target_year=2020, output_path='map.png'):
    """
    Creates a PNG map displaying tree cover for the year 2000, cumulative tree loss up to a target year,
    and overlays geospatial boundaries from a GeoDataFrame, ensuring matching CRS. Adds legends and labels for GDF polygons.
    
    Parameters:
    - raster_path: Path to the raster file containing the data.
    - gdf: A GeoDataFrame containing the boundaries to overlay on the map.
    - names_column: The column in the GDF that contains the names of the polygons.
    - base_year: The base year for the dataset (default is 2000 for the 'treecover2000' band).
    - target_year: The target year up to which to visualize cumulative tree loss.
    - output_path: Path to save the output PNG map.
    """
    # Open the raster file
    with rasterio.open(raster_path) as src:
        # Check and match CRS
        raster_crs = src.crs
        if gdf.crs != raster_crs:
            print(f"Reprojecting GDF from {gdf.crs} to {raster_crs}")
            gdf = gdf.to_crs(raster_crs)
        
        # Assuming band 1 is tree cover and band 2 is loss (year of loss encoded)
        tree_cover = src.read(1)
        loss_year = src.read(4)
        
        # Adjusting for the encoding where 1 corresponds to 2001, hence target_year - 2000
        loss_mask = ((loss_year >= 1) & (loss_year <= (target_year - 2000))).astype(np.uint8)*255
        print(target_year)
        
        # Get the extent of the raster
        extent = [src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top]
        
        # Prepare the figure
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot the tree cover
        tree_cover_layer = ax.imshow(tree_cover, cmap='Greens', extent=extent)
        
        # Overlay the loss mask
        loss_layer = ax.imshow(loss_mask, cmap='Reds', alpha=0.5, extent=extent)
        
        # Overlay the GeoDataFrame boundaries and labels
        gdf.boundary.plot(ax=ax, color='grey', linewidth=1, alpha=0.5)
        for idx, row in gdf.iterrows():
            ax.annotate(text=row[names_column], xy=(row.geometry.centroid.x, row.geometry.centroid.y),
                        horizontalalignment='center', fontsize=5, color='grey')
        
        # Create legends
        tree_cover_legend = mpatches.Patch(color='green', label=f'Bosque en {target_year}')
        loss_legend = mpatches.Patch(color='red', label=f'Pérdida')
        plt.legend(handles=[tree_cover_legend, loss_legend], loc='lower left')
        
        # Set the title and axes labels
        ax.set_title(f'Pérdida de bosque 2000 - {target_year}')
        
        # Remove x and y ticks
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Save the figure
        plt.savefig(output_path, bbox_inches='tight')
        plt.close()
        
        print(f"Map has been saved to '{output_path}'")



def create_virtual_raster(input_folder, output_vrt_path):
    """
    Creates a virtual raster (VRT) from all raster files in a given folder.

    Parameters:
    - input_folder: Path to the folder containing the raster files.
    - output_vrt_path: Path where the output VRT file will be saved.

    Returns:
    - None
    """
    # Ensure GDAL is installed and available
    if not gdal.VersionInfo():
        raise ImportError("GDAL library is not found. Please install GDAL to use this function.")
    
    # Find all raster files in the input folder
    # This example assumes raster files have extensions like .tif or .tiff
    # Modify the tuple as necessary to include other raster file types
    raster_files = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.endswith(('.tif', '.tiff'))]
    
    if not raster_files:
        print("No raster files found in the specified folder.")
        return
    
    # Create the VRT using the gdalbuildvrt command
    command = ['gdalbuildvrt', output_vrt_path] + raster_files
    try:
        subprocess.run(command, check=True)
        print(f"Virtual raster created successfully: {output_vrt_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error in creating virtual raster: {e}")

def create_deforestation_map(raster_path, 
         gdf, 
         names_column, 
         name_of_area,
         base_year=2000, 
         target_year=2020, 
         output_path='map.png',
        basemap=True):
    """
    Creates a PNG map displaying tree cover for the year 2000, cumulative tree loss up to a target year,
    and overlays geospatial boundaries from a GeoDataFrame, ensuring matching CRS. Adds legends and labels for GDF polygons.
    
    Parameters:
    - raster_path: Path to the raster file containing the data.
    - gdf: A GeoDataFrame containing the boundaries to overlay on the map.
    - names_column: The column in the GDF that contains the names of the polygons.
    - name_of_area: The name of the area that appears by the title.
    - base_year: The base year for the dataset (default is 2000 for the 'treecover2000' band).
    - target_year: The target year up to which to visualize cumulative tree loss.
    - output_path: Path to save the output PNG map.
    """
    # Open the raster file
    with rasterio.open(raster_path) as src:
        # Check and match CRS
        raster_crs = src.crs
        if gdf.crs != raster_crs:
            print(f"Reprojecting GDF from {gdf.crs} to {raster_crs}")
            gdf = gdf.to_crs(raster_crs)
        
        # Assuming band 1 is tree cover and band 2 is loss (year of loss encoded)
        tree_cover = src.read(1)
        loss_year = src.read(4)
        
        # Exclude areas where tree_cover is 0
        valid_tree_cover_mask = tree_cover != 0

         # create a mask for pixels that have not lost any cover up until the target year
        no_loss_up_to_target_year_mask = (loss_year == 0) | (loss_year > (target_year - 2000))

        # Combine the valid tree cover mask with the no-loss-up-to-target-year mask
        # This final mask identifies pixels with tree cover that have not experienced loss up until the target year
        preservation_mask = (valid_tree_cover_mask & no_loss_up_to_target_year_mask).astype(np.uint8) * 255
        
        # Create a mask for loss within the specified year range, excluding areas with no tree cover
        target_year_mask = (loss_year >= 1) & (loss_year <= (target_year - 2000))
        
        # Adjusting for the encoding where 1 corresponds to 2001, hence target_year - 2000
        loss_mask = (valid_tree_cover_mask & target_year_mask).astype(np.uint8) * 255
        
        # Get the extent of the raster
        extent = [src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top]
        
        # Prepare the figure
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot the tree cover
        tree_cover_layer = ax.imshow(preservation_mask, cmap='Greens', extent=extent)

        # Overlay the loss mask
        loss_layer = ax.imshow(loss_mask, cmap='Reds', extent=extent, alpha=0.6)
        
        # Adding the OSM background with contextily
        #if basemap == True: 
            #ctx.add_basemap(ax, crs=raster_crs, source=ctx.providers.CartoDB.Positron, alpha=0.5)
        
        # Overlay the GeoDataFrame boundaries and labels
        gdf.boundary.plot(ax=ax, color='black', linewidth=1)
        
        for idx, row in gdf.iterrows():
            ax.annotate(text=row[names_column], xy=(row.geometry.centroid.x, row.geometry.centroid.y),
                        horizontalalignment='center', fontsize=5, color='black')

        
        # Create legends
        tree_cover_legend = mpatches.Patch(color='green', label=f'Bosque en {target_year}')
        loss_legend = mpatches.Patch(color='red', label=f'Pérdida desde {base_year}')
        plt.legend(handles=[tree_cover_legend, loss_legend], loc='lower left')
        
        # Set the title and axes labels
        ax.set_title(f'Pérdida de bosque {base_year} - {target_year} en {name_of_area}')
        
        # Remove x and y ticks
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Save the figure
        plt.savefig(output_path, bbox_inches='tight', dpi=400)
        plt.show()
        plt.close()
        
        #print(f"Map has been saved to '{output_path}'")
