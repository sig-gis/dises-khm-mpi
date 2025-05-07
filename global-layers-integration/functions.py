from rasterio import mask

import os
import rasterio
from rasterio.features import shapes
from shapely.geometry import shape, mapping
import geopandas as gpd

import geopandas as gpd
from shapely.geometry import Point
from shapely.geometry import box
import fiona

from matplotlib import pyplot as plt
import seaborn as sns
import re
import numpy as np

def raster_clipping(shape_path, raster_path, file_name):
    """
    Clipping a raster image to the shape of a given shapefile and ensuring CRS compatibility.
    """
    # Load the shapefile using GeoPandas to easily handle CRS
    gdf = gpd.read_file(shape_path)

    with rasterio.open(raster_path) as src:
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
    with rasterio.open(cropped_raster_path, "w", **out_meta) as dest:
        dest.write(out_image)

    #print(f"Cropped raster saved to: {cropped_raster_path}")
    return cropped_raster_path


def inverted_clip_touching_gdf(gdf_to_clip, gdf_mask):
    """
    Clips all polygons from gdf_to_clip that do not intersect any polygon in gdf_mask, ensuring CRS consistency and reprojecting if necessary.
    
    Parameters:
    - gdf_to_clip (GeoDataFrame): The GeoDataFrame to be clipped.
    - gdf_mask (GeoDataFrame): The GeoDataFrame to use as the clipping mask.
    
    Returns:
    - GeoDataFrame: The inverted clipped GeoDataFrame containing only polygons that do not intersect the mask.
    """
    # Check CRS consistency
    if gdf_to_clip.crs != gdf_mask.crs:
        gdf_mask = gdf_mask.to_crs(gdf_to_clip.crs)
    
    # Combine all mask geometries into a single geometry
    combined_mask = gdf_mask.unary_union
    
    # Select polygons that do not intersect the mask
    not_intersecting_gdf = gdf_to_clip[~gdf_to_clip.geometry.intersects(combined_mask)]
    
    return not_intersecting_gdf

def generate_urban_mask(raster_path, threshold, save_path=None):
    """
    Generate an urban mask GeoDataFrame from a raster file based on a given threshold.

    Parameters
    ----------
    raster_path : str
        Path to the input raster file.
    threshold : int or float
        Threshold value to determine urban areas. Pixels with values above this threshold are considered urban (normal range is 0-30).
    save_path : str, optional
        Path to save the resulting GeoDataFrame as a file. If None, the file is not saved.

    Returns
    -------
    gpd.GeoDataFrame
        GeoDataFrame containing the urban mask polygons.
    """
    # Step 1: Read the raster data
    with rasterio.open(raster_path) as mask_src:
        mask_data = mask_src.read(1)  # Read the first band
        mask_crs = mask_src.crs
        mask_transform = mask_src.transform

    # Step 2: Create a binary mask based on the threshold
    binary_mask = mask_data > threshold

    # Step 3: Convert the binary mask to polygons
    shapes_gen = shapes(binary_mask.astype(np.int16), transform=mask_transform)
    mask_polygons = [shape(geom) for geom, value in shapes_gen if value == 1]

    # Step 4: Convert polygons to GeoJSON-like dictionary
    mask_geom = [mapping(polygon) for polygon in mask_polygons]

    # Step 5: Transform mask_geom to GeoDataFrame
    urban_mask = gpd.GeoDataFrame(geometry=[shape(geom) for geom in mask_geom], crs=mask_crs)

    # Step 6: Optionally save the GeoDataFrame to a file
    if save_path:
        urban_mask.to_file(save_path)

    return urban_mask