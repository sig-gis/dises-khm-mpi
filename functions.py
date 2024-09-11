import numpy as np
import pandas as pd

from sklearn.preprocessing import PowerTransformer
from scipy.stats import skew, kurtosis, boxcox, yeojohnson

from scipy.special import inv_boxcox
from numpy import expm1, sqrt, square, log1p

from scipy.spatial import cKDTree

import rasterio
from rasterio.enums import Resampling
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

def resample_raster(input_path, output_path, scale, resampling_method=Resampling.bilinear):
    """
    Resample a raster dataset to a different resolution.

    Parameters:
    input_path (str): Path to the input raster file.
    output_path (str): Path to save the resampled raster file.
    scale (float): Resampling scale factor. Values <1 will downsample, values >1 will upsample.
    resampling_method (rasterio.enums.Resampling, optional): The resampling method to use.
        Default is Resampling.bilinear. Other options include Resampling.nearest, Resampling.cubic, etc.

    Returns:
    None
    """
    with rasterio.open(input_path) as dataset:
        # Calculate the new dimensions
        new_height = int(dataset.height * scale)
        new_width = int(dataset.width * scale)

        # Perform the resampling
        data = dataset.read(
            out_shape=(
                dataset.count,
                new_height,
                new_width
            ),
            resampling=resampling_method
        )

        # Update the transform for the new dimensions
        transform = dataset.transform * dataset.transform.scale(
            (dataset.width / new_width),
            (dataset.height / new_height)
        )

        # Write the resampled raster to a new file
        with rasterio.open(
            output_path,
            'w',
            driver='GTiff',
            height=new_height,
            width=new_width,
            count=dataset.count,
            dtype=data.dtype,
            crs=dataset.crs,
            transform=transform,
        ) as dst:
            dst.write(data)


def exclude_zero_coordinates(gdf):
    """
    Exclude observations with (0, 0) coordinates from a GeoDataFrame.

    Parameters:
    gdf (GeoDataFrame): Input GeoDataFrame.

    Returns:
    GeoDataFrame: Filtered GeoDataFrame with observations having (0, 0) coordinates excluded.
    """
    # Ensure the geometries are of the correct type (Point)
    gdf = gdf[gdf.geometry.type == 'Point']
    
    # Filter out rows with (0, 0) coordinates
    filtered_gdf = gdf[~((gdf.geometry.x == 0) & (gdf.geometry.y == 0))]

    return filtered_gdf


import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import xy
from shapely.geometry import box
import geopandas as gpd

def raster_to_gdf(raster_path):
    """
    Convert a raster file to a GeoDataFrame with each band as a column and coordinates as geometry,
    excluding cells where all bands are NaN. The geometry will be squares forming a grid.

    Parameters:
    initial_raster_path (str): Path to the input raster file.

    Returns:
    GeoDataFrame: A GeoDataFrame with raster band data and square grid geometries.
    """
    try:
        # Open the raster file
        with rasterio.open(raster_path) as src:
            # Read the raster bands into a numpy array
            bands = src.read()
            band_count = src.count
            
            # Get the band names from the metadata
            band_names = src.descriptions if any(src.descriptions) else [f'band_{i+1}' for i in range(band_count)]
            
            # Create a DataFrame with each band as a column
            band_data = np.array([band.flatten() for band in bands]).T
            df = pd.DataFrame(band_data, columns=band_names)
            
            # Get the affine transformation of the raster
            transform = src.transform
            
            # Calculate the coordinates for each pixel
            rows, cols = np.indices(bands[0].shape)
            xs, ys = xy(transform, rows, cols)
            
            # Flatten the coordinates arrays
            xs = np.array(xs).flatten()
            ys = np.array(ys).flatten()
            
            # Filter out rows where all bands are NaN
            mask = ~np.all(np.isnan(band_data), axis=1)
            df = df[mask]
            xs = xs[mask]
            ys = ys[mask]
            
            # Create square polygons for each pixel
            pixel_size_x = transform.a
            pixel_size_y = -transform.e
            
            polygons = [box(x - pixel_size_x / 2, y - pixel_size_y / 2, x + pixel_size_x / 2, y + pixel_size_y / 2) for x, y in zip(xs, ys)]
            
            # Create a GeoDataFrame
            gdf = gpd.GeoDataFrame(df, geometry=polygons, crs=src.crs)
            
        return gdf
    
    except rasterio.errors.RasterioIOError as e:
        print(f"Error opening raster file: {e}")
        return None

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


def idw_interpolation(gdf, value_column, power=2):
    """
    Perform IDW interpolation to replace null values in the specified column of a GeoDataFrame.

    Parameters:
    gdf (GeoDataFrame): The input GeoDataFrame with geometry and value columns.
    value_column (str): The name of the column with the values to interpolate.
    power (float): The power parameter for IDW (default is 2).

    Returns:
    GeoDataFrame: The GeoDataFrame with null values replaced by IDW interpolated values.
    """
    # Separate the GeoDataFrame into known and unknown value parts
    known = gdf[gdf[value_column].notnull()]
    unknown = gdf[gdf[value_column].isnull()]
    
    # Use centroids for known and unknown geometries
    known_coords = np.array([(geom.centroid.x, geom.centroid.y) for geom in known.geometry])
    unknown_coords = np.array([(geom.centroid.x, geom.centroid.y) for geom in unknown.geometry])
    
    # Create a KDTree for known values
    tree = cKDTree(known_coords)
    
    # Define the IDW interpolation function
    def idw(x, y, z, xi, yi, power):
        distances, indices = tree.query([xi, yi], k=len(known_coords), distance_upper_bound=np.inf)
        weights = 1 / distances**power
        weights /= weights.sum()
        interpolated_value = np.dot(weights, z[indices])
        return interpolated_value

    # Apply IDW interpolation to each unknown value
    interpolated_values = []
    for geom in unknown.geometry:
        xi, yi = geom.centroid.x, geom.centroid.y
        z = known[value_column].values
        interpolated_value = idw(known_coords[:, 0], known_coords[:, 1], z, xi, yi, power)
        interpolated_values.append(interpolated_value)
    
    # Replace null values with interpolated values
    gdf.loc[gdf[value_column].isnull(), value_column] = interpolated_values
    
    return gdf

def plot_missing_values_vertical(df, filepath='.'):
    """
    Creates a heatmap showing missing values for each variable (column) of the DataFrame,
    with column names on the Y-axis.

    Args:
    df (DataFrame): The pandas DataFrame to analyze for missing values.
    """
    # Calculate the missing values in each column
    missing = df.isnull()

    # Create a heatmap visualization with rows and columns inverted
    plt.figure(figsize=(8, max(2, len(df.columns) * 0.25)))  # Adjust the figure size based on the number of columns
    sns.heatmap(missing.transpose(), cbar=False, cmap='viridis', yticklabels=True)

    # Add titles and labels in Spanish
    plt.title('Missing entries in each variable')
    plt.xlabel('Rows')
    plt.ylabel('Variables')

    if filepath != '.':
        plt.savefig(filepath)
    # Show the plot
    plt.show()

def plot_distribution_with_statistics(y,  filepath='.'):
    """
    Plots the distribution of a target variable with skewness and kurtosis values.

    Parameters:
    y (pd.Series or np.ndarray): The target variable data.

    Returns:
    None
    """
    # Calculate skewness and kurtosis
    skewness = skew(y)
    kurt = kurtosis(y)

    # Plot the distribution with skewness and kurtosis
    plt.figure(figsize=(10, 6))
    sns.histplot(y, bins=30, kde=True, color='skyblue')
    plt.title('Distribution of Target Variable')
    plt.xlabel('Value')
    plt.ylabel('Frequency')

    # Add skewness and kurtosis text to the plot
    plt.text(x=0.95, y=0.95, s=f'Skewness: {skewness:.2f}', horizontalalignment='right', verticalalignment='top', transform=plt.gca().transAxes, fontsize=12)
    plt.text(x=0.95, y=0.90, s=f'Kurtosis: {kurt:.2f}', horizontalalignment='right', verticalalignment='top', transform=plt.gca().transAxes, fontsize=12)

    if filepath != '.':
        plt.savefig(filepath)
    
    # Show the plot
    
    plt.show()

# Function to plot histograms in a grid with more bins
def plot_histograms(df, bins=50):
    num_cols = len(df.columns)
    num_rows = (num_cols + 1) // 2  # To arrange histograms in a grid (2 columns)
    
    fig, axes = plt.subplots(num_rows, 2, figsize=(10, num_rows * 3))
    axes = axes.flatten()  # Flatten the axes array to access each subplot easily
    
    for i, col in enumerate(df.columns):
        axes[i].hist(df[col], bins=bins, edgecolor='black')
        axes[i].set_title(col)
    
    # Remove any empty subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    
    plt.tight_layout()
    plt.show()

def log_transform(y):
    return np.log1p(y)

def sqrt_transform(y):
    return np.sqrt(y)

def square_transform(y):
    return np.square(y)

def boxcox_transform(y):
    transformed_y, lmbda = boxcox(y)
    return transformed_y, lmbda

def yeojohnson_transform(y):
    transformed_y, lmbda = yeojohnson(y)
    return transformed_y, lmbda

def select_transformation(y):
    """
    Selects the best transformation for the target variable based on its distribution characteristics,
    including skewness and the presence of negative values or zeros.

    The function considers the following transformations:
    - Logarithmic (log1p)
    - Square root
    - Square
    - Box-Cox (requires positive values)
    - Yeo-Johnson (can handle negative values and zeros)

    Parameters:
    y (pd.Series or np.ndarray): The target variable data.

    Returns:
    tuple: (transformed_y, transformation_name, lmbda)
           - transformed_y (np.ndarray): The transformed target variable.
           - transformation_name (str): The name of the applied transformation.
           - lmbda (float or None): The lambda value used for Box-Cox or Yeo-Johnson transformations. 
                                    None if the transformation does not require lambda.

    Notes:
    - If the target variable contains non-positive values, log and Box-Cox transformations are avoided.
    - The function prints the skewness and kurtosis of the original target variable.
    - Transformation selection is based on skewness:
        - Skewness > 1: Log transformation (if possible) or Yeo-Johnson
        - 0.5 < Skewness <= 1: Square root transformation
        - Skewness < -1: Square transformation
        - -0.5 < Skewness < 0.5: No transformation
        - Other cases: Box-Cox (if possible) or Yeo-Johnson

    Example:
    y = pd.Series([1, 2, 3, 4, 5])
    transformed_y, transformation_name, lmbda = select_transformation(y)
    """
    skewness = skew(y)
    kurt = kurtosis(y)
    print(f"Skewness: {skewness}, Kurtosis: {kurt}")

    if np.any(y <= 0):
        # If there are non-positive values, avoid log and Box-Cox transformations
        if skewness > 1:
            print("Applying Yeo-Johnson transformation due to high positive skewness and non-positive values.")
            transformed_y, lmbda = yeojohnson_transform(y)
            return transformed_y, 'yeo-johnson', lmbda
        elif skewness > 0.5:
            print("Applying square root transformation due to moderate positive skewness and non-positive values.")
            return sqrt_transform(y), 'sqrt', None
        elif skewness < -1:
            print("Applying square transformation due to high negative skewness and non-positive values.")
            return square_transform(y), 'square', None
        elif skewness > -0.5 and skewness < 0.5:
            print("No transformation applied due to low skewness and non-positive values.")
            return y, 'none', None  # No transformation
        else:
            print("Applying Yeo-Johnson transformation due to other skewness values and non-positive values.")
            transformed_y, lmbda = yeojohnson_transform(y)
            return transformed_y, 'yeo-johnson', lmbda
    else:
        # If all values are positive, consider all transformations
        if skewness > 1:
            print("Applying log transformation due to high positive skewness.")
            return log_transform(y), 'log', None
        elif skewness > 0.5:
            print("Applying square root transformation due to moderate positive skewness.")
            return sqrt_transform(y), 'sqrt', None
        elif skewness < -1:
            print("Applying square transformation due to high negative skewness.")
            return square_transform(y), 'square', None
        elif skewness > -0.5 and skewness < 0.5:
            print("No transformation applied due to low skewness.")
            return y, 'none', None  # No transformation
        else:
            try:
                print("Applying Box-Cox transformation due to other skewness values.")
                transformed_y, lmbda = boxcox_transform(y)
                return transformed_y, 'box-cox', lmbda
            except ValueError:
                print("Applying Yeo-Johnson transformation due to other skewness values and failed Box-Cox transformation.")
                transformed_y, lmbda = yeojohnson_transform(y)
                return transformed_y, 'yeo-johnson', lmbda

def revert_standardization(y_standardized, original_mean, original_std):
    """
    Reverts the standardization process by applying the inverse of standardization.

    Parameters:
    y_standardized (np.ndarray): The standardized target variable.
    original_mean (float): The mean of the original target variable before standardization.
    original_std (float): The standard deviation of the original target variable before standardization.

    Returns:
    np.ndarray: The original target variable before standardization.
    """
    return y_standardized * original_std + original_mean

def revert_transformation(y_transformed, transformation_name, original_mean=None, original_std=None, lmbda=None):
    """
    Reverts the transformation applied to the target variable based on the transformation name.

    Parameters:
    y_transformed (np.ndarray): The transformed target variable data.
    transformation_name (str): The name of the applied transformation.
    original_mean (float, optional): The mean of the original target variable, required if standardized.
    original_std (float, optional): The standard deviation of the original target variable, required if standardized.
    lmbda (float, optional): The lambda value used for the Box-Cox or Yeo-Johnson transformation, if applicable.

    Returns:
    np.ndarray: The reverted target variable, in its original form.

    Raises:
    ValueError: If the transformation name is not recognized.
    """

    if transformation_name == 'log':
        return expm1(y_transformed)
    elif transformation_name == 'sqrt':
        return square(y_transformed)
    elif transformation_name == 'square':
        return sqrt(y_transformed)
    elif transformation_name == 'box-cox':
        if lmbda is None:
            raise ValueError("Lambda value is required to revert Box-Cox transformation.")
        return inv_boxcox(y_transformed, lmbda)
    elif transformation_name == 'yeo-johnson':
        if lmbda is None:
            raise ValueError("Lambda value is required to revert Yeo-Johnson transformation.")
        return yeojohnson_inverse(y_transformed, lmbda)
    elif transformation_name == 'none':
        return y_transformed
    else:
        raise ValueError(f"Unrecognized transformation name: {transformation_name}")

def yeojohnson_inverse(y_transformed, lmbda):
    """
    Reverts the Yeo-Johnson transformation given the transformed data and lambda parameter.

    Parameters:
    y_transformed (np.ndarray): The Yeo-Johnson transformed data.
    lmbda (float): The lambda value used in the Yeo-Johnson transformation.

    Returns:
    np.ndarray: The original data before Yeo-Johnson transformation.
    """
    if lmbda == 0:
        return expm1(y_transformed)
    elif lmbda > 0:
        return np.exp(np.log1p(y_transformed * lmbda) / lmbda) - 1
    else:
        return -np.exp(np.log1p(-y_transformed * lmbda) / -lmbda) + 1


def filter_columns_by_year(gdf: pd.DataFrame, year: int) -> pd.DataFrame:
    """
    Filters the columns of a given DataFrame to retain only those that are associated with a specified year,
    one year before, and one year after, as well as any columns that do not contain a year in their name.

    Parameters:
    ----------
    gdf : pd.DataFrame
        The input DataFrame containing columns with and without year-based names.

    year : int
        The reference year for filtering columns. Columns with names containing this year,
        one year before, and one year after will be retained.

    Returns:
    -------
    pd.DataFrame
        A DataFrame containing only the filtered columns that meet the criteria of matching
        the specified year, one year before, and one year after, or having no year in their name.

    Example:
    -------
    Given a DataFrame `gdf` with columns ['tcc1990', 'tcc1991', 'tcc1992', 'population', 'area']:
    - Calling `filter_columns_by_year(gdf, 1991)` will return a DataFrame with columns
      ['tcc1990', 'tcc1991', 'tcc1992', 'population', 'area'].

    Notes:
    -----
    - The function assumes that columns containing years follow the format 'XXXX' where 'XXXX'
      is a four-digit number representing the year.
    - Columns that do not contain any year will be retained in the output DataFrame.
    """
    
    # Convert the year to string for easy matching
    year_str = str(year)
    year_before_str = str(year - 1)
    year_after_str = str(year + 1)
    
    # Function to check if a column name matches the year, one year before, or one year after
    def match_year(column_name):
        # Extract year from column name if it exists
        year_match = re.search(r'\d{4}', column_name)
        if year_match:
            col_year = year_match.group(0)
            return col_year in [year_str, year_before_str, year_after_str]
        else:
            # If no year found in column name, keep it
            return True
    
    # Filter columns based on the matching criteria
    filtered_columns = [col for col in gdf.columns if match_year(col)]
    
    # Return the filtered DataFrame
    return gdf[filtered_columns]

def raster_freq_table(raster_path):

    '''Returns a frequency table for the rasters values'''
    
    # Load the new raster file
    dataset = gdal.Open(raster_path)
    
    # Read the first band
    band = dataset.GetRasterBand(1)
    array = band.ReadAsArray()
    
    # Flatten the array to make frequency counting feasible
    flat_array = array.flatten()
    
    # Replace nodata values with nan to exclude them from the frequency count
    nodata_value = band.GetNoDataValue()
    if nodata_value is not None:
        flat_array = np.where(flat_array == nodata_value, np.nan, flat_array)
    
    # Remove nan values for frequency counting
    flat_array_nonan = flat_array[~np.isnan(flat_array)]
    
    # Generate frequency table
    unique, counts = np.unique(flat_array_nonan, return_counts=True)
    frequency_table = pd.DataFrame({"Value": unique, "Frequency": counts})
    
    return frequency_table


