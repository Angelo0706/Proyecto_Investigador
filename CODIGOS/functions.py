#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File Name: functions.py

Authors:
    Arturo Villarroya-Carpio, Juan M. Lopez-Sanchez  
    University of Alicante

Date: May 2025

Description:
------------
This module provides a collection of utility functions for the handling, visualisation, 
and analysis of remote sensing data from the S.O.D.A.S. dataset. The data, stored
in a NetCDF file, includes imagery of optical reflectance, NDVI, dual-pol SAR covariance matrix
repeta-pass interferometric coherence, and crop type maps. 

Index of Functions:
-------------------
- ensure_dates_as_array(): Ensures a date or list of dates is returned as a list.
- bytescl(): Scales array data into 8-bit (0–255) format using contrast limits.
- save_figure(): Saves a matplotlib figure to disk in a specified location.
- get_rain_data(): Extracts precipitation data and corresponding dates within a specified date range.
- contrast_stretch(): Performs contrast stretching on an RGB image to enhance visual appearance.
- search_date(): Finds the closest date available in a list to a given target date.
- erode_crop_map(): Applies morphological erosion to a crop classification map to remove field borders.
- get_spatial_subset_indices(): Computes array indices for a given spatial bounding box.
- show_RGB_images(): Displays and optionally saves RGB composites from Sentinel-2 optical data.
- show_NDVI_images(): Displays and optionally saves NDVI images from Sentinel-2 data.
- show_crop_type_map(): Displays and optionally saves crop classification maps.
- show_backscatter_images(): Displays and optionally saves backscatter images (σ0).
- show_coherence_images(): Displays and optionally saves SAR coherence images (γ).
- show_phase_images(): Displays and optionally saves SAR interferometric phase images (ϕ).
- compute_and_show_decomposition_images(): Computes and visualises SAR decomposition metrics (e.g., mp, mv, alpha).
- create_time_series(): Creates a time series of average and standard deviation values for a specific crop type.
- save_time_series(): Extracts and optionally plots/saves time series data for a chosen variable (NDVI, SAR backscatter, coherence).
- save_time_series_decomposition(): Extracts, plots, and saves model-based decomposition time series (e.g., mp, mv).
- get_full_path(): Constructs file path for loading a specific crop-variable time series.
- generate_axis_label(): Produces a formatted label for time series axis based on the variable type.
- load_series_data(): Loads a pre-saved pickle file of time series data and returns the variable and corresponding dates.
- compare_saved_time_series(): Compares two time series data sets saved in files and calculate the correlation (R²) and RMSE between them.
- compare_time_series(): Compares two time series and calculates correlation (R²) and RMSE.

Dependencies:
-------------
- numpy, os, pickle, matplotlib, datetime
- netCDF4, scipy.ndimage, scipy.interpolate, scipy.stats

Usage:
------
To use the functions in this module, import the module and call the relevant function with the required parameters.
Each function is designed to handle specific types of remote sensing data and will either return the processed 
data or display/save the results. Many functions accept optional arguments for saving figures or data to disk.
"""

import numpy as np
import os
import pickle
import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.colors
from matplotlib.patches import Patch
from netCDF4 import Dataset
from scipy.interpolate import interp1d
from scipy import ndimage
from scipy.stats import linregress

def ensure_dates_as_array( dates ):
    """
    Ensure that the input is returned as a list of date strings.
    
    Parameters
    ----------
    dates : str or list of str
        A single date string or a list of date strings.
    
    Returns
    -------
    list of str
        A list containing one or more date strings.
    """
    
    if isinstance(dates, str):  # Check if target_dates is a string
        dates = [dates]  # Wrap the string in a list
    
    return dates
    
def bytescl( data, lim, mask = None ):
    """
    Scale data to 8-bit (0–255) range based on specified limits.
    
    Parameters
    ----------
    data : array-like
        Input data to be scaled.
    lim : tuple of float
        Lower and upper limits for scaling (min, max).
    mask : array-like of bool, optional
        Mask to apply scaling only to specific elements.
    
    Returns
    -------
    np.ndarray (uint8)
        Scaled data in the 0–255 range.
    """
    
    data_val = np.clip( (data - lim[0]) / (lim[1] - lim[0]), 0, 1)
                
    data_byte = np.zeros_like( data_val, dtype=np.uint8)
    
    if mask is None:
           data_byte[:] = 255 * data_val[:]
    else:
           data_byte[mask] = 255 * data_val[mask]
    
    return data_byte

def save_figure( save_dir, filename ):
    """
    Save the current matplotlib figure to a specified directory and filename.
    
    Parameters
    ----------
    save_dir : str
        Directory to save the figure in.
    filename : str
        Name of the output file (e.g., 'plot.png').
    
    Returns
    -------
    None
   """

    # Ensure the save_dir exists; create it if it doesn't
    if not os.path.exists(save_dir):
        try:
            os.makedirs(save_dir)  # Create the directory and any necessary parent directories
            print(f'Created directory: {save_dir}')
        except OSError as myerror:
            print(f'Failed to create directory: {save_dir}. Error: {myerror}')
            return  # Exit the function if the directory cannot be created
        
    # Save the figure
    save_path = os.path.join(save_dir, filename)
    try:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)  # Save with high resolution
        print(f'Figure successfully saved to {save_path}')
    except Exception as myerror:
        print(f'Failed to save figure: {save_path}. Error: {myerror}')

    return 
     
def get_rain_data( path_dataset, date_start, date_end ):
    """
    Extract precipitation data and corresponding dates from the NetCDF dataset 
    within a specified date range.
    
    Parameters
    ----------
    path_dataset : str
        Path to the NetCDF file containing precipitation data.
    date_start : str
        Start date in 'YYYY-MM-DD' format.
    date_end : str
        End date in 'YYYY-MM-DD' format.
    
    Returns
    -------
    dates_rain : list of datetime.datetime
        List of dates corresponding to the extracted data.
    rain : np.ndarray
        Array of precipitation values between the start and end dates.
    """

    # Open the dataset file
    nc_file = Dataset(path_dataset, mode='r')

    rain = nc_file.variables['precipitation'][:]    # Load precipitation data
    dates_rain_str = nc_file.variables['times_precipitation'][:]
    
    date1_rain = search_date(dates_rain_str, date_start)
    pos_date_start_rain = list(dates_rain_str).index(date1_rain)
    date2_rain = search_date(dates_rain_str, date_end)
    pos_date_end_rain = list(dates_rain_str).index(date2_rain)
    
    dates_rain = [datetime.datetime.strptime(d, '%Y-%m-%d') for d in dates_rain_str]
    dates_rain = dates_rain[pos_date_start_rain:pos_date_end_rain+1]
    rain = rain[pos_date_start_rain:pos_date_end_rain+1]
       
    # Close the dataset file
    nc_file.close()

    return dates_rain, rain
    
def contrast_stretch(image):
    """
    Apply contrast stretching to a 3-channel image using the 2nd and 98th percentiles.

    Parameters
    ----------
    image : np.ndarray
        RGB image as a NumPy array of shape (H, W, 3).

    Returns
    -------
    np.ndarray (uint8)
        Contrast-stretched image as an 8-bit RGB array.
    """
    
    stretched = np.zeros_like(image, dtype=np.float32)
    for i in range(3):  # Apply for each channel (R, G, B)
        channel = np.asarray(image[:, :, i])  # Ensure it's a normal NumPy array
        channel = np.nan_to_num(channel, nan=0)  # Replace NaNs with 0
        min_val, max_val = np.nanpercentile(channel, (2, 98))  # Remove 2% outliers
        
        if min_val == max_val:  # Avoid division by zero
            stretched[:, :, i] = 0
        else:
            stretched[:, :, i] = np.clip((channel - min_val) / (max_val - min_val) * 255, 0, 255)
    return stretched.astype(np.uint8)

def search_date(dates, target_date):
    """
    Find the closest available date in the list to a given target date.

    Parameters
    ----------
    dates : list of str
        Available dates in 'YYYY-MM-DD' format.
    target_date : str
        Desired target date in 'YYYY-MM-DD' format.

    Returns
    -------
    str
        Exact or closest available date in 'YYYY-MM-DD' format.
    """
    
    dates_datetime = [datetime.datetime.strptime(d, '%Y-%m-%d') for d in dates]
    target_date_datetime = datetime.datetime.strptime(target_date, '%Y-%m-%d')
    
    if target_date in dates:
        return target_date
    else:
        closest_date = dates_datetime[np.argmin(np.abs(np.array(dates_datetime) - target_date_datetime))]
        print('Date ' + target_date + ' is not available.')
        print('The closest date, ' + str(closest_date) + ', was used instead. \n')
        closest_date_str = closest_date.strftime('%Y-%m-%d')
        return closest_date_str

def erode_crop_map( mask, erosion_size = 3 ):
    """
    Apply binary erosion to a crop classification mask to reduce small patches.

    Parameters
    ----------
    mask : np.ndarray
        Binary or categorical mask. Can be 2D or 3D (time, rows, cols).
    erosion_size : int, optional
        Size of the structuring element used for erosion. Default is 3.

    Returns
    -------
    np.ndarray
        Eroded crop mask with the same shape as the input.
    """
    
    if erosion_size > 1:

        erosion_structure = np.ones((erosion_size,erosion_size))

        if len(mask.shape) == 3: # Various years
            
            for i in range(mask.shape[0]): # For each year
        
                mask_binary = mask[i,:,:] > 0
                mask_binary_eroded = ndimage.binary_erosion( mask_binary, structure=erosion_structure).astype(int)
                mask[i,:,:]  = mask[i,:,:]  * mask_binary_eroded
                
        else: # Single year map

                mask_binary = mask > 0
                mask_binary_eroded = ndimage.binary_erosion( mask_binary, structure=erosion_structure).astype(int)
                mask = mask * mask_binary_eroded

    return mask

def get_spatial_subset_indices( x, y, spatial_subset):
    """
    Calculate index bounds corresponding to a spatial subset within coordinate arrays.

    Parameters
    ----------
    x : np.ndarray
        1D array of x-coordinates.
    y : np.ndarray
        1D array of y-coordinates.
    spatial_subset : list or tuple of float
        [x_min, x_max, y_min, y_max] defining the spatial extent to extract.

    Returns
    -------
    x_low_index : int
        Index of the lower x-bound.
    x_up_index : int
        Index of the upper x-bound.
    y_low_index : int
        Index of the lower y-bound.
    y_up_index : int
        Index of the upper y-bound.
    """
    
    if spatial_subset[0] >=  x.min():
        x_low_index = np.argmin( np.abs( spatial_subset[0] - x ) )
    else:
        x_low_index = 0

    if spatial_subset[1] <=  x.max():
        x_up_index = np.argmin( np.abs( spatial_subset[1] - x ) )
    else:
        x_up_index = len(x)-1
        
    if spatial_subset[2] >=  y.min():
        y_low_index = np.argmin( np.abs( spatial_subset[2] - y ) )
    else:
        y_low_index = 0

    if spatial_subset[3] <=  y.max():
        y_up_index = np.argmin( np.abs( spatial_subset[3] - y ) )
    else:
        y_up_index = len(y)-1
            
    return x_low_index, x_up_index, y_low_index, y_up_index
        
def show_RGB_images(path_dataset, target_dates, subset = False, save_dir = 'dummy', save_images = False):
    """
    Display and optionally save RGB images from the NetCDF dataset for specified dates.

    Parameters
    ----------
    path_dataset : str
        Path to the NetCDF dataset.
    target_dates : list or str
        Dates to be visualised (can be a single date or a list of dates).
    subset : list or tuple of float, optional
        Spatial subset [x_min, x_max, y_min, y_max] to extract (default is False, i.e. full extent).
    save_dir : str, optional
        Directory where images will be saved if save_images is True.
    save_images : bool, optional
        If True, the RGB images are saved to disk.

    Returns
    -------
    None
    """
    
    nc_file = Dataset(path_dataset, mode='r')

    dates_optical = nc_file.variables['times_optical'][:]
    
    # Load data ===============================================================
    x = np.array(nc_file.variables['x'])
    y = np.array(nc_file.variables['y'])

    # Spatial subset case
    
    if subset and isinstance(subset, (list, tuple)) and len(subset) == 4:
        x_min, x_max, y_min, y_max = subset
        
        # Ensure subset is within valid bounds
        x_min = max(x.min(), x_min)
        x_max = min(x.max(), x_max)
        y_min = max(y.min(), y_min)
        y_max = min(y.max(), y_max)
    
        # Convert coordinates to indices
        x_low_index, x_up_index, y_low_index, y_up_index = get_spatial_subset_indices(x, y, [x_min, x_max, y_min, y_max])
    
        # Extract the spatial subset
        spatial_subset = [x[x_low_index], x[x_up_index], y[y_low_index], y[y_up_index]]

    else:
        # Default: Use the full image
        spatial_subset = [x.min(), x.max(), y.min(), y.max()]
        
    # Find the closest dates to target_dates
    target_dates = ensure_dates_as_array( target_dates )
    dates = [search_date(dates_optical, d) for d in target_dates]
    
    for date in dates:
        index_date = list(dates_optical).index(date)
    
        blue_band = nc_file.variables['S2'][index_date, 0, :, :]
        green_band = nc_file.variables['S2'][index_date, 1, :, :]
        red_band = nc_file.variables['S2'][index_date, 2, :, :]
        image = np.stack((red_band, green_band, blue_band), axis=-1)
               
        # Optional: Applies contrast stretching
        if subset:
            stretched_image = contrast_stretch(image[ y_up_index:y_low_index+1, x_low_index:x_up_index+1, : ])
        else:
            stretched_image = contrast_stretch(image)

        # Plot image ==========================================================
        plt.figure(figsize=[9.6, 6.4])
        
        plt.title('RGB image: ' + date, fontsize = 20 )
        plt.imshow(stretched_image, extent = spatial_subset )
        
        plt.xlabel('X-UTM (m)')
        plt.ylabel('Y-UTM (m)')
        
        
        # Save the image to the specified directory if indicated
        if save_images == True:            
            save_figure( save_dir, 'RGB_' + date + '.png' )
            
            
    # Close the file
    nc_file.close()

def show_NDVI_images(path_dataset, target_dates, subset = False, save_dir = 'dummy', save_images = False):
    """
    Display and optionally save NDVI images from the NetCDF dataset for specified dates.

    Parameters
    ----------
    path_dataset : str
        Path to the NetCDF dataset.
    target_dates : list or str
        Dates to be visualised (can be a single date or a list of dates).
    subset : list or tuple of float, optional
        Spatial subset [x_min, x_max, y_min, y_max] to extract (default is False, i.e. full extent).
    save_dir : str, optional
        Directory where images will be saved if save_images is True.
    save_images : bool, optional
        If True, the NDVI images are saved to disk.

    Returns
    -------
    None
    """
    
    nc_file = Dataset(path_dataset, mode='r')
    
    dates_NDVI = nc_file.variables['times_optical'][:]

    # Load data ===============================================================
    x = np.array(nc_file.variables['x'])
    y = np.array(nc_file.variables['y'])
 
    # Spatial subset case
    
    if subset and isinstance(subset, (list, tuple)) and len(subset) == 4:
        x_min, x_max, y_min, y_max = subset
        
        # Ensure subset is within valid bounds
        x_min = max(x.min(), x_min)
        x_max = min(x.max(), x_max)
        y_min = max(y.min(), y_min)
        y_max = min(y.max(), y_max)
    
        # Convert coordinates to indices
        x_low_index, x_up_index, y_low_index, y_up_index = get_spatial_subset_indices(x, y, [x_min, x_max, y_min, y_max])
    
        # Extract the spatial subset
        spatial_subset = [x[x_low_index], x[x_up_index], y[y_low_index], y[y_up_index]]

    else:
        # Default: Use the full image
        spatial_subset = [x.min(), x.max(), y.min(), y.max()]
        
    # Find the closest dates to target_dates
    target_dates = ensure_dates_as_array( target_dates )
    dates = [search_date(dates_NDVI, d) for d in target_dates]

    for date in dates:
        index_date = list(dates_NDVI).index(date)
        
        if subset:
            image = nc_file.variables['S2'][index_date, 4, y_up_index:y_low_index+1, x_low_index:x_up_index+1 ]
            
        else:
            image = nc_file.variables['S2'][index_date, 4, :, :]

        # Plot image ==========================================================
        plt.figure(figsize=[9.6,6.4])
        
        plt.title('NDVI:' + date, fontsize = 20 )
        plt.imshow(image, extent = spatial_subset, vmin = 0, vmax = 1, cmap='Greys_r')
        
        plt.xlabel('X-UTM (m)')
        plt.ylabel('Y-UTM (m)')
        
        plt.colorbar()

        # Save the image to the specified directory if indicated
        if save_images == True:            
            save_figure( save_dir, 'NDVI_' + date + '.png' )
            
    # Close the file
    nc_file.close()
    
def show_crop_type_map(path_dataset, year, erosion_size, subset = False, save_dir = 'dummy', save_images = False):
    """
    Display and optionally save an eroded crop-type map for a specified year from the NetCDF dataset.

    Parameters
    ----------
    path_dataset : str
        Path to the NetCDF dataset.
    year : int
        Year for which the crop-type map should be shown.
    erosion_size : int
        Size of the morphological erosion to apply to the map.
    subset : list or tuple of float, optional
        Spatial subset [x_min, x_max, y_min, y_max] to extract (default is False, i.e. full extent).
    save_dir : str, optional
        Directory where the image will be saved if save_images is True.
    save_images : bool, optional
        If True, the crop-type map is saved to disk.

    Returns
    -------
    None
    """ 
    
    nc_file = Dataset(path_dataset, mode='r')
    
    year_index = {2017 : 0, 2018 : 1, 2019 : 2, 2020 : 3, 2021 : 4}
    
    # Load data ===============================================================
    x = np.array(nc_file.variables['x'])
    y = np.array(nc_file.variables['y'])
    
    # Spatial subset case
    
    if subset and isinstance(subset, (list, tuple)) and len(subset) == 4:
        x_min, x_max, y_min, y_max = subset
        
        # Ensure subset is within valid bounds
        x_min = max(x.min(), x_min)
        x_max = min(x.max(), x_max)
        y_min = max(y.min(), y_min)
        y_max = min(y.max(), y_max)
        
        # Convert coordinates to indices
        x_low_index, x_up_index, y_low_index, y_up_index = get_spatial_subset_indices(x, y, [x_min, x_max, y_min, y_max])
    
        # Extract the spatial subset
        spatial_subset = [x[x_low_index], x[x_up_index], y[y_low_index], y[y_up_index]]
        
    else:
        spatial_subset = [x.min(), x.max(), y.min(), y.max()]
  
    if subset:
        image = nc_file.variables['crop_type_map'][year_index[year]][y_up_index:y_low_index+1, x_low_index:x_up_index+1 ]
    else:
        image = nc_file.variables['crop_type_map'][year_index[year]][:, :]      
    
    # Erosion of the crop map =================================================
    image = erode_crop_map(image, erosion_size)
    
    # Define fixed color map and crop labels

    crop_labels = ['Other',
            'Alfalfa',
            'Barley',
            'Carrot',
            'Chickpea',
            'Cotton',
            'Fallow',
            'Maize',
            'Oats',
            'Onion',
            'Pepper',
            'Potato',
            'Pumpkin',
            'Quinoa',
            'Rice',
            'SugarBeet',
            'Sunflower',
            'SweetPotato',
            'Tomato',
            'Wheat']

    crop_colors = [
        (1.0, 1.0, 1.0, 1.0),       # Other - white
        (0.4, 0.76, 0.65, 1.0),     # Alfalfa
        (0.93, 0.79, 0.45, 1.0),    # Barley
        (0.99, 0.55, 0.38, 1.0),    # Carrot
        (0.55, 0.63, 0.8, 1.0),     # Chickpea
        (0.91, 0.54, 0.76, 1.0),    # Cotton
        (0.65, 0.85, 0.33, 1.0),    # Fallow
        (1.0, 0.85, 0.18, 1.0),     # Maize
        (0.86, 0.8, 0.69, 1.0),     # Oats
        (0.9, 0.77, 0.58, 1.0),     # Onion
        (0.7, 0.7, 0.7, 1.0),       # Pepper
        (0.98, 0.5, 0.45, 1.0),     # Potato
        (0.5, 0.8, 0.89, 1.0),      # Pumpkin
        (0.79, 0.7, 0.84, 1.0),     # Quinoa
        (0.8, 0.92, 0.77, 1.0),     # Rice
        (0.67, 0.62, 0.52, 1.0),    # Sugar beet
        (0.88, 0.61, 0.14, 1.0),    # Sunflower
        (0.6, 0.98, 0.6, 1.0),      # Sweet potato
        (0.91, 0.3, 0.24, 1.0),     # Tomato
        (0.33, 0.66, 0.33, 1.0)     # Wheat
    ]
    
    cmap = plt.cm.colors.ListedColormap(crop_colors)
    
    # Plot image ==============================================================
    plt.figure(figsize = [9.6, 6.4])
        
    plt.title('Crop-type map ' + str(year), fontsize = 20 )
    plt.imshow(image, extent = spatial_subset, cmap=cmap, vmin=0, vmax=len(crop_colors))
    
    plt.xlabel('X-UTM (m)')
    plt.ylabel('Y-UTM (m)')
    
    # Legend
    legend_elements = [Patch(facecolor=crop_colors[i], label=crop_labels[i]) for i in range(len(crop_labels))]
    plt.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., fontsize=9)

    
    # Save the image to the specified directory if indicated
    if save_images == True:            
        save_figure( save_dir, 'crop_type_map_' +str(year) + '.png' )
                       
    # Close the file
    nc_file.close()
    
def show_backscatter_images(path_dataset, target_dates, channel, subset = False, save_dir = 'dummy', save_images = False):
    """
    Display and optionally save SAR backscatter images (σ⁰) for given dates and channels.

    Parameters
    ----------
    path_dataset : str
        Path to the NetCDF file containing the SAR data.

    target_dates : str or list of str
        Target acquisition dates to display (format: 'YYYY-MM-DD').

    channel : str
        Polarisation channel to display: 'VV', 'VH', or 'ALL'.

    subset : bool or list of float, optional
        If list of four floats [x_min, x_max, y_min, y_max], a spatial subset is used. 
        If False (default), the full image extent is used.

    save_dir : str, optional
        Directory to save the generated figures if `save_images` is True. Default is 'dummy'.

    save_images : bool, optional
        Whether to save the figures to disk. Default is False.

    Returns
    -------
    None
        Displays the backscatter images and optionally saves them as PNG files.
    """
    
    nc_file = Dataset(path_dataset, mode='r')
    
    channel = channel.upper()

    if channel == 'VV':
        channel_option = [0]
        channel_label = [channel]
    elif channel == 'VH': 
        channel_option = [3]
        channel_label = [channel]
    elif channel == 'ALL':
        channel_option = [0, 3]
        channel_label = ['VV', 'VH']

        
    dates_sigma0 = nc_file.variables['times_C2'][:]
    
    # Load data ===============================================================
    x = np.array(nc_file.variables['x'])
    y = np.array(nc_file.variables['y'])
        
    # Spatial subset case
    
    if subset and isinstance(subset, (list, tuple)) and len(subset) == 4:
        x_min, x_max, y_min, y_max = subset
        
        # Ensure subset is within valid bounds
        x_min = max(x.min(), x_min)
        x_max = min(x.max(), x_max)
        y_min = max(y.min(), y_min)
        y_max = min(y.max(), y_max)
    
        # Convert coordinates to indices
        x_low_index, x_up_index, y_low_index, y_up_index = get_spatial_subset_indices(x, y, [x_min, x_max, y_min, y_max])
    
        # Extract the spatial subset
        spatial_subset = [x[x_low_index], x[x_up_index], y[y_low_index], y[y_up_index]]

    else:
        # Default: Use the full image
        spatial_subset = [x.min(), x.max(), y.min(), y.max()]
    
    # Find the closest dates to target_dates
    target_dates = ensure_dates_as_array( target_dates )
    dates = [search_date(dates_sigma0, d) for d in target_dates]
    
    for i, chan in enumerate(channel_option):
        for date in dates:
            index_date = list(dates_sigma0).index(date)
            
            if subset:
                c2 = nc_file.variables['S1_C2'][index_date, chan, y_up_index:y_low_index+1, x_low_index:x_up_index+1 ]
            else:
                c2 = nc_file.variables['S1_C2'][index_date, chan, :, :]
               
            # Masking zeros in the image
            c2 = np.where(c2 == 0, np.nan, c2)
            
            # Change to logarithmic scale
            image = 10 * np.log10(c2)
            
            # Plot image ==================================================================
            plt.figure(figsize=[9.6,6.4] )
    
            plt.title(r'$\sigma_{0}$' + channel_label[i] + ' : ' + date, fontsize = 20)
            plt.imshow(image, extent = spatial_subset, cmap='gray')
            
            plt.colorbar()
            
            plt.xlabel('X-UTM (m)')
            plt.ylabel('Y-UTM (m)')
            
            # Save the image to the specified directory if indicated
            if save_images == True:            
                save_figure( save_dir, 'sigma0_' + channel_label[i] + '_' + date + '.png' )
            
        
    # Close the file
    nc_file.close()

def show_coherence_images(path_dataset, target_dates, channel, baseline, subset = False, save_dir = 'dummy', save_images = False):
    """
    Display and optionally save interferometric coherence images (γ) for given dates and channels.

    Parameters
    ----------
    path_dataset : str
        Path to the NetCDF file containing coherence data.

    target_dates : str or list of str
        Target acquisition dates to display (format: 'YYYY-MM-DD').

    channel : str
        Polarisation channel to display: 'VV', 'VH', or 'ALL'.

    baseline : int
        Temporal baseline in days for the coherence layer (e.g., 6, 12).

    subset : bool or list of float, optional
        If list of four floats [x_min, x_max, y_min, y_max], a spatial subset is used. 
        If False (default), the full image extent is used.

    save_dir : str, optional
        Directory to save the generated figures if `save_images` is True. Default is 'dummy'.

    save_images : bool, optional
        Whether to save the figures to disk. Default is False.

    Returns
    -------
    None
        Displays the coherence images and optionally saves them as PNG files.
    """
    
    nc_file = Dataset(path_dataset, mode='r')

    channel = channel.upper()
    if channel == 'VV':
        channel_option = [0]
        channel_label = [channel]
    elif channel == 'VH': 
        channel_option = [1]
        channel_label = [channel]
    elif channel == 'ALL':
        channel_option = [0, 1]
        channel_label = ['VV', 'VH']
        
    variable_dates = 'times_coher_' + str(baseline).zfill(2) + 'days'
    dates_coherence = nc_file.variables[variable_dates][:]
    
    # Load data ===============================================================
    x = np.array(nc_file.variables['x'])
    y = np.array(nc_file.variables['y'])
    
    # Spatial subset case
    
    if subset and isinstance(subset, (list, tuple)) and len(subset) == 4:
        x_min, x_max, y_min, y_max = subset
        
        # Ensure subset is within valid bounds
        x_min = max(x.min(), x_min)
        x_max = min(x.max(), x_max)
        y_min = max(y.min(), y_min)
        y_max = min(y.max(), y_max)
    
        # Convert coordinates to indices
        x_low_index, x_up_index, y_low_index, y_up_index = get_spatial_subset_indices(x, y, [x_min, x_max, y_min, y_max])
    
        # Extract the spatial subset
        spatial_subset = [x[x_low_index], x[x_up_index], y[y_low_index], y[y_up_index]]

    else:
        # Default: Use the full image
        spatial_subset = [x.min(), x.max(), y.min(), y.max()]

    # Find the closest dates to target_dates
    target_dates = ensure_dates_as_array( target_dates )
       
    dates = [search_date(dates_coherence, d) for d in target_dates]
    
    for i, chan in enumerate(channel_option):
        for date in dates:
            index_date = list(dates_coherence).index(date)
            variable_coher = 'S1_' + str(baseline).zfill(2) + 'days' + '_coherence'
            
            if subset == False:
                image = nc_file.variables[variable_coher][index_date, chan, :, :]
            else:
                image = nc_file.variables[variable_coher][index_date, chan, y_up_index:y_low_index+1, x_low_index:x_up_index+1 ]
            
            # Masking zeros in the image
            image = np.where(image == 0, np.nan, image)
            
            # Plot image ==========================================================
            plt.figure(figsize=[9.6,6.4])
            
            plt.title(rf'{baseline}-day $\gamma_{{{channel_label[i]}}}$: {date}', fontsize = 20)
            plt.imshow(image, extent = spatial_subset, vmin = 0, vmax = 1, cmap='Greys_r')
            
            plt.colorbar()
            
            plt.xlabel('X-UTM (m)')
            plt.ylabel('Y-UTM (m)')
    
            # Save the image to the specified directory if indicated
            if save_images == True:            
                save_figure( save_dir, 'coherence_' + str(baseline).zfill(2) +'days_' + channel_label[i] + '_' + date + '.png'  )

    # Close the file
    nc_file.close()

def show_phase_images(path_dataset, target_dates, channel, baseline, subset = False, save_dir = 'dummy', save_images = False):
    """
    Display and optionally save interferometric phase images (ϕ) for given dates and channels.

    Parameters
    ----------
    path_dataset : str
        Path to the NetCDF file containing phase data.

    target_dates : str or list of str
        Target acquisition dates to display (format: 'YYYY-MM-DD').

    channel : str
        Polarisation channel to display: 'VV', 'VH', or 'ALL'.

    baseline : int
        Temporal baseline in days for the interferometric pair (e.g., 6, 12).

    subset : bool or list of float, optional
        If list of four floats [x_min, x_max, y_min, y_max], a spatial subset is used. 
        If False (default), the full image extent is used.

    save_dir : str, optional
        Directory to save the generated figures if `save_images` is True. Default is 'dummy'.

    save_images : bool, optional
        Whether to save the figures to disk. Default is False.

    Returns
    -------
    None
        Displays the phase images and optionally saves them as PNG files.
    """
    
    nc_file = Dataset(path_dataset, mode='r')

    channel = channel.upper()
    if channel == 'VV':
        channel_option = [0]
        channel_label = [channel]
    elif channel == 'VH': 
        channel_option = [1]
        channel_label = [channel]
    elif channel == 'ALL':
        channel_option = [0, 1]
        channel_label = ['VV', 'VH']
        
    variable_dates = 'times_coher_' + str(baseline).zfill(2) + 'days'
    dates_coherence = nc_file.variables[variable_dates][:]
    
    # Load data ===============================================================
    x = np.array(nc_file.variables['x'])
    y = np.array(nc_file.variables['y'])
    
    # Spatial subset case
    
    if subset and isinstance(subset, (list, tuple)) and len(subset) == 4:
        x_min, x_max, y_min, y_max = subset
        
        # Ensure subset is within valid bounds
        x_min = max(x.min(), x_min)
        x_max = min(x.max(), x_max)
        y_min = max(y.min(), y_min)
        y_max = min(y.max(), y_max)
    
        # Convert coordinates to indices
        x_low_index, x_up_index, y_low_index, y_up_index = get_spatial_subset_indices(x, y, [x_min, x_max, y_min, y_max])
    
        # Extract the spatial subset
        spatial_subset = [x[x_low_index], x[x_up_index], y[y_low_index], y[y_up_index]]

    else:
        # Default: Use the full image
        spatial_subset = [x.min(), x.max(), y.min(), y.max()]

    # Find the closest dates to target_dates
    target_dates = ensure_dates_as_array( target_dates )
       
    dates = [search_date(dates_coherence, d) for d in target_dates]
    
    for i, chan in enumerate(channel_option):
        for date in dates:
            index_date = list(dates_coherence).index(date)
            variable_phase = 'S1_' + str(baseline).zfill(2) + 'days' + '_phase'
            
            if subset == False:
                image = nc_file.variables[variable_phase][index_date, chan, :, :]
            else:
                image = nc_file.variables[variable_phase][index_date, chan, y_up_index:y_low_index+1, x_low_index:x_up_index+1 ]
            
            # Masking zeros in the image
            image = np.where(image == 0, np.nan, image)
            
            # Plot image ==========================================================
            plt.figure(figsize=[9.6,6.4])
            
            plt.title(rf'{baseline}-day $\phi_{{{channel_label[i]}}}$: {date}', fontsize = 20)
            plt.imshow(image, extent = spatial_subset, vmin = -np.pi, vmax = np.pi, cmap='hsv')
            
            plt.colorbar()
            
            plt.xlabel('X-UTM (m)')
            plt.ylabel('Y-UTM (m)')
    
            # Save the image to the specified directory if indicated
            if save_images == True:            
                save_figure( save_dir, 'phase_' + str(baseline).zfill(2) +'days_' + channel_label[i] + '_' + date + '.png'  )

    # Close the file
    nc_file.close()

def compute_and_show_decomposition_images(path_dataset, target_dates, subset = False, save_dir = 'dummy', save_images = False):
    """
    Compute and visualise model-based decomposition images (RGB and HSV) from a dual-polarisation SAR covariance dataset.

    Parameters
    ----------
    path_dataset : str
        Path to the NetCDF dataset containing the C2 covariance matrices and coordinate information.
    target_dates : list of str or datetime-like
        List of target acquisition dates for which to perform the decomposition.
    subset : bool or tuple, optional
        If False, use the full spatial extent. If a tuple is provided, it should define the spatial subset
        in coordinates or indices (depending on `get_spatial_subset_indices`).
    save_dir : str, optional
        Directory where output images will be saved if `save_images` is True.
    save_images : bool, optional
        Whether to save the generated figures to `save_dir`.

    Returns
    -------
    None
        The function displays decomposition results as matplotlib figures and optionally saves them.
    
    Notes
    -----
    This function assumes a specific band order in the C2 matrix and uses the formulation in Mascolo (2022).
    It creates:
      - RGB images (mp: red, mv: green, mp-mv ratio: blue)
      - HSV images (hue: delta, saturation: cross-pol coherence, value: span)
    """
    
    # Load data ===============================================================

    ds = Dataset(path_dataset, mode='r')
           
    dates_C2 = ds.variables['times_C2'][:]

    # Parameter required 
    pol_mode = 'VV-VH'
    
    x = np.array(ds.variables['x'])
    y = np.array(ds.variables['y'])
        
    # Spatial subset case
    
    if subset == False:
        spatial_subset = [x.min(), x.max(), y.min(), y.max()]
    else:         
        x_low_index, x_up_index, y_low_index, y_up_index = get_spatial_subset_indices( x, y, subset)
        spatial_subset = [x[x_low_index], x[x_up_index], y[y_low_index], y[y_up_index]]


    # Find the closest dates to target_dates
    target_dates = ensure_dates_as_array( target_dates )
        
    dates = []
    [dates.append(search_date(dates_C2, d)) for d in target_dates]
    
       
    for date in dates:
        index_date = list(dates_C2).index(date)
        
        if subset == False:
            c2 = ds.variables['S1_C2'][index_date, :, :, :]
        else:
            c2 = ds.variables['S1_C2'][index_date, :, y_up_index:y_low_index+1, x_low_index:x_up_index+1 ]

        # Masking zeros in the image
        c2 = np.where(c2 == 0, np.nan, c2)

        # Only the area with data    
        # mask =  (data[:] != 0) 
        mask = (c2[0,:,:] != 0)
        
        # Image size
        nrows, ncols = c2.shape[1:]
        
        # Initialization
        
        mv = np.zeros( (nrows, ncols), dtype = np.float32 ) 
        mp = np.zeros( (nrows, ncols), dtype = np.float32 )
        alpha =  np.zeros( (nrows, ncols), dtype = np.float32 ) 
        delta = np.zeros( (nrows, ncols), dtype = np.float32 ) 
        
        # Elements of C2. Caution with the 1st and 2nd channel.
        # Here we follow the formulation of the paper [Mascolo 2022].
        
        if pol_mode == 'VV-VH': # 'VV-VH'
        
            C11 = c2[3,:,:] 
            C22 = c2[0,:,:] 
            C12 = c2[1,:,:] - 1j* c2[2,:,:]
            # C12 = c2[1,0,:,:]
            
        else: # 'HH-HV' or 'VH-VV'
        
            C11 = c2[0,:,:] 
            C22 = c2[3,:,:] 
            C12 = c2[1,:,:] + 1j* c2[2,:,:]
            # C12 = C2[0,1,:,:]
        
        # Build the Stokes vector from C2
        
        s1 = C11 + C22
        s2 = C11 - C22
        s3 = 2 * np.real(C12)
        s4 = 2 * np.imag(C12)
            
        # Select the case of dual-pol data
        
        if pol_mode == 'HH-HV':
            F = 1/2
        else:
            F = - 1/2
        
        ### Apply the decomposition
        
        # Solve the quadratic (volume term)
        
        a = 0.75
        b = -2 * ( s1 - (F*s2) )
        c = s1*s1 - s2*s2 - s3*s3 - s4*s4
        delta = b**2 - 4 * a * c
        
        mv1 = (-b + np.sqrt(delta)) / (2*a)
        mv2 = (-b - np.sqrt(delta)) / (2*a)
        
        # Check which solution satisfes s1 > mv and sets solution for mv

        flag1 = s1 > mv1
        
        if len(flag1) > 0:
            mv[flag1] = mv1[flag1]
        
        flag2 = s1 < mv1
        
        if len(flag2) > 0:
            mv[flag2] = mv2[flag2]

        # Obtain mp (polarized term)
        
        mp[mask] = s1[mask] - mv[mask]
        
        # Obtain alpha and delta for the polarized term
        
        if pol_mode == 'HH-HV':
            
            arg = np.clip( (s2[mask]-F*mv[mask])/mp[mask], -1, 1)
            alpha[mask] = 0.5 * np.arccos( arg )
            
        else:
            
            arg = np.clip( -(s2[mask]-F*mv[mask])/mp[mask], -1, 1)
            alpha[mask] = 0.5 * np.arccos( arg )
        
        delta[mask] = np.angle( s3[mask] + 1j*s4[mask] )

        mp_dB  = 10*np.log10( mp + 1e-12 )
        mv_dB  = 10*np.log10( mv + 1e-12 )
                        
        # Create an RGB image
           
        ratio_dB = mp_dB - mv_dB
        
        red_band = bytescl( mp_dB, [-20,5], mask )
        green_band = bytescl( mv_dB, [-20,5], mask )
        blue_band = bytescl( ratio_dB, [-10,10], mask )

        image = np.stack((red_band, green_band, blue_band), axis=-1)
        
        plt.figure( figsize=[9.6,6.4] )
        
        plt.title('Model-based decomp. RGB: ' + date, fontsize = 20 )
        plt.imshow( image, extent = spatial_subset)
        
        plt.xlabel('X-UTM (m)')
        plt.ylabel('Y-UTM (m)')
                  
        # Save the image to the specified directory if indicated
        if save_images == True:            
            save_figure( save_dir, 'ModelBasedDecomposition_RGB_' + date + '.png' )
                
        # Create an HSV inage
               
        # Hue (delta)
        h_band = np.clip( (delta + np.pi)/(2*np.pi), 0, 1)
        
        # Saturation (cross-polarized coherence)
        rho = np.zeros( (nrows, ncols), dtype = np.float32) 
        
        rho[mask] = np.abs(C12[mask]) / np.sqrt(C11[mask]*C22[mask])
        s_band = rho
        
        # Value (span):
        span_dB = 10*np.log10( C11 + C22 + 1e-12)
        v_band = np.clip( (span_dB+25.0)/30.0, 0, 1)
        
        hsv_image = np.stack((h_band, s_band, v_band), axis=-1)

        # Convert to RGB
        HSV_RGB = np.uint8( matplotlib.colors.hsv_to_rgb( hsv_image ) * 255 )
        
        plt.figure( figsize=[9.6,6.4] )
        
        plt.title('Model-based decomp. HSV: ' + date, fontsize = 20 )
        plt.imshow( HSV_RGB, extent = spatial_subset)
        
        plt.xlabel('X-UTM (m)')
        plt.ylabel('Y-UTM (m)')

        # Save the image to the specified directory if indicated
        if save_images == True:            
            save_figure( save_dir, 'ModelBasedDecomposition_HSV_' + date + '.png' )
            
    # Close the file
    ds.close()

def create_time_series(dates, images, gt, crop):
    """
    Create a time series of average and standard deviation values for a specific crop type.

    Parameters
    ----------
    dates : list of datetime.date
        List of dates corresponding to the input images.
    images : list of np.ndarray
        List of 2D arrays representing image data for each date.
    gt : np.ndarray
        4D array of ground truth masks (year, height, width) indicating crop types.
    crop : int or str
        The crop class to filter in the ground truth.

    Returns
    -------
    average : np.ndarray
        Time series of mean values computed over the masked crop area for each date.
    std : np.ndarray
        Time series of standard deviation values over the masked crop area for each date.

    Notes
    -----
    This function handles year boundaries by shifting masks if the image is from November or later,
    assuming the ground truth labels are associated with the following year's crop cycle.
    """
    
    # Create a mask for the crop
    crop_mask = gt == crop
       
    years_vs_index = {2017 : 0, 2018 : 1, 2019 : 2, 2020 : 3, 2021 : 4}
    
    masked_images = []
    
    for d, im in zip(dates, images):
        year = d.year
        
        if d.year == 2021:
            mask = crop_mask[years_vs_index[2021], :, :]     
        else:
            if d.month < 11:
                mask = crop_mask[years_vs_index[year], :, :]
            else: 
                mask = crop_mask[years_vs_index[year + 1], :, :]
                
        # Apply the corresponding mask to the image
        masked_image = np.where(mask, im, np.nan)
        masked_images.append(masked_image)
    
    masked_images = np.array(masked_images)
    
    # Compute statistics along the time axis
    average = np.nanmean(masked_images, axis=(1, 2))
    std = np.nanstd(masked_images, axis=(1, 2))
        
    return average, std

def save_time_series(path_dataset, option, date_start, date_end, crop, baseline, border_erosion_size, save_dir, show_series, save_figures = False, save_series = False):
    """
    This script generates time series plots for various crop types based on different remote sensing options
    (e.g., Optical, NDVI, Backscatter, and Coherence). The plot is adjusted with labels, legends, and multiple y-axes
    where necessary (e.g., for precipitation data). The script also provides options to save both the generated 
    figures as image files and the time series data as pickle files for future use. 
    
    The main functionality includes:
    1. **Plotting Time Series**: 
       - For each crop, the time series data is plotted with options for optical reflectance (Blue, Green, Red, NIR), NDVI, 
         backscatter (VV, VH), and coherence.
       - The x-axis represents dates, with custom formatting for date labels and tick intervals.
       - A secondary y-axis is added for precipitation data, with precipitation values shown as bars.
       
    2. **Saving Plots**:
       - Figures are saved as PNG files with high resolution in a specified directory. The directory is created if it does not exist.
       
    3. **Saving Time Series Data**:
       - Time series data for each crop is saved as a pickle file. Data is stored separately for different types (e.g., optical, NDVI, backscatter).
       - The pickle files contain the mean and standard deviation for each time series as well as the associated dates.
    
    4. **Error Handling**:
       - The script handles errors related to directory creation and figure saving, ensuring that the process is smooth.
       - Checks are in place to ensure the correct saving of time series data.
    
    Parameters:
    -----------
        show_series (bool): If True, the time series plots will be displayed.
        save_figures (bool): If True, the figures will be saved to the specified directory.
        save_series (bool): If True, the time series data will be saved as pickle files.
        crops (list): List of crop names to generate plots and save data for.
        dates (list): List of dates for the time series.
        dates_rain (list): List of dates corresponding to precipitation data.
        rain (list): Precipitation data corresponding to the dates_rain.
        option (str): The type of data to plot. Options include 'Optical', 'NDVI', 'Backscatter', and 'Coherence'.
        save_dir (str): The directory where figures and pickle files will be saved.
        ts_b_mean, ts_b_std, ts_g_mean, ts_g_std, ts_r_mean, ts_r_std, ts_nir_mean, ts_nir_std: Time series data for different bands (Blue, Green, Red, NIR).
        ts_ndvi_mean, ts_ndvi_std: Time series data for NDVI.
        ts_vv_mean, ts_vv_std, ts_vh_mean, ts_vh_std: Time series data for VV and VH backscatter.
        baseline (int): The baseline in days used for coherence calculation.
    
    Returns:
    --------
        If `option` is 'Optical':
            Returns time series data for different optical bands (Blue, Green, Red, NIR).
        If `option` is 'NDVI':
            Returns time series data for NDVI.
        If `option` is 'Backscatter' or 'Coherence':
            Returns time series data for VV and VH backscatter or coherence values.
    
    Notes:
    ------
        - Ensure that the specified directory exists or that it can be created, otherwise saving operations may fail.
        - Figures are saved with the crop name included in the filename to distinguish between different crops.
        - Time series data is saved with meaningful names indicating the type of measurement (e.g., NDVI, VV, VH).
        - The code allows for easy expansion if more remote sensing data types need to be included.
    """    
    nc_file = Dataset( path_dataset, mode='r')

    option = option.capitalize()
    if option == 'Ndvi':
        option = option.upper()

    save_dir = os.path.join( save_dir, option)
        
    if option not in ['Optical', 'NDVI', 'Backscatter', 'Coherence']:
        print('Error: incorrect option selected')
        return
    
    # Load data ===============================================================
    gt_orig = nc_file.variables['crop_type_map']    # Ground truth for crop mapping
    
    # Erosion of the crop map
    gt = erode_crop_map( gt_orig[:], border_erosion_size)
    
    crop_types = nc_file.variables['crops']    # Names of crop types
    if crop == 'all':
        crops = crop_types
    else:
        crops = crop
    
    # Ensure crops is always treated as a list
    if isinstance(crops, str):  # Check if crops is a string
        crops = [crops]  # Wrap the string in a list
        crops = [c.capitalize() for c in crops]

    # Loading dates
    if option == 'Optical':
        dates_str = nc_file.variables['times_optical'][:]    # Dates for the optical data
    
    elif option == 'NDVI':
        dates_str = nc_file.variables['times_optical'][:]    # Dates for the optical data
        
    elif option == 'Backscatter':
        dates_str = nc_file.variables['times_C2'][:]    # Dates for the backscatter data

    elif option == 'Coherence':        
        dates_str = nc_file.variables['times_coher_' + str(baseline).zfill(2) + 'days'][:]    # Dates for the coherence data

    # Find starting and ending date
    date1 = search_date(dates_str, date_start)
    pos_date_start = list(dates_str).index(date1)
    date2 = search_date(dates_str, date_end)
    pos_date_end = list(dates_str).index(date2)

    dates = [datetime.datetime.strptime(d, '%Y-%m-%d') for d in dates_str]    # Conversion to date
    dates = dates[pos_date_start:pos_date_end+1]
        
    rain = nc_file.variables['precipitation'][:]    # Load precipitation data
    dates_rain_str = nc_file.variables['times_precipitation'][:]
        
    date1_rain = search_date(dates_rain_str, date_start)
    pos_date_start_rain = list(dates_rain_str).index(date1_rain)
    date2_rain = search_date(dates_rain_str, date_end)
    pos_date_end_rain = list(dates_rain_str).index(date2_rain)
        
    dates_rain = [datetime.datetime.strptime(d, '%Y-%m-%d') for d in dates_rain_str]
    dates_rain = dates_rain[pos_date_start_rain:pos_date_end_rain+1]
    rain = rain[pos_date_start_rain:pos_date_end_rain+1]
    
    print('========================================')
        
    if option == 'Optical':     # OPTICAL TIME SERIES
    
        blue_images = nc_file.variables['S2'][pos_date_start:pos_date_end+1, 0, :, :]
        green_images = nc_file.variables['S2'][pos_date_start:pos_date_end+1, 1, :, :]
        red_images = nc_file.variables['S2'][pos_date_start:pos_date_end+1, 2, :, :]
        nir_images = nc_file.variables['S2'][pos_date_start:pos_date_end+1, 3, :, :]
        
        # Create time series for each band ========================================
        
        # Blue band
        ts_b_mean = np.zeros((len(crops), len(dates)))
        ts_b_std = np.zeros((len(crops), len(dates)))
        
        # Green band
        ts_g_mean = np.zeros((len(crops), len(dates)))
        ts_g_std = np.zeros((len(crops), len(dates)))
        
        # Red band
        ts_r_mean = np.zeros((len(crops), len(dates)))
        ts_r_std = np.zeros((len(crops), len(dates)))
        
        # NIR band
        ts_nir_mean = np.zeros((len(crops), len(dates)))
        ts_nir_std = np.zeros((len(crops), len(dates)))
                   
        for i, c in enumerate(crops):
            index_crop = list(crop_types[:]).index(c) + 1

            # The same process is repeated for blue, green, red and NIR

            ts_b_mean[i, :], ts_b_std[i, :] = create_time_series(dates, blue_images, gt[:], index_crop)
            print('Blue time series for ' + c + ' finished (' + str(i + 1) + '/' + str(len(crops)) + '). \n')
            
            ts_g_mean[i, :], ts_g_std[i, :] = create_time_series(dates, green_images, gt[:], index_crop)
            print('Green time series for ' + c + ' finished (' + str(i + 1) + '/' + str(len(crops)) + '). \n')
            
            ts_r_mean[i, :], ts_r_std[i, :] = create_time_series(dates, red_images, gt[:], index_crop)
            print('Red time series for ' + c + ' finished (' + str(i + 1) + '/' + str(len(crops)) + '). \n')
                    
            ts_nir_mean[i, :], ts_nir_std[i, :] = create_time_series(dates, nir_images, gt[:], index_crop)
            print('NIR time series for ' + c + ' finished (' + str(i + 1) + '/' + str(len(crops)) + '). \n')
            
    elif option == 'NDVI':     # NDVI TIME SERIES
    
        ndvi_images = nc_file.variables['S2'][pos_date_start:pos_date_end+1, 4, :, :]
        
        # Create time series ======================================================
        
        ts_ndvi_mean = np.zeros((len(crops), len(dates)))
        ts_ndvi_std = np.zeros((len(crops), len(dates)))      
               
        for i, c in enumerate(crops):
            index_crop = list(crop_types[:]).index(c) + 1
            ts_ndvi_mean[i, :], ts_ndvi_std[i, :] = create_time_series(dates, ndvi_images, gt[:], index_crop)
            print('NDVI time series for ' + c + ' finished (' + str(i + 1) + '/' + str(len(crops)) + ').\n')

        
    elif option == 'Backscatter':     # BACKSCATTER TIME SERIES
    
        c2_1 = nc_file.variables['S1_C2'][pos_date_start:pos_date_end+1, 0, :, :]
        c2_2 = nc_file.variables['S1_C2'][pos_date_start:pos_date_end+1, 3, :, :]
        
        # Change to logarithmic scale
        images_vv = []
        images_vh = []

        for aux1, aux2 in zip(c2_1, c2_2):
            
            image1 = np.where(aux1 == 0, np.nan, aux1)
            image1 = 10 * np.log10(image1)
            images_vv.append(image1)
            
            image2 = np.where(aux2 == 0, np.nan, aux2)
            image2 = 10 * np.log10(image2)
            images_vh.append(image2)
            
        # Create time series ======================================================
             
        ts_vv_mean = np.zeros((len(crops), len(dates)))
        ts_vv_std = np.zeros((len(crops), len(dates)))
        ts_vh_mean = np.zeros((len(crops), len(dates)))
        ts_vh_std = np.zeros((len(crops), len(dates)))

        for i, c in enumerate(crops):
            index_crop = list(crop_types[:]).index(c) + 1
            ts_vv_mean[i, :], ts_vv_std[i, :] = create_time_series(dates, images_vv, gt[:], index_crop)
            print('VV time series for ' + c + ' finished (' + str(i + 1) + '/' + str(len(crops)) + '). \n')
            ts_vh_mean[i, :], ts_vh_std[i, :] = create_time_series(dates, images_vh, gt[:], index_crop)
            print('VH time series for ' + c + ' finished (' + str(i + 1) + '/' + str(len(crops)) + '). \n')

                
    elif option == 'Coherence':     # COHERENCE TIME SERIES
    
        images_vv = nc_file.variables['S1_' + str(baseline).zfill(2) + 'days' + '_coherence'][pos_date_start:pos_date_end+1, 0, :, :]
        images_vh = nc_file.variables['S1_' + str(baseline).zfill(2) + 'days' + '_coherence'][pos_date_start:pos_date_end+1, 1, :, :]
        
        # Create time series ======================================================

        ts_vv_mean = np.zeros((len(crops), len(dates)))
        ts_vv_std = np.zeros((len(crops), len(dates)))
        ts_vh_mean = np.zeros((len(crops), len(dates)))
        ts_vh_std = np.zeros((len(crops), len(dates)))
        
        for i, c in enumerate(crops):
            index_crop = list(crop_types[:]).index(c) + 1
            ts_vv_mean[i, :], ts_vv_std[i, :] = create_time_series(dates, images_vv, gt[:], index_crop)
            print('VV time series for ' + c + ' finished (' + str(i + 1) + '/' + str(len(crops)) + '). \n')
            ts_vh_mean[i, :], ts_vh_std[i, :] = create_time_series(dates, images_vh, gt[:], index_crop)
            print('VH time series for ' + c + ' finished (' + str(i + 1) + '/' + str(len(crops)) + '). \n')

    print('========================================')
    
    # Plot the time series =================================================
    if show_series or save_figures:
        for i, c in enumerate(crops):
            
            plt.figure()
            plt.subplots_adjust(left=0.15, right=0.9, top=0.9, bottom=0.2)
            plt.title(c)
        
            if option == 'Optical':
                plt.plot(dates, ts_b_mean[i, :], color='b', label = 'Blue')
                plt.fill_between(dates, ts_b_mean[i, :] - ts_b_std[i, :], ts_b_mean[i, :] + ts_b_std[i, :], \
                                  color='b', alpha=0.1)
                
                plt.plot(dates, ts_g_mean[i, :], color='g', label = 'Green')
                plt.fill_between(dates, ts_g_mean[i, :] - ts_g_std[i, :], ts_g_mean[i, :] + ts_g_std[i, :], \
                                  color='g', alpha=0.1)
                
                plt.plot(dates, ts_r_mean[i, :], color='r', label = 'Red')
                plt.fill_between(dates, ts_r_mean[i, :] - ts_r_std[i, :], ts_r_mean[i, :] + ts_r_std[i, :], \
                                  color='r', alpha=0.1)
                
                plt.plot(dates, ts_nir_mean[i, :], color='darkred', label = 'NIR')
                plt.fill_between(dates, ts_nir_mean[i, :] - ts_nir_std[i, :], ts_nir_mean[i, :] + ts_nir_std[i, :], \
                                  color='darkred', alpha=0.1)

                plt.ylabel('Reflectance')
                
            elif option == 'NDVI':
                plt.plot(dates, ts_ndvi_mean[i, :], color='g')
                plt.fill_between(dates, ts_ndvi_mean[i, :] - ts_ndvi_std[i, :], ts_ndvi_mean[i, :] + ts_ndvi_std[i, :], \
                                 color='g', alpha=0.1, label = 'NDVI')
                
                plt.ylabel('NDVI')
                
            elif option == 'Backscatter':
                plt.plot(dates, ts_vv_mean[i, :], color='r')
                plt.fill_between(dates, ts_vv_mean[i, :] - ts_vv_std[i, :], ts_vv_mean[i, :] + ts_vv_std[i, :], \
                                 color = 'r', alpha = 0.1, label = 'VV')
            
                plt.plot(dates, ts_vh_mean[i, :], color='g')
                plt.fill_between(dates, ts_vh_mean[i, :] - ts_vh_std[i, :], ts_vh_mean[i, :] + ts_vh_std[i, :], \
                                 color='g', alpha=0.1, label = 'VH')

                plt.ylabel(r'$\sigma_{0}$ (dB)')
                
            elif option == 'Coherence':
                plt.plot(dates, ts_vv_mean[i, :], color='r')
                plt.fill_between(dates, ts_vv_mean[i, :] - ts_vv_std[i, :], ts_vv_mean[i, :] + ts_vv_std[i, :], \
                                 color='r', alpha=0.1, label = 'VV')
            
                plt.plot(dates, ts_vh_mean[i, :], color='g')
                plt.fill_between(dates, ts_vh_mean[i, :] - ts_vh_std[i, :], ts_vh_mean[i, :] + ts_vh_std[i, :], \
                                 color='g', alpha=0.1, label = 'VH')
                
                plt.ylabel(r'$\gamma$')
                
            plt.legend()
            
            # Format x-axis
            ax = plt.gca()
            ax.xaxis.set_major_locator(mdates.MonthLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
            plt.xticks(rotation=45)
            
            # Adjust x-axis limits
            combined_dates = dates + dates_rain  # Combine dates to find full range
            plt.xlim([min(combined_dates), max(combined_dates)])
            
            # Add second y-axis for precipitation
            new_ax = ax.twinx()
            
            # Convert dates_rain to Matplotlib's numeric format
            numeric_dates_rain = mdates.date2num(dates_rain)
            
            # Plot precipitation bars
            new_ax.bar(numeric_dates_rain, rain, color='lightskyblue', alpha=0.6, label='Precipitation')
            
            new_ax.set_ylabel('Precipitation (mm)')
            
            # plt.show()
    
            # Save the time series plot ===============================================
            if save_figures:
                # Ensure the save_dir exists; create it if it doesn't
                if not os.path.exists(save_dir):
                    try:
                        os.makedirs(save_dir)  # Create the directory and any necessary parent directories
                        print(f'Created directory: {save_dir}')
                    except OSError as myerror:
                        print(f'Failed to create directory: {save_dir}. Error: {myerror}')
                        return  # Exit the function if the directory cannot be created
                    
                # Save the figure
                if option == 'Optical':
                    save_path = os.path.join(save_dir, f'ts_optical_plot_{c}.png')
                elif option == 'NDVI':
                    save_path = os.path.join(save_dir, f'time_series_NDVI_plot_{c}.png')
                elif option == 'Backscatter':
                    save_path = os.path.join(save_dir, f'time_series_sigma0_plot_{c}.png')
                elif option == 'Coherence':
                    save_path = os.path.join(save_dir, f'time_series_coherence_plot_{c}_{baseline}days.png')
                    
                try:
                    plt.savefig(save_path, bbox_inches='tight', dpi=300)  # Save with high resolution
                    print(f'Figure successfully saved to {save_path}')
                except Exception as myerror:
                    print(f'Failed to save figure: {save_path}. Error: {myerror}')
                    
            else:
                
                plt.show()
    
    # Save the time series to a pickle file ==============================
    
    if save_series:
                
        # Check and create directory if it doesn't exist
        if not os.path.exists(os.path.dirname(save_dir)):
            try:
                os.makedirs(os.path.dirname(save_dir), exist_ok=True)
                print(f'Created drirectory: {save_dir}')
            except OSError as e:
                print(f'Failed to create directory: {save_dir}. Error: {e}')
                return  # Exit the function if the directory cannot be created

        # Save the data into a pickle file
        if option == 'Optical':
            for i, c in enumerate(crops):
                crop_data = {
                    'blue_mean': ts_b_mean[i, :],
                    'blue_std': ts_b_std[i, :],
                    'green_mean': ts_g_mean[i, :],
                    'green_std': ts_g_std[i, :],
                    'red_mean': ts_r_mean[i, :],
                    'red_std': ts_r_std[i, :],
                    'nir_mean': ts_nir_mean[i, :],
                    'nir_std': ts_nir_std[i, :],
                    'dates' : dates
                }
                
                save_file = os.path.join(save_dir, f'time_series_optical_{c}.pkl')
                
                with open(save_file, 'wb') as f:
                    pickle.dump(crop_data, f)
                print(f'Data successfully saved to: {save_file}')
                
        elif option == 'NDVI':
            for i, c in enumerate(crops):
                crop_data = {
                            'ndvi_mean': ts_ndvi_mean[i, :],
                            'ndvi_std': ts_ndvi_std[i, :],
                            'dates' : dates
                }
                save_file = os.path.join(save_dir, f'time_series_NDVI_{c}.pkl')
                
                with open(save_file, 'wb') as f:
                    pickle.dump(crop_data, f)
                print(f'Data successfully saved to: {save_file}')

        elif option == 'Backscatter':
            for i, c in enumerate(crops):

                crop_data = {
                    'vv_mean': ts_vv_mean[i, :],
                    'vv_std': ts_vv_std[i, :],
                    'dates': dates
                    }
                    
                save_file = os.path.join(save_dir, f'time_series_sigma0_VVchannel_{c}.pkl')

                with open(save_file, 'wb') as f:
                    pickle.dump(crop_data, f)
                print(f'Data successfully saved to: {save_file}')

                crop_data = {
                    'vh_mean': ts_vh_mean[i, :],
                    'vh_std': ts_vh_std[i, :],
                    'dates': dates
                    }
                    
                save_file = os.path.join(save_dir, f'time_series_sigma0_VHchannel_{c}.pkl')

                with open(save_file, 'wb') as f:
                    pickle.dump(crop_data, f)
                print(f'Data successfully saved to: {save_file}')
                
        elif option == 'Coherence':
            for i, c in enumerate(crops):
                
                crop_data = {
                   'vv_mean': ts_vv_mean[i, :],
                   'vv_std': ts_vv_std[i, :],
                   'dates': dates
                   }
                
                save_file = os.path.join(save_dir, f'time_series_coherence_VVchannel_{c}_{baseline}days.pkl')

                with open(save_file, 'wb') as f:
                    pickle.dump(crop_data, f)
                print(f'Data successfully saved to: {save_file}')

                crop_data = {
                   'vh_mean': ts_vh_mean[i, :],
                   'vh_std': ts_vh_std[i, :],
                   'dates': dates
                   }
                
                save_file = os.path.join(save_dir, f'time_series_coherence_VHchannel_{c}_{baseline}days.pkl')

                with open(save_file, 'wb') as f:
                    pickle.dump(crop_data, f)
                print(f'Data successfully saved to: {save_file}')
                
    # Close the NetCDF file
    nc_file.close()
    
    if option == 'Optical':
        return ts_b_mean, ts_b_std, ts_g_mean, ts_g_std, ts_r_mean, ts_r_std, ts_nir_mean, ts_nir_std, dates
    
    elif option == 'NDVI':
        return ts_ndvi_mean, ts_ndvi_std, dates
    
    else:
        return ts_vv_mean, ts_vv_std, ts_vh_mean, ts_vh_std, dates   

def save_time_series_decomposition(path_dataset, date_start, date_end, crop, border_erosion_size, save_dir, show_series = True, save_figures = False, save_series = False):
    """
    Processes and saves time series data for different crop types, including multiple variables (e.g., mv, mp, alpha, delta, rvi).
    The function generates and optionally saves plots for the data and saves the processed time series data as pickle files.
    
    Parameters:
    -----------
    crops : list of str
        List of crop names for which the time series data is to be processed.
    ts_mv_mean, ts_mv_std, ts_mp_mean, ts_mp_std, ts_alpha_mean, ts_alpha_std, ts_delta_mean, ts_delta_std, ts_rvi_mean, ts_rvi_std : 
        ndarray
        Arrays containing the mean and standard deviation for the time series data of different variables (e.g., mv, mp, alpha, delta, rvi).
        Each array should have a shape where the first dimension corresponds to the crop index and the second dimension corresponds to time steps.
    dates : list of datetime
        List of dates corresponding to the time steps for which the time series data is available.
    dates_rain : list of datetime
        List of dates corresponding to the precipitation data.
    rain : list of float
        List of precipitation values for the corresponding `dates_rain`.
    save_figures : bool
        Flag indicating whether to save the generated plots as images.
    save_series : bool
        Flag indicating whether to save the processed time series data as pickle files.
    save_dir : str
        Directory path where the plots and data files should be saved.
    save_figure : function, optional
        Optional function to save a figure (default is None).
    nc_file : NetCDF4.Dataset, optional
        A NetCDF4 file object to close after processing, default is None.
    
    Returns:
    --------
    ts_mv_mean, ts_mv_std, ts_mp_mean, ts_mp_std, ts_alpha_mean, ts_alpha_std, ts_delta_mean, ts_delta_std, ts_rvi_mean, ts_rvi_std, dates : 
        ndarray, list of datetime
        Processed time series data arrays for different variables and the list of dates.
    
    Notes:
    ------
    - The function generates a plot for each crop, showing time series for the mean and standard deviation of variables such as mv, mp, alpha, delta, and rvi.
    - Precipitation data is plotted as bars using a secondary y-axis.
    - If `save_figures` is set to True, the generated plots are saved as PNG files.
    - If `save_series` is set to True, the processed time series data for each crop is saved as pickle files.
    - The function ensures the specified directory exists before attempting to save files and handles errors when creating directories.
    - The function closes the NetCDF4 file (`nc_file`) if provided.
    """
    
    nc_file = Dataset(path_dataset, mode='r')

    save_dir = os.path.join( save_dir, 'Decomposition')

 
    # Load data ===============================================================
    gt_orig = nc_file.variables['crop_type_map']    # Ground truth for crop mapping
    
    # Erosion of the crop map
    gt = erode_crop_map( gt_orig[:], border_erosion_size)
    
    crop_types = nc_file.variables['crops']    # Names of crop types
    if crop == 'all':
        crops = crop_types
    else:
        crops = crop
    
    # Ensure crops is always treated as a list
    if isinstance(crops, str):  # Check if crops is a string
        crops = [crops]  # Wrap the string in a list
        crops = [c.capitalize() for c in crops]

    
    dates_str = nc_file.variables['times_C2'][:]  # Dates for the optical data
    
    # Find starting and ending date
    date1 = search_date(dates_str, date_start)
    pos_date_start = list(dates_str).index(date1)
    date2 = search_date(dates_str, date_end)
    pos_date_end = list(dates_str).index(date2)

    dates = [datetime.datetime.strptime(d, '%Y-%m-%d') for d in dates_str]    # Conversion to date
    dates = dates[pos_date_start:pos_date_end+1]
        
    rain = nc_file.variables['precipitation'][:]    # Load precipitation data
    dates_rain_str = nc_file.variables['times_precipitation'][:]
        
    date1_rain = search_date(dates_rain_str, date_start)
    pos_date_start_rain = list(dates_rain_str).index(date1_rain)
    date2_rain = search_date(dates_rain_str, date_end)
    pos_date_end_rain = list(dates_rain_str).index(date2_rain)
        
    dates_rain = [datetime.datetime.strptime(d, '%Y-%m-%d') for d in dates_rain_str]
    dates_rain = dates_rain[pos_date_start_rain:pos_date_end_rain+1]
    rain = rain[pos_date_start_rain:pos_date_end_rain+1]
            
    print('========================================')
    
    images_mv = []
    images_mp = []
    images_alpha = []
    images_delta = []
    images_rvi = []

    C11all = nc_file.variables['S1_C2'][pos_date_start:pos_date_end+1, 3, :, :] 
    C22all = nc_file.variables['S1_C2'][pos_date_start:pos_date_end+1, 0, :, :] 
    C12all = nc_file.variables['S1_C2'][pos_date_start:pos_date_end+1, 1, :, :] \
        - 1j*nc_file.variables['S1_C2'][pos_date_start:pos_date_end+1, 2, :, :]

    C11all = np.where(C11all == 0, np.nan, C11all)
    C22all = np.where(C11all == 0, np.nan, C22all)
    C12all = np.where(C11all == 0, np.nan, C12all)

    ndates = C11all.shape[0]
    
    for idate in range(ndates):
        
        C11 = C11all[idate,:,:]
        C22 = C22all[idate,:,:]
        C12 = C12all[idate,:,:]
        
        mask = (C11 != 0)
    
        # Build the Stokes vector from C2
        
        s1 = C11 + C22
        s2 = C11 - C22
        s3 = 2 * np.real(C12)
        s4 = 2 * np.imag(C12)
            
        # Select the case of dual-pol data
        
        F = - 1/2
    
    
        ### Apply the decomposition
        
        mv = np.zeros_like( C11 )
        mp = np.zeros_like( C11 )
        alpha = np.zeros_like( C11 )
        
        # Solve the quadratic (volume term)
        
        a = 0.75
        b = -2 * ( s1 - (F*s2) )
        c = s1*s1 - s2*s2 - s3*s3 - s4*s4
        delta = b**2 - 4 * a * c
        
        mv1 = (-b + np.sqrt(delta)) / (2*a)
        mv2 = (-b - np.sqrt(delta)) / (2*a)
        
        # Check which solution satisfes s1 > mv and set solution for mv
        
        flag1 = s1 > mv1
        
        if len(flag1)>0:
            mv[flag1] = mv1[flag1]
        
        flag2 = s1 < mv1
        
        if len(flag2)>0:
            mv[flag2] = mv2[flag2]
        
        # Obtain mp (polarized term)
        
        mp[mask] = s1[mask] - mv[mask]
    
        arg = np.clip( -(s2[mask]-F*mv[mask])/mp[mask], -1, 1)
        alpha[mask] = 0.5 * np.arccos( arg )
            
        delta[mask] = np.angle( s3[mask] + 1j*s4[mask] )
    
        mp_dB  = 10*np.log10( mp + 1e-12 )
        mv_dB  = 10*np.log10( mv + 1e-12 )
    
        rvi = mv / ( mp + mv + 1e-12 )
    
        # Append images
        
        images_mv.append( mv_dB )
        images_mp.append( mp_dB )
        images_alpha.append( alpha )
        images_delta.append( delta )
        images_rvi.append( rvi )


            
    # Create time series ======================================================
                
    ts_mv_mean = np.zeros((len(crops), len(dates)))
    ts_mv_std = np.zeros((len(crops), len(dates)))
    ts_mp_mean = np.zeros((len(crops), len(dates)))
    ts_mp_std = np.zeros((len(crops), len(dates)))
    ts_alpha_mean = np.zeros((len(crops), len(dates)))
    ts_alpha_std = np.zeros((len(crops), len(dates)))
    ts_delta_mean = np.zeros((len(crops), len(dates)))
    ts_delta_std = np.zeros((len(crops), len(dates)))
    ts_rvi_mean = np.zeros((len(crops), len(dates)))
    ts_rvi_std = np.zeros((len(crops), len(dates)))
   

    for i, c in enumerate(crops):
        
        code_crop = list(crop_types[:]).index(c) + 1

        ts_mv_mean[i, :], ts_mv_std[i, :] = create_time_series(dates, images_mv, gt[:], code_crop)
        print('mv time series for ' + c + ' finished (' + str(i + 1) + '/' + str(len(crops)) + '). \n')

        ts_mp_mean[i, :], ts_mp_std[i, :] = create_time_series(dates, images_mp, gt[:], code_crop)
        print('mp time series for ' + c + ' finished (' + str(i + 1) + '/' + str(len(crops)) + '). \n')

        ts_alpha_mean[i, :], ts_alpha_std[i, :] = create_time_series(dates, images_alpha, gt[:], code_crop)
        print('alpha time series for ' + c + ' finished (' + str(i + 1) + '/' + str(len(crops)) + '). \n')

        ts_delta_mean[i, :], ts_delta_std[i, :] = create_time_series(dates, images_delta, gt[:], code_crop)
        print('delta time series for ' + c + ' finished (' + str(i + 1) + '/' + str(len(crops)) + '). \n')

        ts_rvi_mean[i, :], ts_rvi_std[i, :] = create_time_series(dates, images_rvi, gt[:], code_crop)
        print('rvi time series for ' + c + ' finished (' + str(i + 1) + '/' + str(len(crops)) + '). \n')
        
    print('========================================')
    
    # Plot the time series ====================================================
    if show_series or save_figures:
        for i, c in enumerate(crops):
            
            # mv, mp
            
            plt.figure()
            plt.subplots_adjust(left=0.15, right=0.9, top=0.9, bottom=0.2)
            plt.title(c)

            
            plt.plot(dates, ts_mp_mean[i, :], color='r', label = '$m_p$')
            plt.fill_between(dates, ts_mp_mean[i, :] - ts_mp_std[i, :], ts_mp_mean[i, :] + ts_mp_std[i, :], \
                              color = 'r', alpha = 0.1)
        
            plt.plot(dates, ts_mv_mean[i, :], color='g', label = '$m_v$')
            plt.fill_between(dates, ts_mv_mean[i, :] - ts_mv_std[i, :], ts_mv_mean[i, :] + ts_mv_std[i, :], \
                              color = 'g', alpha = 0.1)

            plt.ylabel(r'$\sigma_{0}$ (dB)')
            plt.legend()
            
            plt.legend()
            
            # Format x-axis
            ax = plt.gca()
            ax.xaxis.set_major_locator(mdates.MonthLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
            plt.xticks(rotation=45)
            
            # Adjust x-axis limits
            combined_dates = dates + dates_rain  # Combine dates to find full range
            plt.xlim([min(combined_dates), max(combined_dates)])
            
            # Add second y-axis for precipitation
            new_ax = ax.twinx()
            
            # Convert dates_rain to Matplotlib's numeric format
            numeric_dates_rain = mdates.date2num(dates_rain)
            
            # Plot precipitation bars
            new_ax.bar(numeric_dates_rain, rain, color='lightskyblue', alpha=0.6, label='Precipitation')
            
            new_ax.set_ylabel('Precipitation (mm)')
            
            #plt.show()
            
            # Save the time series plot 
            
            if save_figures:
                save_figure( save_dir, f'time_series_mv_mp_plot_{c}.png' )
            else:
                plt.show()
                
            # rvi
           
            plt.figure()
            plt.subplots_adjust(left=0.15, right=0.9, top=0.9, bottom=0.2)
            plt.title(c)
             
            plt.plot(dates, ts_rvi_mean[i, :], color='g', label = 'RVI-dec')
            plt.fill_between(dates, ts_rvi_mean[i, :] - ts_rvi_std[i, :], ts_rvi_mean[i, :] + ts_rvi_std[i, :], \
                              color = 'g', alpha = 0.1)
         
            plt.ylabel(r'RVI')
            
            plt.legend()
           
            # Format x-axis
            ax = plt.gca()
            ax.xaxis.set_major_locator(mdates.MonthLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
            plt.xticks(rotation=45)
            
            # Adjust x-axis limits
            combined_dates = dates + dates_rain  # Combine dates to find full range
            plt.xlim([min(combined_dates), max(combined_dates)])
            
            # Add second y-axis for precipitation
            new_ax = ax.twinx()
            
            # Convert dates_rain to Matplotlib's numeric format
            numeric_dates_rain = mdates.date2num(dates_rain)
            
            # Plot precipitation bars
            new_ax.bar(numeric_dates_rain, rain, color='lightskyblue', alpha=0.6, label='Precipitation')
            
            new_ax.set_ylabel('Precipitation (mm)')
            
            # plt.show()
            
            # Save the time series plot 
            
            if save_figures:
                save_figure( save_dir, f'time_series_rvi_dec_plot_{c}.png' )
            else:
                plt.show()
                 
            # alpha
                
            plt.figure()
            plt.subplots_adjust(left=0.15, right=0.9, top=0.9, bottom=0.2)
            plt.title(c)
             
            plt.plot(dates, ts_alpha_mean[i, :], color='b', label = 'alpha')
            plt.fill_between(dates, ts_alpha_mean[i, :] - ts_alpha_std[i, :], ts_alpha_mean[i, :] + ts_alpha_std[i, :], \
                              color = 'b', alpha = 0.1)
         
            plt.ylabel(r'alpha')
            
            plt.legend()
           
            # Format x-axis
            ax = plt.gca()
            ax.xaxis.set_major_locator(mdates.MonthLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
            plt.xticks(rotation=45)
            
            # Adjust x-axis limits
            combined_dates = dates + dates_rain  # Combine dates to find full range
            plt.xlim([min(combined_dates), max(combined_dates)])
            
            # Add second y-axis for precipitation
            new_ax = ax.twinx()
            
            # Convert dates_rain to Matplotlib's numeric format
            numeric_dates_rain = mdates.date2num(dates_rain)
            
            # Plot precipitation bars
            new_ax.bar(numeric_dates_rain, rain, color='lightskyblue', alpha=0.6, label='Precipitation')
            
            new_ax.set_ylabel('Precipitation (mm)')
            
            # plt.show()
           
            # Save the time series plot 
            
            if save_figures:
                save_figure( save_dir, f'time_series_alpha_plot_{c}.png' )
            else:
                plt.show()
                
        
    # Save the time series ==============================
    if save_series:
        # Check and create directory if it doesn't exist
        if not os.path.exists(os.path.dirname(save_dir)):
            try:
                os.makedirs(os.path.dirname(save_dir), exist_ok=True)
                print(f'Created drirectory: {save_dir}')
            except OSError as e:
                print(f'Failed to create directory: {save_dir}. Error: {e}')
                return  # Exit the function if the directory cannot be created
        
        # Save the data
        for i, c in enumerate(crops):
                 
            # mv
            
            crop_data = {
                'mv_mean': ts_mv_mean[i, :],
                'mv_std': ts_mv_std[i, :],
                'dates': dates
                }
            
            save_file = os.path.join(save_dir, f'time_series_mv_{c}.pkl')
            with open(save_file, 'wb') as f:
                pickle.dump(crop_data, f)
            print(f'Data successfully saved to: {save_file}')
            
            # mp
            
            crop_data = {
                'mp_mean': ts_mp_mean[i, :],
                'mp_std': ts_mp_std[i, :],
                'dates': dates
                }

            save_file = os.path.join(save_dir, f'time_series_mp_{c}.pkl')
            with open(save_file, 'wb') as f:
                pickle.dump(crop_data, f)
            print(f'Data successfully saved to: {save_file}')
                
            # alpha
            
            crop_data = {
                'alpha_mean': ts_alpha_mean[i, :],
                'alpha_std': ts_alpha_std[i, :],
                'dates': dates
                }
            
            save_file = os.path.join(save_dir, f'time_series_alpha_{c}.pkl')
            with open(save_file, 'wb') as f:
                pickle.dump(crop_data, f)
            print(f'Data successfully saved to: {save_file}')

            # delta
            
            crop_data = {
                'delta_mean': ts_delta_mean[i, :],
                'delta_std': ts_delta_std[i, :],
                'dates': dates
                }
            
            save_file = os.path.join(save_dir, f'time_series_delta_{c}.pkl')
            with open(save_file, 'wb') as f:
                pickle.dump(crop_data, f)
            print(f'Data successfully saved to: {save_file}')
            
            # rvi
            
            crop_data = {
                'rvi_mean': ts_rvi_mean[i, :],
                'rvi_std': ts_rvi_std[i, :],
                'dates': dates
                }
            
            save_file = os.path.join(save_dir, f'time_series_rvi_{c}.pkl')
            with open(save_file, 'wb') as f:
                pickle.dump(crop_data, f)
            print(f'Data successfully saved to: {save_file}')


    # Close the NetCDF file
    nc_file.close()

    return ts_mv_mean, ts_mv_std, ts_mp_mean, ts_mp_std, ts_alpha_mean, ts_alpha_std, ts_delta_mean, ts_delta_std, ts_rvi_mean, ts_rvi_std, dates

def get_full_path(option, main_folder, c, channel=None, baseline=None):
    """
    Generate the full path for a time series data file based on the given option and parameters.
    
    Parameters:
    option (str): The type of data to generate the path for. Options include 'NDVI', 'Backscatter', and 'Coherence'.
    main_folder (str): The main directory where the data is stored.
    c (str): The crop name or identifier.
    channel (str, optional): The channel identifier for 'Backscatter' or 'Coherence'. Defaults to None.
    baseline (int, optional): The baseline duration for 'Coherence' data, in days. Defaults to None.
    
    Returns:
    str: The full file path for the specified time series data.
    """
    
    if option == 'NDVI':
        return os.path.join(main_folder, option, f'time_series_NDVI_{c}.pkl')
    elif option == 'Backscatter':
        return os.path.join(main_folder, option, f'time_series_sigma0_{channel}channel_{c}.pkl')
    elif option == 'Coherence':
        return os.path.join(main_folder, option, f'time_series_coherence_{channel}channel_{c}_{baseline}days.pkl')

def generate_axis_label(option, channel, baseline):
    """
    Generate the label for the y-axis of a plot based on the given option and parameters.
    
    Parameters:
    option (str): The type of data for which the label is generated. Options include 'NDVI', 'Backscatter', and 'Coherence'.
    channel (str): The channel identifier for 'Backscatter' or 'Coherence'.
    baseline (int): The baseline duration for 'Coherence' data, in days.
    
    Returns:
    str: The label for the y-axis corresponding to the specified option, channel, and baseline.
    """
    
    if option == 'NDVI':
        label = 'NDVI'
    elif option == 'Backscatter':
        label = r'$\sigma_{0}$' + channel
    elif option == 'Coherence':
        label = r'$\gamma$' + channel + ', baseline = ' + str(baseline) + 'days'
    return label

def load_series_data(full_path, option, channel):
    """
    Load time series data from a pickle file and extract the relevant series and dates.
    
    Parameters:
    full_path (str): The full path to the pickle file containing the time series data.
    option (str): The type of data to load. Options include 'NDVI', 'Backscatter', and 'Coherence'.
    channel (str): The channel identifier for 'Backscatter' or 'Coherence'.
    
    Returns:
    tuple: A tuple containing the time series data (series) and corresponding dates.
    """
    # Load data
    with open(full_path, 'rb') as file:
        data = pickle.load(file)
        
    # Load series data
    if option != 'NDVI':
        series = data[channel.lower() + '_mean']
    else:
        series = data['ndvi_mean']
    dates = data['dates']
    
    return series, dates

def compare_saved_time_series( main_folder, option1, option2, crops, channel1, channel2, baseline1, baseline2, show_scatterplot = False):
    """
    Compare two time series data sets saved in files and calculate the correlation (R²) and RMSE between them.
    
    The function loads two sets of time series data for different options (e.g., NDVI, backscatter, etc.) for different crops,
    interpolates the second series to match the dates of the first, and then computes the correlation and RMSE between them.
    A scatterplot of the comparison can also be shown.

    Parameters:
    ----------
    main_folder : str
        Path to the main folder where the time series data is stored.
    option1 : str
        The first option (e.g., 'NDVI', 'Backscatter', 'Coherence') to compare.
    option2 : str
        The second option (e.g., 'NDVI', 'Backscatter', 'Coherence') to compare.
    crops : list or str
        List of crops to consider for the comparison. If 'all', the function will use a predefined list of crops.
    channel1 : str
        The channel for the first option (e.g., 'VV', 'VH', etc.).
    channel2 : str
        The channel for the second option (e.g., 'VV', 'VH', etc.).
    baseline1 : str
        The baseline for the first option.
    baseline2 : str
        The baseline for the second option.
    show_scatterplot : bool, optional
        Whether to display a scatterplot of the comparison. Default is False.

    Returns:
    -------
    None
        The function does not return any value but prints the results (R² and RMSE) to the console.
        If `show_scatterplot` is True, a scatterplot will also be shown.
    """
    
    option1 = option1.capitalize()
    option2 = option2.capitalize()
    
    option1, option2 = [opt.upper() if opt.lower() == 'ndvi' else opt for opt in (option1, option2)]
    
    valid_options = {'NDVI', 'Backscatter', 'Coherence'}
    if option1 not in valid_options or option2 not in valid_options:
        print('At least one of the options for the time series is incorrect')
        return
    
    if crops == 'all': # List copied here to avoid adding an extra input parameter (dataset)
        crops = ['Alfalfa','Barley','Carrot','Chickpea','Cotton','Fallow',
                 'Maize','Oats','Onion','Pepper','Potato','Pumpkin','Quinoa',
                 'Rice','SugarBeet','Sunflower' 'SweetPotato' 'Tomato' 'Wheat']
            
    if isinstance(crops, str):
        crops = [crops]  # Wrap the string in a list if there is only 1 crop
    crops = [c.capitalize() for c in crops]
    
    for i, c in enumerate(crops):

        full_path1 = get_full_path(option1, main_folder, c, channel1, baseline1)
        full_path2 = get_full_path(option2, main_folder, c, channel2, baseline2)

        # Loading time series
        series1, dates1 = load_series_data(full_path1, option1, channel1)
        series2, dates2 = load_series_data(full_path2, option2, channel2)

        # Find the common date range
        dates1_array = np.array(dates1)
        dates2_array = np.array(dates2)
        start_date = max(dates1_array.min(), dates2_array.min())
        end_date = min(dates1_array.max(), dates2_array.max())
    
        # Mask the dates and series to the common range
        mask1 = (dates1_array >= start_date) & (dates1_array <= end_date)
        mask2 = (dates2_array >= start_date) & (dates2_array <= end_date)
    
        dates1_common = dates1_array[mask1]
        dates1_common.sort()
        series1_common = series1[mask1]
        
        dates2_common = dates2_array[mask2]
        
        # Adding previous and following date from the second dates list if necessary
        if start_date not in dates2_array:
            index = np.squeeze(np.where(dates2_array == dates2_common[0]))
            
            if index > 0:
                np.insert(dates2_common, 0, dates2_array[index-1])
        
        if end_date not in dates2_array:
            index = np.squeeze(np.where(dates2_array == dates2_common[-1]))
            
            if index < len(dates2_array) - 1:
                np.insert(dates2_common, -1, dates2_array[index+1])

        dates2_common.sort()
        series2_common = series2[mask2]
    
        # Interpolate the data from the seconds series to match dates of the first series
        dates2_common_numeric = mdates.date2num(dates2_common)
        dates1_common_numeric = mdates.date2num(dates1_common)
        interpolator = interp1d(dates2_common_numeric, series2_common, kind='linear', bounds_error=False, fill_value=np.nan)
        series2_interpolated = interpolator(dates1_common_numeric)
    
        # Remove NaN values resulting from interpolation
        valid_mask = (~np.isnan(series2_interpolated) * ~np.isnan(series1_common))
        series1_final = series1_common[valid_mask]
        series2_final = series2_interpolated[valid_mask]
    
        # Compute R^2 and RMSE
        slope, intercept, r_value, p_value, std_err = linregress(series1_final, series2_final)
        r2 = r_value**2  # r_value is the Pearson correlation coefficient
        # r2 = r2_score(series1_final, series2_final)
        predicted_series2 = slope * series1_final + intercept
        rmse = np.sqrt(np.mean((series2_final - predicted_series2) ** 2))
        
        label1 = generate_axis_label(option1, channel1, baseline1)
        label2 = generate_axis_label(option2, channel2, baseline2)
        
        # Show the scatterplot if indicated
        if show_scatterplot:
            plt.figure()
            plt.title(r'Correlation between ' + label1 + ' and ' + label2 + ' for ' + c)
            plt.scatter(series1_final, series2_final)
                   
            # Plot the fit line
            plt.plot(series1_final, predicted_series2, label=f'Fit line (R²={r2:.2f}, RMSE={rmse:.2f})', color='red')
        
            plt.xlabel(label1)
            plt.ylabel(label2)
        
            plt.legend()
        print(f'Correlation between {label1} and {label2}: ')
        print(f'R² = {r2:.2f}, RMSE = {rmse:.2f}')
           
def compare_time_series( dates1, series1, dates2, series2):
    """
    Compare two time series data sets and calculate the correlation (R²) and RMSE between them.

    The function finds the common date range between two time series, interpolates the second series to match the dates of the first,
    and then computes the correlation and RMSE between them.

    Parameters:
    ----------
    dates1 : list or array
        List or array of dates for the first time series.
    series1 : list or array
        List or array of values for the first time series.
    dates2 : list or array
        List or array of dates for the second time series.
    series2 : list or array
        List or array of values for the second time series.

    Returns:
    -------
    r2 : float
        The R² value (Pearson correlation squared) between the two time series.
    rmse : float
        The RMSE (Root Mean Square Error) between the two time series.
    """
    
    # Find the common date range
    dates1_array = np.array(dates1)
    dates2_array = np.array(dates2)
    start_date = max(dates1_array.min(), dates2_array.min())
    end_date = min(dates1_array.max(), dates2_array.max())

    # Mask the dates and series to the common range
    mask1 = (dates1_array >= start_date) & (dates1_array <= end_date)
    mask2 = (dates2_array >= start_date) & (dates2_array <= end_date)

    dates1_common = dates1_array[mask1]
    dates1_common.sort()
    s1 = np.squeeze(series1)
    series1_common = s1[mask1]
    
    dates2_common = dates2_array[mask2]
    
    # Adding previous and following date from the second dates list if necessary
    if start_date not in dates2_array:
        index = np.squeeze(np.where(dates2_array == dates2_common[0]))
        
        if index > 0:
            np.insert(dates2_common, 0, dates2_array[index-1])
    
    if end_date not in dates2_array:
        index = np.squeeze(np.where(dates2_array == dates2_common[-1]))
        
        if index < len(dates2_array) - 1:
            np.insert(dates2_common, -1, dates2_array[index+1])

    dates2_common.sort()
    s2 = np.squeeze(series2)
    series2_common = s2[mask2]

    # Interpolate the data from the seconds series to match dates of the first series
    dates2_common_numeric = mdates.date2num(dates2_common)
    dates1_common_numeric = mdates.date2num(dates1_common)
    interpolator = interp1d(dates2_common_numeric, series2_common, kind='linear', bounds_error=False, fill_value=np.nan)
    series2_interpolated = interpolator(dates1_common_numeric)

    # Remove NaN values resulting from interpolation or in the input series
    valid_mask = (~np.isnan(series2_interpolated) * ~np.isnan(series1_common))
    series1_final = series1_common[valid_mask]
    series2_final = series2_interpolated[valid_mask]


    # Compute R^2 and RMSE

    slope, intercept, r_value, p_value, std_err = linregress(series1_final, series2_final)

    r2 = r_value**2  # r_value is the Pearson correlation coefficient

    predicted_series2 = slope * series1_final + intercept

    rmse = np.sqrt(np.mean((series2_final - predicted_series2) ** 2))
        
    return r2, rmse    
           
    
