from typing import Union, Sequence
from rasterio import windows
from rasterio import coords
import rasterio
import numpy as np

DataFile = rasterio.io.DatasetReader
PixelWindow = windows.Window
GeoWindow = coords.BoundingBox

# Rasterio has a class Window that represents spans of a dataset in pixel space,
# and a class BoundingBox that can represents a span in Geo Space.
# Here we add some utility functions for using them and converting between them
# One thing to keep in mind is that the y direction is opposite:
# Geo goes from small at the bottom (south) to large at the top (north), 
# while pixels go the other way.

def datafile_pixel_window(datafile:DataFile) -> PixelWindow:
    return windows.Window(0,0,*datafile.shape)

def datafile_geo_window(datafile:DataFile) -> GeoWindow:
    return datafile.bounds

def pixel_to_geo(datafile:DataFile, window:PixelWindow) -> GeoWindow:
    """Return the geo BoundingBox corresponding to the pixel window on the datafile"""
    return coords.BoundingBox(*windows.bounds(window,datafile.transform))

def geo_to_pixel(datafile:DataFile, boundingbox:GeoWindow) -> PixelWindow:
    """Return the pixel Window corresponding to a geographic range in this datafile.
    Note there is no guarantee the Window is in the range covered by datafile"""
    upperleft = datafile.index(boundingbox.left,boundingbox.top)
    lowerright = datafile.index(boundingbox.right,boundingbox.bottom)

    return windows.Window(*upperleft,lowerright[0]-upperleft[0],lowerright[1]-upperleft[1])


def pixel_window_intersect(datafile:DataFile, window:PixelWindow) -> PixelWindow:
    """Return the intersection of the window with the dataset (use to avoid reading out of bounds)"""
    try:
        isect = datafile_pixel_window(datafile).intersection(window)
    except rasterio.errors.WindowError:
        return None
    else:
        return isect


def pad_dataset_to_window(dataset:np.ndarray, actual_window:PixelWindow, desired_window:PixelWindow, pad_value:Union[int,Sequence[int]]=0) -> np.ndarray:
    """Given an array corresponding to actual_window, return another array corresponding to desired_window, padding any values that
    are outside actual_window's range.
    The pad value is either a single value, or one value per band"""

    if actual_window == desired_window:  # nothing needs to be done
        return dataset
    else: # create a new dataset that is appropriately extended
        # make sure pad_value is what we want
        num_bands = dataset.shape[0]
        if np.isscalar(pad_value):
            pad_value = [pad_value] * num_bands
        if len(pad_value) != num_bands:
            raise ValueError("pad_value must match number of bands")

        # It is a lot easier to do fill operations we need to do if we move bands to be the last axis instead of the first
        # We willmove it back at the end.
        dataset = np.moveaxis(dataset,0,-1)

        # function to create a block of size width x height x num_bands properly initialized with the pad_values
        make_block = lambda width, height, pv=pad_value : np.ones([width,height,len(pv)]) * pv

        # now check each side to see if we need to pad it.
        ((desired_y_low,desired_y_high),(desired_x_low,desired_x_high)) = desired_window.toranges()
        ((actual_y_low,actual_y_high),(actual_x_low,actual_x_high)) = actual_window.toranges()
        
        if desired_x_low < actual_x_low:  # pad on the left
            delta = actual_x_low - desired_x_low
            block = make_block( delta, dataset.shape[1] )
            dataset = np.concatenate((block,dataset), axis=0)

        if desired_x_high > actual_x_high:  # pad on the right
            delta = desired_x_high - actual_x_high
            block = make_block( delta, dataset.shape[1] )
            dataset = np.concatenate((dataset, block), axis=0)

        if desired_y_low < actual_y_low:  # pad above
            delta = actual_y_low - desired_y_low
            block = make_block( dataset.shape[0], delta )
            dataset = np.concatenate((block, dataset), axis=1)

        if desired_y_high > actual_y_high:  # pad below
            delta = desired_y_high - actual_y_high
            block = make_block( dataset.shape[0], delta )
            dataset = np.concatenate((dataset, block), axis=1)
        
        # restore the band to first axis
        return np.moveaxis(dataset,-1,0)




