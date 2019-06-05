from typing import Union, Sequence, Optional, Tuple
from rasterio import windows
import rasterio
import numpy as np

DataFile = rasterio.io.DatasetReader
PixelWindow = windows.Window
GeoWindow = rasterio.coords.BoundingBox

# Rasterio has a class Window that represents spans of a dataset in pixel space,
# and a class BoundingBox that can represents a span in Geo Space.
# Here we add some utility functions for using them and converting between them

# Notes to self: for pixels, x and y aren't what you think they are.
#
#   win = windows.Window(0,0,width=3,height=5)
#   dat = fp.read(1,window=win)
#   dat.shape
#   -> (5,3)
#
#  that is, the order of dimensions is height,width, not width,height
#
#  1st dimension ~ rows ~ height ~ upper/lower ~ y on the map
#  2nd dimension ~ columns ~ width ~ left/right ~ x on the map


def datafile_pixel_window(datafile:DataFile) -> PixelWindow:
    return windows.Window(0, 0, width=datafile.width, height=datafile.height)

def datafile_geo_window(datafile:DataFile) -> GeoWindow:
    return datafile.bounds

def pixel_to_geo(datafile:DataFile, window:PixelWindow) -> GeoWindow:
    """Return the geo BoundingBox corresponding to the pixel window on the datafile"""
    return rasterio.coords.BoundingBox(*windows.bounds(window,datafile.transform))

def geo_to_pixel(datafile:DataFile, boundingbox:GeoWindow, fixed_size:Optional[Tuple[int,int]]=None) -> PixelWindow:
    """Return the pixel Window corresponding to a geographic range in this datafile.
    Note there is no guarantee the Window is in the range covered by datafile.
    If fixed_size is provided, that is used as the size (width, height) of the result; this may be needed to avoid rounding issues
    in the floating point arithmetic."""
    result = datafile.window(*boundingbox)
    if fixed_size:
        result = rasterio.windows.Window(result.col_off, result.row_off, *fixed_size)
    return result.round_lengths().round_offsets()


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

        # function to create a block of size height x width x num_bands properly initialized with the pad_values
        make_block = lambda height, width, pv=pad_value : np.ones([height,width,len(pv)]) * pv

        # now check each side to see if we need to pad it.
        ((desired_y_low,desired_y_high),(desired_x_low,desired_x_high)) = desired_window.toranges()
        ((actual_y_low,actual_y_high),(actual_x_low,actual_x_high)) = actual_window.toranges()
        
        if desired_x_low < actual_x_low:  # pad on the left
            delta = actual_x_low - desired_x_low
            block = make_block( dataset.shape[0], delta )
            dataset = np.concatenate((block,dataset), axis=1)

        if desired_x_high > actual_x_high:  # pad on the right
            delta = desired_x_high - actual_x_high
            block = make_block( dataset.shape[0], delta )
            dataset = np.concatenate((dataset, block), axis=1)

        if desired_y_low < actual_y_low:  # pad above
            delta = actual_y_low - desired_y_low
            block = make_block( delta, dataset.shape[1] )
            dataset = np.concatenate((block, dataset), axis=0)

        if desired_y_high > actual_y_high:  # pad below
            delta = desired_y_high - actual_y_high
            block = make_block( delta, dataset.shape[1] )
            dataset = np.concatenate((dataset, block), axis=0)
        
        # restore the band to first axis
        return np.moveaxis(dataset,-1,0)

def padded_read(fp: rasterio.io.DatasetReader, region: PixelWindow, band: Optional[int]=None, pad_value:Union[int,Sequence[int]]=0) -> np.ndarray:
    """Read the specified region from the rasterio dataset.  If any part of the region is out of bounds of the dataset, that part is padded
    with the padding value.  If band is supplied, only that band is read, otherwise all bands are read."""
    available_region = pixel_window_intersect(fp,region)
    dat = fp.read(indexes=band, window=available_region)
    # pad_dataset_to_window assumes we are dealing with multiple bands, and it would be tricky to change that.
    # instead, we just add-then-remove an extra level in the case that we are dealing with a single band
    if band:
        dat = dat[None]  # add an extra dimension
    dat = pad_dataset_to_window(dat, available_region, region, pad_value)
    if band:
        dat = dat[0]
    return dat 
