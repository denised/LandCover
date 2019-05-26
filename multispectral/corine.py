import rasterio
from rasterio import windows
import numpy as np
from pathlib import Path
from . import coords
from . import bands


#The directory where we keep the corine dataset, projected into UTM
corine_directory = Path("/home/firewise/corine")

def set_corine_directory(p):
    global corine_directory
    corine_directory = p

counts = {'lsempty':0,'mapoverlap':0,'cempty':0,'dataoverlap':0,'good':0}
tile_counts = {}
def reset_counts():
    global tile_counts
    tile_counts = {}
def count_plus(tile,ctype):
    if tile not in tile_counts.keys():
        tile_counts[tile] = counts.copy()
    tile_counts[tile][ctype] += 1

    
def bi(srcname,band):
    # syntactic sugar to make the code more readable
    # convert logical name of band to index
    src = bands.LANDSAT_BANDS if srcname == 'landsat' else bands.CORINE_BANDS
    return bands.band_index(src,band) 

def corine_attributes():
    """Return the c and classes attributes required by fastai"""
    cs = bands.CORINE_BANDS
    return (len(cs), cs)

def corine_labeler(lsdat, region:windows.Window):
    """Merge a region of a landsat tile with its equivalent region of the Corine dataset to create a src, target pair for learning.

    lsdat: a rasterio dataset object for a landsat tile
    region: a rasterio.windows.Window object
    The results are returned as two numpy arrays.
    
    Merging does three things: identifies the corresponding part of the corine dataset, merges the nodata regions of 
    both datasets, and unpacks the cloud and cloud shadow data from the landsat data and adds it as additional target 
    classes to the corine data"""

    # get the equivalent pixel boundaries for the matching corine dataset
    corine = fetch_corine(lsdat.crs)
    geo_span = coords.pixel_to_geo(lsdat,region)
    corine_region = coords.geo_to_pixel(corine,geo_span)

    # read the two datasets, and do various checks that they properly overlap
    # and have actual data in them.

    # get landsat dataset.  if it is empty, return None
    ls_data = lsdat.read(window=region)
    qa = ls_data[bi('landsat','qa')] 
    ls_nodata = (qa == 0)  # for some reason mask bit isn't always there, 
                           # but this works
    # if the entire window is nodata, return None
    if ls_nodata.all():
        count_plus(lsdat.name,'lsempty')
        return None
    
    # get the corine dataset.
    corine_actual_region = coords.pixel_window_intersect(corine,corine_region)  # account for partially- or non-overlapping datasets
    if corine_actual_region is None:
        count_plus(lsdat.name,'mapoverlap')
        return None
    c_data = corine.read(window=corine_actual_region)
    c_nodata = c_data[ bi('corine','mask') ]
    if c_nodata.all():
        count_plus(lsdat.name,'cempty')
        return None
    # possibly pad back
    c_data = coords.pad_dataset_to_window(c_data, corine_actual_region, corine_region)

    # Now start transfering data between the two data sets.
    # add the cloud and shadow bitmaps to the corine data
    cloud = ((qa & bands.PIXEL_QA['cloud']) != 0)
    shadow = ((qa & bands.PIXEL_QA['shadow']) != 0)
    c_data = np.append( c_data, [cloud, shadow], axis=0 )

    ls_data = np.delete(ls_data, bi('landsat','qa'), axis=0)  # we're done with the qa band

    # Figure out the combined NODATA
    either_nodata = np.logical_or(ls_nodata, np.logical_not(c_nodata))
    if either_nodata.all():  # they don't have *any* overlap
        count_plus(lsdat.name,'dataoverlap')
        return None
    
    count_plus(lsdat.name,'good')
    
    # otherwise, synchronize the two
    if np.any(ls_nodata):  # propagate to corine
        c_data[:, either_nodata] = 0
        c_data[ bi('corine','mask') ] = 255 * np.logical_not(either_nodata)  # put mask back
    if np.any(c_nodata):   # propagate to tile
        ls_data[:, either_nodata] = 0
    
    # finally convert to 0..1 floats
    # both ls and c data are bytes with full range.
    return (ls_data.astype(float)/255, c_data.astype(float)/255)

_corine_open_datasets = {}

def fetch_corine(crs):
    """Return a rasterio dataset for Corine reprojected into the specified crs."""

    epsg = str(crs.to_epsg())
    # Have we already opened it?
    if epsg in _corine_open_datasets.keys():
        corine = _corine_open_datasets[epsg]
        # check that it is still open
        if corine.closed:
            del(_corine_open_datasets[epsg])
            # fall through
        else:
            return corine

    # Do we already have a saved reprojection?
    corine_name = corine_directory / ("corine_" + epsg + ".tif")
    if not corine_name.exists():
        raise Exception('Corine projection {} not found!'.format(epsg)) 
    
    _corine_open_datasets[epsg] = rasterio.open(corine_name)
    return _corine_open_datasets[epsg]
