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

    # get the two datasets
    ls_data = lsdat.read(window=region)
    c_data = corine.read(window=corine_region)

    # get the landsat QA data
    qa = ls_data[bi('landsat','qa')]

    # add the cloud and shadow bitmaps to the corine data
    cloud = ((qa & bands.PIXEL_QA['cloud']) != 0)
    shadow = ((qa & bands.PIXEL_QA['shadow']) != 0)
    c_data = np.append( c_data, [cloud, shadow], axis=0 )

    ls_data = np.delete(ls_data, bi('landsat','qa'), axis=0)  # we're done with the qa band

    # Figure out the combined NODATA
    ls_nodata = (qa == 0)  # for some reason mask bit isn't always there, 
                           # but this works
    c_nodata = c_data[ bi('corine','mask') ]
    either_nodata = np.logical_or(ls_nodata, np.logical_not(c_nodata))

    # if there is *no* common data, this isn't a good dataset to use; return None
    # TODO: could do some of this testing earlier, for efficiency
    if either_nodata.all():
        #print('skipping')
        return None
    
    if np.any(ls_nodata):  # propagate to corine
        c_data[:, either_nodata] = 0
        c_data[ bi('corine','mask') ] = 255 * np.logical_not(either_nodata)  # put mask back
    if np.any(c_nodata):   # propagate to tile
        ls_data[:, either_nodata] = 0
    
    # finally convert to 0..1 floats
    # both ls and c data are bytes with full range.
    return (ls_data/255.0, c_data/255.0)

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
