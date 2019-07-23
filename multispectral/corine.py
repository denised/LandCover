import rasterio
from rasterio import windows
import numpy as np
from pathlib import Path
from . import coords
from . import bands
from pathlib import Path
   
def bi(srcname,band):
    # syntactic sugar to make the code more readable
    # convert logical name of band to 0-based index
    src = bands.LANDSAT_BANDS if srcname == 'landsat' else bands.CORINE_BANDS
    return bands.band_index(src,band)

def bn(srcname,band):
    # return band number (one-based, as used in landsat data)
    return bi(srcname,band)+1

def corine_filter(lsdat:rasterio.io.DatasetReader, region:windows.Window):
    """Determine whether the given region of a landsat tile has valid data and corresponds to a valid part of the corine dataset"""

    shortname = Path(lsdat.name).stem
    #print('checking {} {} '.format(shortname, region),end='')
    # does the landsat tile have any data?
    ls_qa = coords.padded_read(lsdat, region, bn('landsat','qa'))
    if not np.any(ls_qa):
        _count(shortname,'lsempty')
        return False
    
    geo_span = coords.pixel_to_geo(lsdat,region)
    corine = fetch_corine(lsdat.crs)
     
    # does the landsat tile intersect the corine map at all?
    if coords.geo_window_intersect(geo_span,corine.bounds) is None:
        _count(shortname,'nomapoverlap')
        return False
    
    # is the corresponding corine data empty?
    corine_region = coords.geo_to_pixel(corine, geo_span, fixed_size=(region.width,region.height))  
    c_mask = coords.padded_read(corine,corine_region,bn('corine','mask'))
    if not np.any(c_mask):
        _count(shortname,'cempty')
        return False
    
    # finally, is the overlapping data of enough size to interest us?
    both_data = np.logical_and(ls_qa, c_mask)
    if np.count_nonzero(both_data) < both_data.size*0.2:
        _count(shortname,'nodataoverlap')
        return False
    
    _count(shortname,'good')
    return True

def corine_labeler(lsdat:rasterio.io.DatasetReader, region:windows.Window):
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
    corine_region = coords.geo_to_pixel(corine, geo_span, fixed_size=(region.width,region.height))

    # get landsat dataset.
    ls_data = coords.padded_read(lsdat, region)
    qa = ls_data[bi('landsat','qa')] 
    ls_nodata = (qa == 0)
    ls_data = np.delete(ls_data, bi('landsat','qa'), axis=0)  # we don't need the qa band in the ls_data

    # get the corine dataset.
    c_data = coords.padded_read(corine, corine_region) 
    c_nodata = (c_data[bi('corine','mask')] == 0)

    # Unpack the cloud and shadow data from the landsat data and transfer it as channels to the target data
    cloud = ((qa & bands.PIXEL_QA['cloud']) != 0)
    shadow = ((qa & bands.PIXEL_QA['shadow']) != 0)
    c_data = np.append( c_data, [cloud, shadow], axis=0 )

    # Synchronize the nodata in both
    either_nodata = np.logical_or(ls_nodata, c_nodata)
    if np.any(ls_nodata):  # propagate to corine
        c_data[:, either_nodata] = 0
    if np.any(c_nodata):   # propagate to tile
        ls_data[:, either_nodata] = 0
    
    # finally convert to 0..1 floats
    # both ls and c data are bytes with full range.
    return (ls_data.astype(np.dtype('float32'))/255, c_data.astype(np.dtype('float32'))/255)


def corine_attributes():
    """Return the c and classes attributes required by fastai"""
    cs = bands.CORINE_BANDS
    return (len(cs), cs)

#The directory where we keep the corine dataset, projected into UTM
_corine_directory = Path("/home/firewise/corine")
_corine_open_datasets = {}

def set_corine_directory(p):
    global _corine_directory
    _corine_directory = Path(p)

def fetch_corine(crs) -> rasterio.io.DatasetReader:
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

    # Do we  have a saved reprojection?
    corine_name = _corine_directory / ("corine_" + epsg + ".tif")
    if not corine_name.exists():
        raise Exception('Corine projection {} not found!'.format(epsg)) 
    
    _corine_open_datasets[epsg] = rasterio.open(corine_name)
    return _corine_open_datasets[epsg]


# Some bookkeeping to track what we're seeing in the data.
_per_tile_counts = {'lsempty':0,'nomapoverlap':0,'cempty':0,'nodataoverlap':0,'good':0}
_tile_counts = {}
def reset_counts():
    global _tile_counts
    _tile_counts = {}
def _count(tile,ctype):
    if tile not in _tile_counts.keys():
        _tile_counts[tile] = _per_tile_counts.copy()
    _tile_counts[tile][ctype] += 1
    #print(ctype)
