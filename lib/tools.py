import rasterio
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot
from pathlib import Path

#
# Generate lists of train/validate targets from the data we have available
#

source = Path('/home/usgs/landsat5')

def getsets(n=0,p=0.1,rand=True):
    """Return a tuple of training and validation file names.  The training set will be of size (1-p)*n, and the validation set will be p*n.
    If n is not specified, all the available data will be used.  If rand is true, the files are selected and ordered randomly, otherwise
    they will be repeatable (providing the data in the directory hasn't changed)"""
    allfiles = list(source.glob('*.tif'))
    if rand:
        np.random.shuffle(allfiles)
    if n==0:
        n = len(allfiles)
    
    nv = max(1, int(n*p))
    return (allfiles[nv:n], allfiles[0:nv])


#
# Tools for quickly using pyplot with huge and multi-dimensional data
#

def smaller(ds,factor=64):
    """Return an array of a smaller size to read into.  Used to get a 'decimated read' from GDAL.
    Note: assumes all bands are the same type."""
    return np.empty(shape=(ds.count, ds.width//factor, ds.height//factor), dtype=ds.dtypes[0])

def show_thumbnail(rfp,band=1,level=64):
    """Show a thumbnail (overview) from the provided open rasterio file pointer."""   
    # Note: gdal will use a "decimated read" to compute the requested size on the fly if it doesn't exist
    # as an explicit thumbnail
    thumb = rfp.read(band, out_shape=(1, rfp.height//level, rfp.width//level))
    pyplot.imshow(thumb)

# Pattern: if you want to do a *single image* at a larger size, you have to add a subplot to do it.
# pyplot.figure(figsize=(15,15)).add_subplot(111).imshow(...)

def pltnoaxes(plt):
    return plt.tick_params(axis='both', left=False, bottom=False, labelleft=False, labelbottom=False)

def plotsize(nrows, ncols, basesize=20):   
    # "20" seems to be a good width for a jupyter screen, for some reason
    # ncols then sets the overall (sub)figure size, and the calculation detemines
    # the height needed to accomodate that.  Assumes approximately square tiles.
    return (20, (20//ncols) * nrows)

def showbands(filep, ncols=3, basesize=20, level=64):
    """Show all the bands in a geotiff file pointer as subplots.  Automatically resizes them down by
    specified level (set level=1 if you want full size)"""
    n = filep.count
    nrows = (n//ncols) + (1 if n%ncols else 0)
    fig = pyplot.figure(figsize=plotsize(nrows,ncols,basesize))
    dat = filep.read(out=smaller(filep,level))
    for i in range(0, n):
        plt = fig.add_subplot( nrows, ncols, i+1 )
        pltnoaxes(plt)
        plt.imshow( dat[i] )
    fig.show()

def showaxes(arry, ncols=3, basesize=20):
    """Show each of the bands of a 3d array as subplots.  No resizing is done."""
    n = arry.shape[0]
    nrows = (n//ncols) + (1 if n%ncols else 0)
    fig = pyplot.figure(figsize=plotsize(nrows,ncols,basesize))
    for i in range(0, n):
        plt = fig.add_subplot( nrows, ncols, i+1 )
        pltnoaxes(plt)
        plt.imshow( arry[i] )
    fig.show()

def showbits(dat, n=8, ncols=3, basesize=20):
    """Show the unpacked bits of a numpy array (or geotiff band) as subplots.  n controls how many bits
    to unpack"""
    nrows = (n//ncols) + (1 if n%ncols else 0)
    fig = pyplot.figure(figsize=plotsize(nrows,ncols,basesize))
    for i in range(0, n):
        plt = fig.add_subplot( nrows, ncols, i+1 )
        pltnoaxes(plt)
        mask = 1<<i
        plt.imshow( (dat&mask) >> i, cmap="Greys" )
    fig.show()

