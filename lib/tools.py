import numpy as np
from matplotlib import pyplot
from matplotlib import patches
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

def _smaller(ds,factor=64):
    """Return an array of a smaller size to read into.  Used to get a 'decimated read' from GDAL.
    Note: assumes all bands are the same dtype."""
    return np.empty(shape=(ds.count, ds.height//factor, ds.width//factor), dtype=ds.dtypes[0])

# Pattern: if you want to do a *single image* at a larger size, you have to add a subplot to do it.
# pyplot.figure(figsize=(15,15)).add_subplot(111).imshow(...)

def _plotstyle(ncols, nrows, imsize):
    """Return the style parameters for plotting images in ncols x nrows grid, each of which is imsize."""
    width = 20  # for some reason 20 seems to be a good setting in jupyter notebooks.  weird because it is supposed to be inches?
    height = (width/ncols) * (imsize[0]/imsize[1]) * nrows
    return {
        "figure.figsize": (width, height),
        "xtick.bottom" : False,
        "xtick.labelbottom" : False,
        "ytick.left": False,
        "ytick.labelleft": False,
        "image.aspect" : "auto"
    }

def showbands(filep, ncols=3, level=64, showrect=None):
    """Show all the bands in a geotiff file pointer as subplots.  Automatically resizes them down by
    specified level (set level=1 if you want full size)
    showrect, if provided, is a rectangle to highlight, in the form of a tuple (x_offset, y_offset, width, height)"""
    n = filep.count
    nrows = (n//ncols) + (1 if n%ncols else 0)
    with pyplot.style.context(_plotstyle(ncols,nrows,filep.shape)):
        fig = pyplot.figure()
        dat = filep.read(out=_smaller(filep,level))
        for i in range(0, n):
            plt = fig.add_subplot( nrows, ncols, i+1 )
            plt.imshow( dat[i] )
            if showrect:
                # Rectangle expects coordinates from bottom left, not the top. (oh, and the shape is y,x instead of x,y)
                r = list(showrect)
                r[1] = filep.shape[0] - r[1]
                r = [ i / level for i in r ]
                plt.add_patch(patches.Rectangle((r[0],r[1]),r[2],r[3],facecolor="none",edgecolor="r"))
        fig.show()

def showarry(arry, ncols=3):
    """Show each of the bands of a 3d array as subplots.  No resizing is done."""
    n = arry.shape[0]
    nrows = (n//ncols) + (1 if n%ncols else 0)
    with pyplot.style.context(_plotstyle(ncols, nrows, arry.shape[1:])):
        fig = pyplot.figure()
        for i in range(0, n):
            plt = fig.add_subplot( nrows, ncols, i+1 )
            plt.imshow( arry[i] )
        fig.show()

def showbits(dat, n=8, ncols=3):
    """Show the unpacked bits of a numpy array (or geotiff band) as subplots.  n controls how many bits
    to unpack"""
    nrows = (n//ncols) + (1 if n%ncols else 0)
    with pyplot.style.context(_plotstyle(ncols, nrows, dat.shape)):
        fig = pyplot.figure()
        for i in range(0, n):
            plt = fig.add_subplot( nrows, ncols, i+1 )
            mask = 1<<i
            plt.imshow( (dat&mask) >> i, cmap="Greys" )
        fig.show()

