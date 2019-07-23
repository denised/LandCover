from typing import Union
import numpy as np
from torch.tensor import Tensor
from matplotlib import pyplot
from matplotlib import patches
from matplotlib.cm import get_cmap
from matplotlib.colors import ListedColormap
import cycler
import numbers


# ###################################################################################################################
#
# Tools for using pyplot with huge and multi-dimensional data (by downscaling it)
# These are primarily meant for looking at entire landsat tiles.
#
# Convention used throughout: we never set the figure width.  Instead, we set the height to be compatible with the width
# You can use the function tools.set_figure_width(n) to set the figure width however you like.
# Setting it to 20 seems to work well in Jupyter (which is odd, since it is supposed to be inches?)
# You can also use the matplotlib context manager like this, if you want to override it only temporarily:
#  with pyplot.style.context({'figure.figsize': (x,y)}):
#     do my thing
#
# ###################################################################################################################

def set_figure_width(n):
    pyplot.rcParams["figure.figsize"] = (n,n)

def _plotstyle(ncols, nrows, imsize):
    """Return the style parameters for plotting images in ncols x nrows grid, each of which is imsize."""
    (width, _) = pyplot.rcParams["figure.figsize"]
    height = (width/ncols) * (imsize[0]/imsize[1]) * nrows
    return {
        "figure.figsize": (width, height),
        "xtick.bottom" : False,
        "xtick.labelbottom" : False,
        "ytick.left": False,
        "ytick.labelleft": False
    }

# Showing rectangles: this is used to show windows on giant landsat images, and maybe other things.

def _showrects(plt, rects, level):
    """Shows the rectangle(s) on the plot, which has been reduced by level."""
    # the rects argument may be either None, a single tuple, or a list of tuples;
    # normalize it to a list of tuples
    if rects is None or len(rects) == 0:
        rects = []
    elif isinstance(rects[0], numbers.Number):  # single tuple
        rects = [rects]
   
    for r in rects:
        r = [ i / level for i in r ]   # take level into account
        plt.add_patch(patches.Rectangle((r[0],r[1]),r[2],r[3],facecolor="none",edgecolor="r"))

def showband(filep, band, level=64, showrect=None):
    fig = pyplot.figure() 
    dat = filep.read(band,out=_smaller(filep,level)[0])
    # For reasons I don't grok, pyplot treats image size differently when we have a multi-plot display
    # vs just a single one.  So we pretend we are going to have two side by side to get it to do the right thing.
    # The result is that it is half the size you would expect, but you can always up the size temporarily.
    # Hacky, but not worth figuring it out right now.
    with pyplot.style.context(_plotstyle(2,1,filep.shape)):
        plt = fig.add_subplot(1,2,1)
        plt.imshow(dat)
        if showrect:
            _showrects(plt, showrect, level) 
        fig.show()

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
                _showrects(plt, showrect, level)
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


def _smaller(ds,factor=64):
    """Return an array of a smaller size to read into.  Used to get a 'decimated read' from GDAL.
    Note: assumes all bands are the same dtype."""
    return np.empty(shape=(ds.count, ds.height//factor, ds.width//factor), dtype=ds.dtypes[0])


# ###################################################################################################################
# 
# Useful statistical data about imaages and sets of images
# Each image is represented as an array of bands (satellite bands if input images, land-use classes if output)
# These functions are generally meant for windowed data; they may be slow for full landsat images
# Most of these functions will operate on either a single image or an array of them.
#
# ###################################################################################################################

# Utilities

# unfortunately there's no way to capture that these have 3 or 4 dimensions...
ImageOrImages = Union[np.ndarray, Tensor]

def _is_image_array(imgs: ImageOrImages):
    """Return true if the array represents a *set* of images.  A set of images has 4 dimensions; a single image has 3."""
    return len(imgs.shape) == 4

def _as_numpy(x: ImageOrImages):
    return x.numpy() if isinstance(x,Tensor) else x

# colormap for categorical data (i.e. the output of collapse_bands).  This works up to 21 categories and the first value is black
ccmap = ListedColormap( [np.array([0,0,0,1])] + list(get_cmap('tab20', 20).colors) )
ccmap_vmax = 21  # specifying vmin, vmax to imshow will keep color use consistent across images that have different actual ranges
cc_cycle = cycler.cycler(color=ccmap.colors)  # matching color cycler for plots


# In the functions below, we are generally singling out the dimension corresponding to 'band' for special treatment.
# This is dimension 0 for a single image or dimension 1 for a set of images.

def collapse_bands(imgs: ImageOrImages):
    """For each image in imgs, replace the set of bands with a single band whose pixel values select the highest-valued source band
    For classifications, this means something like selecting the most probable class for each pixel.  Not useful on satellite image data."""
    
    # First we have to rejigger the mask band or it will override all the other bands.  What we really want is it's inverse (1 where nodata),
    # but it turns out it is sufficient to set it to a very small value uniformly (all the other bands will be zero if nodata)
    imgs = _as_numpy(imgs).copy()
    if _is_image_array(imgs):
        imgs[:,0] = 0.0001
    else:
        imgs[0] = 0.0001

    axis = 1 if _is_image_array(imgs) else 0
    return np.argmax(_as_numpy(imgs), axis=axis)

def plot_categorical_img(img):
    """Plot an array interpreted as a segmented categories: each pixel should be an integer value, and the values are assigned unique colors."""
    fig, ax = pyplot.subplots()
    ax.imshow(img, interpolation='nearest', cmap=ccmap, vmin=0, vmax=ccmap_vmax)
    fig.show()

_statpoints = [0, 0.2, 0.5, 0.8, 1.0]  # quantiles returned by image-stats
def image_stats(imgs: ImageOrImages):
    """Return a set of quantile values for each band across an image or set of images."""
    axes = (0, 2, 3) if _is_image_array(imgs) else (1, 2)
    return np.quantile(_as_numpy(imgs), _statpoints, axis=axes).transpose()

def plot_image_stats(imgs: ImageOrImages, band_labels = None):
    """Show the statistics for the image/image set.  Set band_labels to a list of band names, e.g. bands.CORINE_BANDS"""
    stats = list(image_stats(imgs))
    if band_labels is None:
        band_labels = range(len(stats))
    with pyplot.style.context({'axes.prop_cycle': cc_cycle}):
        fig, ax = pyplot.subplots()
        for blab, bstat in zip(band_labels, stats):
            ax.plot(_statpoints, bstat, 's-', label=blab)
        ax.legend()
        fig.show()
    