import numpy as np
from matplotlib import pyplot
from matplotlib import patches
from pathlib import Path
import numbers

#
# Tools for quickly using pyplot with huge and multi-dimensional data
#
# Convention used throughout: we never set the figure width.  Instead, we set the height to be compatible with the width
# You can use the function tools.set_figure_width(n) to set the figure width however you like.
# Setting it to 20 seems to work well in Jupyter (which is odd, since it is supposed to be inches?)
# You can also use the matplotlib context manager like this, if you want to override it only temporarily:
#  with pyplot.style.context(figsize=(x,y)):
#     do my thing


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