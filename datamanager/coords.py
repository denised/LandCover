from collections import namedtuple
from rasterio import windows
from rasterio import coords

# Rasterio has a class Window that represents spans of a dataset in pixel space,
# and a class BoundingBox that can represents a span in Geo Space.
# Here we add some utility functions to convert between them

def pixel_to_geo(dataset, window):
    """Return the geo BoundingBox corresponding to the window on the dataset"""
    (xrange, yrange) = window.toranges()
    upperleft = dataset.xy(xrange[0],yrange[0])
    lowerright = dataset.xy(xrange[1],yrange[1])
    return coords.BoundingBox(left=upperleft[0],right=lowerright[0],top=upperleft[1],bottom=lowerright[1])

def geo_to_pixel(dataset, boundingbox):
    """Return the pixel Window corresponding to a geographic range in this dataset.
    The range is represented by a rasterio.coords.BoundingBox"""
    upperleft = dataset.index(boundingbox.left,boundingbox.top)
    lowerright = dataset.index(boundingbox.right,boundingbox.bottom)
    return windows.Window.from_slices(rows=(upperleft[0],lowerright[0]),cols=(upperleft[1],lowerright[1]))

# General creation for a Window:
# windows.Window(x,y,width,height)
