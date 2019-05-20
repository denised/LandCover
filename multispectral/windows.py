from typing import *
import torch
import numpy
import itertools
import rasterio
 
"""A DataSet designed to enumerate windows over large raster data (e.g. satellite images)."""

# The type of a labeler callback.
Labeler = Callable[[rasterio.io.DatasetReader,rasterio.windows.Window],Optional[Tuple[numpy.ndarray,numpy.ndarray]]]

class WindowedDataset(torch.utils.data.Dataset):
    def __init__(self, images:Iterable[str], labeler:Labeler, window_size:int=256, size:int=65535):
        """
        images: an iterable that returns paths to the satellite tiles (in any format accepted by rasterio.open)
        labeler: func(rasterio_dataset, window) -> (x,y) or None
            A function that produces the x and y values for the specific window, or None if this window should be skipped
        window_size: generate windows of data of this size (ws x ws pixels).  Will perform best if window size == tif block size
        size: typically we won't know how many samples we will produce.  size is what we return when asked this impertinent question.
        """
        self.ws = (window_size,window_size)
        self.labeler = labeler
        self.size = size
        self.im = None
        self.im_iter = itertools.cycle(images)
        self.curr_image: Optional[rasterio.io.DatasetReader] = None

    """For performance reasons, we completely subvert the random access API of Dataset.
    Every call to __getitem__ will get the next item in the list, ignoring the requested index"""
    # Future improvements:
    # Open multiple files at once and round robin between them, to mix the data up more.

    def __len__(self):
        """We don't know the size of the dataset, so return a very large number"""
        return self.size
    
    def next_window(self) -> Tuple[rasterio.io.DatasetReader, rasterio.windows.Window]:
        """Returns an img, window pair"""
        # Advance the image iterator, if necessary.
        if self.im is None:
            self.im = rasterio.open(next(self.im_iter))
            self.w_iter = window_iterator(self.im.shape, self.ws)
        try:
            window = next(self.w_iter)
        except Exception:   # TODO: iterator expired exception only
            self.im = None
            return self.next_window()
        else:
            return (self.im, window)
    
    def next_sample(self) -> Tuple[numpy.ndarray, numpy.ndarray]:
        """Returns the next valid sample"""
        s = None
        while s is None:
            s = self.labeler(*self.next_window())
        return s

    def __getitem__(self, index=0):
        # Completely ignore index.
        return self.next_sample()


def window_iterator(full_size, window_size, stride=0) -> Iterator[rasterio.windows.Window]:
    """Return windows of size window_size over an array of size full_size.
    Both full_size and window_size are (x,y) tuples.
    If stride is provided, each window is moved by that amount.  Stride may either be
    an integer or an (x,y) tuple.  By default, stride is equal to window_size"""
    
    if stride == 0:
        stride = window_size
    if isinstance(stride,int):
        stride = (stride,stride)

    maxx = full_size[0] - window_size[0]
    maxy = full_size[1] - window_size[1]

    i = 0
    j = 0
    while i <= maxx:
        while j <= maxy:
            yield rasterio.windows.Window(j,i,*window_size)
            j = j+stride[0]
        j = 0
        i = i+stride[1]
