from typing import *
import torch
import torch.utils.data
import numpy
import rasterio
 
"""A DataSet designed to enumerate windows over large raster data (e.g. satellite images).
There are two parts to this process:
1) Filtering, which takes a windowing strategy and generates a list of good windows (skipping non-interesting ones), and
2) The Dataset proper, which accesses data from a filtered list.
The two may be chained together in real time, but more often filtering will be done first and the results saved for re-use.
Utility functions help with this task.
"""

Window = rasterio.windows.Window
WindowList = Iterable[Tuple[rasterio.io.DatasetReader, Window]]  
WindowSize = Tuple[int,int]
Windower = Callable[[WindowSize,WindowSize], Iterable[Window]]  # callback generates candidate window instances
Filterer = Callable[[rasterio.io.DatasetReader, Window], bool]   # filterer determines which windows are 'good'
Labeler = Callable[[rasterio.io.DatasetReader, Window], Tuple[numpy.ndarray,numpy.ndarray]]  # labeler gets the x,y data for a good window

    
def window_iterator(full_size:WindowSize, window_size:WindowSize, stride=0) -> Iterator[rasterio.windows.Window]:
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


def prefilter(images:Iterable[str], check:Filterer, generator:Windower=window_iterator, window_size:WindowSize=(256,256)) -> WindowList:
    """Given an iterable list of paths for the data tiles, return a list of the usable Windows.
    images: an iterable that returns paths to the satellite tiles (in any format accepted by rasterio.open)
    filter: func(rasterio_dataset, window) -> bool
       A function that returns true if the window is a 'good' window (in bounds, has data, etc.)
    generator: func(full_size, window_size) -> iterable(Window)
       A function that generates windows of size window_size over a space of size full_size.  By default a simple sliding
       window with no overlap is used
    window_size: the size of the windows to generate.  Performance will be best if the size if the tif file block size"""
    for image in images:
        fp = rasterio.open(image)
        for win in generator(fp.shape,window_size):
            if check(fp,win):
                yield (fp, win)

# TODO: round-robin filter generator

def to_file(windows:WindowList, file_name:str) -> None:
    with open(file_name,'w') as fp:
        for (rfp,win) in windows:
            print("{},{},{},{},{}".format(rfp.name,win.col_off,win.row_off,win.width,win.height), file=fp)

_open_files = {}
def from_file(file_name:str) -> WindowList:
    """Return an iterable list of windows from a file stored as a csv.  Uses/keeps a reusable list of open rasterio file pointers."""
    with open(file_name,'r') as fp:
        for line in fp:
            (name,col_off,row_off,width,height) = line.split(',')
            if name not in _open_files.keys():
                _open_files[name] = rasterio.open(name)
            yield (_open_files[name], Window(int(col_off),int(row_off),int(width),int(height)))


class WindowedDataset(torch.utils.data.dataset.Dataset):
    def __init__(self, windows:WindowList, labeler:Labeler, c:int, classes:Tuple[str]):
        """
        windows: an iterator over the windows into raster files.  No assumption is made about the ordering of windows: they may be out of order with
            respect to the underlying raster file, or interleaved across multiple raster files, etc.  Performance will be better if ordering
            within a raster file is maintained.
        labeler: func(rasterio_dataset, window) -> (x,y)  Generate the actual x and y data for this window.
        c, classes: the attributes required by fastai
        """
        # Here we materialize the WindowList iterator into a list.
        # We really would not want to do this if we were producing the list directly from prefilter.
        # But the usage pattern of prefiltering ahead of time and saving in a file seems like it will be the normal one, so it is OK.
        # (I do wish torch allowed for a forward-only dataset)
        self.windows = list(windows)  
        self.labeler = labeler
        self.c = c
        self.classes = classes
    
    def as_loader(self, bs=8, num_workers=0) -> torch.utils.data.DataLoader:
        """Return an appropriate DataLoader for this Dataset"""
        return torch.utils.data.DataLoader(self, batch_size=bs, num_workers=num_workers)

    def __len__(self):
        return len(self.windows)
    
    def __getitem__(self, index):
        (fp,w) = self.windows[index]
        return self.labeler(fp,w)

    def _set_item(self,index):
        # TODO: implement the extra bit that fastai needs.
        raise NotImplementedError