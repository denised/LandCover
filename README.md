# LandCover
Exploration of Machine Learning applied to Landsat data

My hypothesis is that there is enough interesting and unique about multispectral satellite data to make it worthwhile to
develop a transfer model or models for them, similar to imagenet for traditional image data.  To do an initial test of that hypothesis, 
I'm trying to develop an ML model that can predict land-use categories from landsat5 data, trained on the CORINE dataset.  I'm in the
process of trying to develop 'native' satellite models, and compare them to what you would get with using a standard imagenet
trained model on just the RGB portion of the landsat data--looking at both the final accuracy and the learning rate.  

Assuming the first phase is successful, a second phase will be to use the model used in phase 1 as a transfer model in a new ML 
problem and see how well it adapts.

To manage the large data sizes, I'm using a windowing technique to extract smaller 256x256 windows of data from the landsat tiles.
The inputs and targets for the ML process are matching windows from the landsat and corine data, respectively.
Currently this is done in a very efficient way, taking advantage of tiled tif storage, and maintaining order of windows in the dataset.
But it may also be interesting to try different window sizes, locations, orientations, etc.

Some interesting / unusual bits:

* The data we have may have non-overlapping extents.  That is we have landsat data that doesn't correspond to any region covered by
corine, and vice versa.  So when actually generating the x,y pairs, we intersect them and propagate a 'NODATA' value both ways.
* We using a one-hot encoding of land use class, but not quite: each pixel may have multiple land-use classes with positive values.
(So I guess you could call it a 'multi-hot' encoding.)  These can be thought of as the probability or proportion of that land use 
within that pixel (aka 30m^2 region).  I started doing this because map reprojections necessarily "blur" at the edges, and it seemed
less wrong to make the blurry edges have multiple classes than to arbitrarily choose one, but it remains to be seen if this is a good
approach or not.
* Landsat provides 'cloud' and other quality assessment.  We propagate that information to the target as well, so the learner
will see something like: "there is both ocean and cloud here", and hopefully learn to recognize cloud, while still having the
advantage of seeing the extent-behavior of ocean.  (This also uses the multi-hot encoding.)

Technology: fastai (https://fast.ai), pytorch and GDAL/rasterio

TODO: add references to the landsat and corine data<br>
TODO: create public store for our version of the corine dataset, and other artifacts.

# What is in this Repo

The code in this repo is organized into several parts:
* landsat5fetch: Code to script download of landsat tiles from the USGS directly.
* corine: (Some of the) code used to modify the corine dataset so that it is comparable to the landsat data
* multispectral: tools for managing multispectral data, and in particular for managing very large data multispectral 
data sets (e.g. landsat tiles)
* infra (short for infrastructure):  Additions to fastai that I have found to make life easier.  Some of these are pretty
significant and may be of interest on their own:
    * CycleHandler: a fastai Callback that makes it possible to run callbacks on a fixed cycle of less than an epoch.
    It is intended to be used primarily to get more frequent metrics reporting.
    * TrainTracker: yet another way to keep track of experiments you have run.  The way I have it set up, it logs runs
    to a Google spreadhsheet, which is a particularly light-weight and simple.  Of particular interest: it watermarks
    models with an ID, so that you can keep track of the _sequence_ of training events that produced a particular model.


# Using this Code
## Setup

The setup directory contains shell scripts that I use to set up an Ubuntu 16.04 VM.  I don't claim that these will work for everyone, but
at least it covers all the steps.  The environment.yml file, in particular, specifies the complete conda environment.

There are some environment variables that I use to specify some configuration: these are described in the setup/Readme.txt file.

## Getting the Data

_TODO_

## Jupyter Notebooks

Start with the notebook StartHere.ipynb :-)
The file ./runner.py is a python script that also has all the setup required to run learn.fit with this code.  I use it by making a copy and
customizing the setup and calls to what I want to do in my next experiment.

In the landsatfetch directory is another notebook that walks you through the steps to download data from USGS and process it into the right format.