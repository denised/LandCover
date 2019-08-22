# LandCover
Exploration of Machine Learning applied to Landsat data

My hypothesis is that there is enough interesting and unique about multispectral satellite data to make it worthwhile to
develop a transfer model or models for them, similar to imagenet for traditional image data.  To do an initial test of that hypothesis, 
I'm trying to develop an ML model that can predict land-use categories from landsat5 data, trained on the CORINE dataset.  I'm in the
process of trying to develop 'native' satellite models, and compare them to what you would get with using a standard imagenet
trained model on just the RGB portion of the landsat data--looking at both the final accuracy and the learning rate.  

Assuming the first phase is successful, a second phase will be to use the model used in phase 1 as a transfer model in a new ML 
problem and see how well it adapts.

The code in this repo has several parts:
* landsat5fetch: Code to script download of landsat tiles from the USGS directly.
* corine: (Some of the) code used to modify the corine dataset so that it is comparable to the landsat data
* multispectral: tools for managing multispectral data, and in particular for managing very large data multispectral 
data sets (e.g. landsat tiles)

To manage the large data sizes, I'm using a windowing technique to extract smaller 256x256 windows of data from the landsat tiles.
The inputs and targets for the ML process are matching windows from the landsat and corine data, respectively.
Currently this is done in a very efficient way, taking advantage of tiled tif storage, and maintaining order of windows in the dataset.
But it may also be interesting to try different window sizes, locations, orientations, etc.

Some other interesting / unusual bits:

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
TODO: add a 'get started' notebook<br>
TODO: create public store for our version of the corine dataset, and other artifacts.

# Using this Code - Setup

There are several environment variables that must be set for this code to function properly.  The code snippet below
specifies them in bash format:

````bash
# The location of the corine data (in a format that can be opened by pathlib.Path):
export CORINE_DIR=location_of_corine_data  # TODO: tell people how to download/reference this!

# If you want to use the landsat fetch code, you need to create an account with USGS (which you can obtain via the registration
# button at https://earthexplorer.usgs.gov).  The code will look for your username and password here:
export USGS_USER=your_user_name
export USGS_PASSWORD=your_password

# If you want to use neptune.ml for experiment monitoring and reporting, you will need an account with them
# and a API token, set here
export NEPTUNE_API_TOKEN=your_long_api_key
````

This code also builds on the fastai `default` feature.  The default values used and defined in this code can be found
in the file `infra.py`.  (See examples in the top-level notebooks of overriding these values.)

# Using this Code - Jupyter Notebooks

_TODO_