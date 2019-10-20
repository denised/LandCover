In order to use the code in LandCover, you will need to set a few environment variables.
Get the appropriate values for the following and add this text to your .bashrc file


# Locations of data files and corine files
export LANDSAT_DIR=landsat_directory
export CORINE_DIR=corine_directory

# If you want to download data from USGS using the code in landsatfetch, you need to create
# an account with USGS and set these
export USGS_USER=your_user_name
export USGS_PASSWORD=your_password

# If you want to use neptune.ml for experiment recording, you will need an account with them
# and a API token
export NEPTUNE_API_TOKEN=your_long_api_key

# To use TrainTracker with the webhook backend, you need to have a webhook URI (see traintracker.py for details)
export TRAINTRACKER_URI=your_webhook_uri

