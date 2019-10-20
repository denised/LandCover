#!/usr/bin/env bash

set -e
set -o xtrace
DEBIAN_FRONTEND=noninteractive

# Install miniconda.  See https://docs.conda.io/en/latest/miniconda.html to verify url
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash ./Miniconda3-latest-Linux-x86_64.sh -b

# create conda environment
conda env create -f environment.yml

# set up jupyter for a headless environment.  Omit all the rest of this script if you are using
# this code locally.
jupyter notebook password   # prompts for password

# create a self-signed certificate
# This leaves the files in the current directory.  Change the filenames here and
# in hte jupyter config below if you'd rather put them someplace else.
openssl req -x509 -nodes -days 365 -newkey rsa:2048 -keyout mykey.key -out mycert.pem

# Configure jupyter to operate over https, on port 8888.
jupyter notebook --generate-config
echo "c.NotebookApp.ip = '*'" >> ~/.jupyter/jupyter_notebook_config.py
echo "c.NotebookApp.open_browser = False" >> ~/.jupyter/jupyter_notebook_config.py
echo "c.NotebookApp.certfile = u'$(cwd)/mycert.pem'" >> ~/.jupyter/jupyter_notebook_config.py
echo "c.NotebookApp.keyfile = u'$(cwd)/mykey.key'" >> ~/.jupyter/jupyter_notebook_config.py
echo "c.NotebookApp.port = 8888" >> ~/.jupyter/jupyter_notebook_config.py

# Make sure the 8888 port is open.
sudo ufw allow 8888

