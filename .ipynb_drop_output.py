#!/usr/bin/env python

"""
Suppress output and prompt numbers in git version control.

This script tells git how to treat ipynb files.  With this script in stalled,
By default, git will keep only the cell inputs, ignoring prompt numbers and cell output.
However, if the noteook metadata contains

    "git" : { "keep_outputs" : true }

then the entire notebook it saved.

The notebooks themselves are never changed.

Derived from https://gist.github.com/pbugnion/ea2797393033b54674af,
but I switched the default behavior to stripping output.

Usage instructions
==================

There are various ways a git filter can be installed: the following instructions are for using
this filter locally in this repository only (which has the advantage that the behavior is copied
with the repo when it is cloned):

1. Put this script in the root directory of your git project.  
2. Make sure it is executable by typing the command
   `chmod +x <script file>`
3. Register a filter for ipython notebooks by putting the following line in the file .gitattributes 
   at the root directory of your git project:
   `*.ipynb  filter=clean_ipynb`
4. Connect this script to the filter by running the following 
   git commands from the root directory of your repository:

   git config filter.clean_ipynb.clean ./<script file>
   git config filter.clean_ipynb.smudge cat

To tell git to _keep_ the output for a notebook,
open the notebook's metadata (Edit > Edit Notebook Metadata). A
panel should open containing the lines:

    {
        "name" : "",
        "signature" : "some very long hash"
    }

Add an extra line so that the metadata now looks like:

    {
        "name" : "",
        "signature" : "don't change the hash, but add a comma at the end of the line",
        "git" : { "keep_outputs" : true }
    }

You may need to "touch" the notebooks for git to actually register a change, if
your notebooks are already under version control.
"""

import sys
import json

nb = sys.stdin.read()
json_in = json.loads(nb)

nb_metadata = json_in["metadata"]
if hasattr(nb_metadata,"git") and getattr(nb_metadata["get"],"keep_outputs",False):
    # don't strip, just return the entire content
    sys.stdout.write(nb)
    exit() 

#else strip outputs

ipy_version = int(json_in["nbformat"])-1 # nbformat is 1 more than actual version.

def strip_output_from_cell(c):
    if "outputs" in c:
        c["outputs"] = []
    if "prompt_number" in cell:
        del c["prompt_number"]

if ipy_version == 2:
    for sheet in json_in["worksheets"]:
        for cell in sheet["cells"]:
            strip_output_from_cell(cell)
else:
    for cell in json_in["cells"]:
        strip_output_from_cell(cell)

json.dump(json_in, sys.stdout, sort_keys=True, indent=1, separators=(",",": "))