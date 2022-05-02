# Post-processing

Code for the project on wheel rut detection, funded by the Norwegian Environmental Agency

## Requirements

In order to be able to run the post-processing 
routines, the folowing dependencies need to be installed on the machine (tested on Ubuntu 18.04):

* GRASS GIS 8.0
* grass_session python library

A virtual Ubuntu machine can be setup with the following 
commands:

    sudo add-apt-repository ppa:ubuntugis/ubuntugis-unstable
    sudo apt-get update
    sudo apt-get install grass-dev
    sudo python3 -m pip install grass_session numpy

## Content
This directory contains both the developed Python script post-processing named `post_process`
and a Unix Shell script (`post_processing.sh`) that documents the application as an example.

In order to be able to run the post-processing script it needs to be executable and
available on the system `PATH`. To achieve that run the following code assuming that
the current working directory is the root of this repository.

    # make sure Python script is executable
    chmod ugo+x ${pwd}/post_processing/post_process
    # Add executable Python script to PATH
    export PATH=${pwd}/post_processing:$PATH

    
Then (with the required input data and dependencies in place) post-processing could be run like this:

    bash ${pwd}/post_processing/post_processing.sh

or interactively:

    bash ${pwd}/post_processing/post_process

Available options can be explored with:

    bash ${pwd}/post_processing/post_process --help
