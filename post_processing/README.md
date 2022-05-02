# Post-processing

Code for the project on wheel rut detection, funded by the Norwegian Environmental Agency

## Content
This directory contains both the developed Python script post-processing named `post_process`
and a Unix Shell script (`post_processing.sh`) that documents the application as an example.

In order to be able to run the post-processing script it needs to be executable and
available on the system PATH. To achieve that run the following code assuming that
the current working directory is the root of this repository.

    # make sure Python script is executable
    chmod ugo+x ${pwd}/post_processing/post_process
    # Add executable Python script to PATH
    export PATH=${pwd}/post_processing:$PATH

    
Then (with the required input data and dependencies in place) post-processing could be run like this:

    bash ${pwd}/post_processing/post_processing.sh

or interactively:

    bash ${pwd}/post_processing/post_process

