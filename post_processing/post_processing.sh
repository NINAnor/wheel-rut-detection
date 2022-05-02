#!/bin/bash

# The following code documents how the post-processing script is applied to
# predictions produced by NIBIO

# It assumes that input data is organized in the "project_directory"
# and that the Python script post_processing is executable and on the
# sytem PATH

# The code has been run on a server with 48 cores and 384 GB memory
# the combinations both study sites and input imagery type (drone, aerial)
# are processed in parallel and each process is allocated 5 cores for
# tiled processing

# Roads and streams represent respective FKB data

# Input and output data is located in the project directory
project_directory=/data/P-Prosjekter/15215600_autmatisert_kartlegging_av_barmarksskader

# The temporary directory is preferably located on a high performance storage
# medium as most of the processing happens here
tempdir=/data/scratch

mkdir ${project_directory}/post_processing

# Loop over study sites
for s in balsfjord rjukan
do
  # Define input variables
  if [ "$s" == "rjukan" ] ; then
    site=Rjukan
    # Acquisition day
    d=1
    # Input ancillary data
    roads=Rjukan_roads_32.shp
    streams=Rjukan_streams_32.shp
  else
    site=Skutviksvatnet
    # Acquisition day
    d=2
    # Input ancillary data
    roads=Balsfjord_roads_34.shp
    streams=Balsfjord_streams_34.shp
  fi
  # Loop over predictions
  for p in aerial_20220222 drone_20220221
  do
    post_process -nprocs 5 -verbose -output ${project_directory}/post_processing -workdir ${tempdir} -prefix ${s}_${p} -dtm ${project_directory}/ODM_results/P4M_${site}_Day${d}/P4M_${site}_Day${d}_dsm.tif -ndvi ${project_directory}/ODM_results/P4M_${site}_Day1/P4M_${site}_Day1_orthophoto.tif -input ${project_directory}/NIBIO/predictions_${s}_${p}.tif -roads=${project_directory}/FKB/$roads -streams=${project_directory}/FKB/$streams -resolution 0.15 -tiling 1,5 &
  done
done

