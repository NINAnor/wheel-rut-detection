# Stuff from Nibio

## 1_training_preprosess_bash

This dir is for a bash script (or rather a list of commands) that can
burn the ground truth (a shape file) to geotiff.

To use it, change 'your_*' to your own stuff.

The ground truth is in dir ground_truth_gpkg.

## 2_training_preprocess_python

This dir has python scripts (or rather a list of commands) that can
tile and split an image and a ground truth.

To use it, chnange 'your_dir' to your own stuff.

## 3_training_python_modules

Here, the track dataset is (track_zero_to_one_dataset.py).

Also, there are copies from torchvision - references/segmentation (in
from_torchvision_reference the). In
from_torchvision_reference_and_modified, the train_modified.py is a
modified version of train.py.

The modules should be copied to working dir.

## 4_train

Here, python scripts (or rather a list of commands) for training can
be found.

To use it, chnange 'your_dir' and 'your_data_dir' to your own stuff.

## 5_predict_preprocess_python

The scripts here tile the whole imput images.

To use it, change 'your_*' to your own stuff.

Note that the Rjukan drone image had to be grouped in two because of
the merging in 6_predict.

## 6_predict

predict.py is a python script to predict. It should be copied to
working dir, together with the training modules.

predict_tracks_atv_aerial_rgb_all.sh and
predict_tracks_atv_drone_rgb_all.sh are scripts (commands) to predict.
To use it, change 'your_*' to your own stuff.

