#!/bin/bash

set -exuo pipefail

bash ./1_training_preprosess_bash/burn_fasit.sh
python3 ./2_training_preprocess_python/tile_and_split_train_val_test_interactive_aerial.py
python3 ./2_training_preprocess_python/tile_and_split_train_val_test_interactive_drones.py

