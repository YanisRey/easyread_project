#!/bin/bash


# Specify directory
cd /work/courses/dslab/team4/easyread_project/data
# Clone OpenMoji repo without checking out files
git clone --no-checkout https://github.com/hfg-gmuend/openmoji.git openmoji

# Go inside the folder
cd openmoji || exit

# Initialize sparse checkout
git sparse-checkout init --cone

# Select only the folders we're interested in
git sparse-checkout set color data src

# Checkout files
git checkout

echo "✅ Done! Only 'color', 'data', and 'src' folders were cloned into openmoji/"

cd ../../src/dataset_creation/openmoji
