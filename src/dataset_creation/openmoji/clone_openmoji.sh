#!/bin/bash

cd ../../../data

# Clone OpenMoji repo without checking out files
git clone --no-checkout https://github.com/hfg-gmuend/openmoji.git openmoji-lite

# Go inside the folder
cd openmoji-lite || exit

# Initialize sparse checkout
git sparse-checkout init --cone

# Select only the folders we're interested in
git sparse-checkout set color data src

# Checkout files
git checkout

echo "✅ Done! Only 'color', 'data', and 'src' folders were cloned into openmoji-lite/"

cd ../src/dataset_creation/openmoji
