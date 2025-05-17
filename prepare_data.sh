#!/bin/bash
# This script runs during repo2docker build to prepare the dataset
set -e  # Exit immediately if a command fails
set -x  # Print each command before executing it

echo "Starting prepare_data.sh script from $(pwd)"

# Create a dedicated directory for test data
mkdir -p demo_astrodata

# Print environment variables for debugging
echo "CI=$CI"
echo "GITHUB_ACTIONS=$GITHUB_ACTIONS"
echo "BINDER_SERVICE_HOST=$BINDER_SERVICE_HOST"

# If running in CI environment, prepare the data
if [ -n "$CI" ] || [ -n "$GITHUB_ACTIONS" ] || [ -n "$BINDER_SERVICE_HOST" ]; then
    echo "CI environment detected - downloading minimal dataset..."
    
    # Install requirements
    pip install requests tqdm
    
    # Download the smaller testing dataset (3.4GB)
    cd demo_astrodata
    
    echo "Downloading from Zenodo..."
    wget -q --show-progress https://zenodo.org/records/11117528/files/5x64x64_testing_with_morphology.hdf5
    
    # Create copies for train and validation (this is just to make the paths work)
    echo "Creating train/val datasets from test data..."
    cp 5x64x64_testing_with_morphology.hdf5 5x64x64_training_with_morphology.hdf5
    cp 5x64x64_testing_with_morphology.hdf5 5x64x64_validation_with_morphology.hdf5
    
    echo "Listing files in demo_astrodata:"
    ls -la
    
    echo "Dataset preparation complete!"
    cd ..
    
    # Create a marker file
    touch .ci_mode
    echo "Created .ci_mode marker file"
    
    # List files in current directory
    echo "Files in $(pwd):"
    ls -la
else
    echo "Local environment detected - using existing datasets"
fi