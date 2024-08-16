#!/bin/bash

# Check if the environment name is provided
if [ -z "$1" ]; then
    echo "Usage: $0 <environment_name>"
    exit 1
fi

ENV_NAME=$1
REQUIREMENTS_FILE="pip_requirements.txt"

# Check if the requirements.txt file exists in the current directory
if [ ! -f "$REQUIREMENTS_FILE" ]; then
    echo "Error: $REQUIREMENTS_FILE not found in the current directory."
    exit 1
fi

# Activate conda.sh script
source ~/miniconda3/etc/profile.d/conda.sh 

# Create the Conda environment
echo "Creating Conda environment: $ENV_NAME"
conda create --name "$ENV_NAME" python=3.8 -y

# Activate the newly created environment
echo "Activating Conda environment: $ENV_NAME"
conda activate "$ENV_NAME"

# Check if activation was successful
if [ $? -ne 0 ]; then
    echo "Failed to activate Conda environment: $ENV_NAME"
    exit 1
fi

# Install the packages using pip from requirements.txt
echo "Installing packages from $REQUIREMENTS_FILE"
pip install -r "$REQUIREMENTS_FILE"

# Confirm success
if [ $? -eq 0 ]; then
    echo "Environment $ENV_NAME successfully created and packages installed."
else
    echo "There was an error installing packages from $REQUIREMENTS_FILE."
    exit 1
fi
