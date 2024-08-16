#!/bin/bash

# Check if the environment name is provided
if [ -z "$1" ]; then
    echo "Usage: $0 <environment_name>"
    exit 1
fi

ENV_NAME=$1

# Activate the Conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate "$ENV_NAME"

# Check if activation was successful
if [ $? -eq 0 ]; then
    echo "Activated Conda environment: $ENV_NAME"
else
    echo "Failed to activate Conda environment: $ENV_NAME"
    exit 1
fi
