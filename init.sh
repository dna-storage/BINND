#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

ENV_NAME="BINND" # Name of your Conda environment
YML_FILE="BINND.yml"
REQUIREMENTS_FILE="requirements.txt"
SET_ENV_SCRIPT="set_env.sh" # Name of your environment setup script

echo "--- Starting Environment Setup for $ENV_NAME ---"

# 1. Check if Conda is installed
echo "Checking for Conda installation..."
if ! command -v conda &> /dev/null
then
    echo "ERROR: Conda is not found. Please install Miniconda or Anaconda first."
    echo "Download from: https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi
echo "Conda is installed."

# 2. Create or Update Conda Environment
echo "--- Creating/Updating Conda environment: $ENV_NAME ---"
if [ ! -f "$YML_FILE" ]; then
    echo "ERROR: '$YML_FILE' not found. Please ensure it's in the same directory as init.sh."
    exit 1
fi

# Try to create the environment. If it already exists, update it.
# To makes the script safe to run multiple times.
conda env create -f "$YML_FILE" --name "$ENV_NAME" || \
conda env update -f "$YML_FILE" --name "$ENV_NAME"

echo "Conda environment '$ENV_NAME' created or updated."

# 3. Activate the Conda Environment for subsequent commands
echo "--- Activating Conda environment: $ENV_NAME ---"
eval "$(conda shell.bash hook)"
conda activate "$ENV_NAME"

echo "Environment activated."

# 4. Source the custom environment setup script
# echo "--- Sourcing custom environment variables from $SET_ENV_SCRIPT ---"
# if [ -f "$SET_ENV_SCRIPT" ]; then
#     . "./$SET_ENV_SCRIPT" # Use '.' or 'source' to run the script in the current shell context
#     echo "Custom environment variables loaded for this script's execution."
# else
#     echo "WARNING: '$SET_ENV_SCRIPT' not found. Skipping custom environment setup."
# fi

# 5. Install specific PyTorch version
echo "--- Installing specific PyTorch version via pip ---"
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu126

echo "PyTorch installed."

# 6. Install other pip packages from requirements.txt
echo "--- Installing other pip packages from $REQUIREMENTS_FILE ---"
if [ ! -f "$REQUIREMENTS_FILE" ]; then
    echo "WARNING: '$REQUIREMENTS_FILE' not found. Skipping additional pip package installation."
else
    pip install -r "$REQUIREMENTS_FILE"
    echo "Other pip packages installed."
fi

echo "--- Setup Complete! ---"
echo "Your environment '$ENV_NAME' is ready."
echo "To use it, you need to activate it in your terminal for future sessions:"
echo "   conda activate $ENV_NAME"
echo "   . $SET_ENV_SCRIPT"
echo "Then you can run your inference script:"
echo "   cd inference_demo"
echo "   python sample_inference.py"