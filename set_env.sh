#!/bin/bash

# This script sets up project-specific environment variables.
# It should be sourced, not executed directly (e.g., '. ./set_env.sh').

# Get the absolute path to the directory containing this script.
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Define the paths relative to the script's directory that should be added to PYTHONPATH.
# Add your project root and your 'src' directory.
PROJECT_ROOT_PATH="$SCRIPT_DIR"
SRC_PATH="$SCRIPT_DIR/src"

# --- Add PROJECT_ROOT_PATH to PYTHONPATH if not already present ---
# Check if PYTHONPATH is already set and if PROJECT_ROOT_PATH is already in it.
if [[ -z "$PYTHONPATH" || ! ":$PYTHONPATH:" =~ ":$PROJECT_ROOT_PATH:" ]]; then
    # Prepend the path to ensure it's searched first.
    export PYTHONPATH="$PROJECT_ROOT_PATH${PYTHONPATH:+:$PYTHONPATH}"
    echo "Added $PROJECT_ROOT_PATH to PYTHONPATH."
else
    echo "$PROJECT_ROOT_PATH already in PYTHONPATH. Skipping."
fi

# --- Add SRC_PATH to PYTHONPATH if not already present ---
if [[ -z "$PYTHONPATH" || ! ":$PYTHONPATH:" =~ ":$SRC_PATH:" ]]; then
    # Prepend the path to ensure it's searched first.
    export PYTHONPATH="$SRC_PATH${PYTHONPATH:+:$PYTHONPATH}"
    echo "Added $SRC_PATH to PYTHONPATH."
else
    echo "$SRC_PATH already in PYTHONPATH. Skipping."
fi

# You can add other environment variables here if needed
# export MY_CUSTOM_VAR="some_value"

# Optional: Print the current PYTHONPATH for verification
# echo "Current PYTHONPATH: $PYTHONPATH"