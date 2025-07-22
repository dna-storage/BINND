import os
from pathlib import Path

def find_project_root(marker_file_or_dir="setup.py"):
    """
    Finds the project root by searching for a marker file/directory
    up the directory tree from the current file's location.
    """
    current_path = Path(os.path.abspath(os.path.dirname(__file__)))
    for parent in current_path.parents:
        if (parent / marker_file_or_dir).exists():
            return parent
    raise FileNotFoundError(f"Project root marker '{marker_file_or_dir}' not found.")

try:
    ROOT_DIR = find_project_root() # Use your chosen marker
except FileNotFoundError:
    # Fallback for environments where __file__ might not be available or marker is missing
    # This is common in interactive shells or some notebook setups
    print("Warning: Project root marker not found. Falling back to os.getcwd().")
    ROOT_DIR = Path(os.getcwd()) # Fallback to current working directory

DATA_DIR = ROOT_DIR / 'data'
# Add other common paths here
# MODELS_DIR = ROOT_DIR / 'models'
# CONFIGS_DIR = ROOT_DIR / 'configs'

# Example usage (for testing this file directly)
if __name__ == "__main__":
    print(f"Project Root: {ROOT_DIR}")
    print(f"Data Directory: {DATA_DIR}")
    # print(f"Models Directory: {MODELS_DIR}")