# BINND: Binding and Interaction Neural Network for DNA

This repository contains the official implementation of **BINND (Binding and Interaction Neural Network for DNA)**, a novel framework designed to predict interactions between DNA molecules with high accuracy and speed.

## Table of Contents

-   [Project Overview](#project-overview)
-   [Setup and Installation](#setup-and-installation)
    -   [Prerequisites](#prerequisites)
    -   [Step-by-Step Setup Guide](#step-by-step-setup-guide)

## Project Overview

The ability to predict the interactions between DNA molecules is broadly important. It underlies core research methods in molecular and cell biology, drives the accuracy of disease detection and diagnostic tools, supports high throughput bioengineering, and could unlock next generation technologies like DNA-based data storage and computation. Here, we present BINND - Binding and Interaction Neural Network for DNA. BINND comprises an ultra-high throughput wet lab platform that measures millions of DNA-DNA interactions and a deep learning model that attains accuracies greater than 80% and generalizability across a diverse sequence space and varied interaction environments. Our framework leverages a Convolutional Neural Network (CNN) architecture to learn complex binding patterns directly from raw DNA sequences. In addition to its leading accuracy, BINND achieves speeds 2 orders of magnitude greater than current state-of-the-art models.

## Setup and Installation

To get started with this project, please follow these steps. This will set up the necessary software environment and download any required components.

### Prerequisites

Before you begin, ensure you have the following installed on your system:

- **Git:** For cloning this repository.
- **Conda (or Miniconda/Anaconda):** A package and environment manager for Python. If you don't have it, you can download Miniconda from [here](https://docs.conda.io/en/latest/miniconda.html).
- **Make:** A build automation tool, typically pre-installed on Linux and macOS. For Windows, you might need to install it (e.g., via Chocolatey or Git Bash which often includes it).

### Step-by-Step Setup Guide

Follow these steps in your terminal or command prompt:

### 1. Clone the Repository

First, get a copy of this project onto your computer:

```bash
git clone https://github.ncsu.edu/dna-based-storage/BINND.git
cd BINND
```

### 2. Initialize the Project Environment

This command will automatically create your Conda environment, install all necessary Python packages (including the specific PyTorch version), and set up any custom environment variables needed during the installation process.

```bash
make init
```

- **What to expect:** This command will print various messages as it downloads and installs packages. This process might take several minutes, depending on your internet connection and computer speed.
- **Important:** If you see any `ERROR` messages, please read them carefully. Common issues include Conda not being installed, or `BINND.yml` or `requirements.txt` files being missing.

### 3. Activate Your New Environment

After `make init` finishes, the environment is created, but it's not automatically active in your current terminal session. You need to activate it manually.

**You must run this command in your terminal *after* `make init` completes, and every time you open a new terminal session to work on this project:**

```bash
conda activate BINND
```

- **`conda activate BINND`**: This command switches your terminal to use the Python and packages from the `BINND` Conda environment. You'll usually see `(BINND)` appear at the beginning of your terminal prompt when it's active.

### 4. Load Custom Environment Variables

Your project requires specific environment variables (like `PYTHONPATH`) to be set for the code to find its modules correctly.

**Run this command *after* activating your Conda environment, and every time you open a new terminal session to work on this project:**

```bash
. set_env.sh
```

- **`. set_env.sh`**: This command loads the necessary environment variables for your project. The `.` (dot) at the beginning is important!

### 5. Run the Sample Inference

Once your environment is fully set up and activated, you can run the demo inference script:

```bash
cd inference_demo
python sample_inference.py
```

- **What to expect:** This will run the model on sample data and print the prediction results to your terminal.