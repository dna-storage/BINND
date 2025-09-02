# 🧬 BINND: Binding and Interaction Neural Network for DNA

This repository contains the official implementation of **BINND (Binding and Interaction Neural Network for DNA)**, a novel framework designed to predict interactions between DNA molecules with high accuracy and speed.

## Table of Contents

-   [Project Overview](#project-overview-✨)
-   [Setup and Installation](#setup-and-installation-🛠️)
    -   [Prerequisites](#prerequisites)
    -   [Step-by-Step Setup Guide](#step-by-step-setup-guide)
        -   [1. Clone the Repository](#1-clone-the-repository)
        -   [2. Initialize the Project Environment](#2-initialize-the-project-environment)
        -   [3. Activate Your New Environment](#3-activate-your-new-environment)
-   [Using BINND for Prediction](#using-binnd-for-prediction-🚀)
-   [Expanding BINND](#expanding-binnd)


## Project Overview ✨

The ability to predict the interactions between DNA molecules is broadly important. It underlies core research methods in molecular and cell biology, drives the accuracy of disease detection and diagnostic tools, supports high throughput bioengineering, and could unlock next generation technologies like DNA-based data storage and computation. Here, we present BINND - Binding and Interaction Neural Network for DNA. BINND comprises an ultra-high throughput wet lab platform that measures millions of DNA-DNA interactions and a deep learning model that attains accuracies greater than 80% and generalizability across a diverse sequence space and varied interaction environments. Our framework leverages a Convolutional Neural Network (CNN) architecture to learn complex binding patterns directly from raw DNA sequences. In addition to its leading accuracy, BINND achieves speeds 2 orders of magnitude greater than current state-of-the-art models.

## Setup and Installation 🛠️

To get started with this project, please follow these steps. This will set up the necessary software environment and download any required components.

### Prerequisites

Before you begin, ensure you have the following installed on your system:

- **Git:** For cloning this repository.
- **Conda (or Miniconda/Anaconda):** A package and environment manager for Python. If you don't have it, you can download Miniconda from [here](https://docs.conda.io/en/latest/miniconda.html).
- **Make:** A build automation tool, typically pre-installed on Linux and macOS. For Windows, you might need to install it (e.g., via Chocolatey or Git Bash which often includes it).

### Step-by-Step Setup Guide 👇

Follow these steps in your terminal or command prompt:

### 1. Clone the Repository

First, get a copy of this project onto your computer:

```bash
git clone https://github.com/dna-storage/BINND.git
cd BINND
```

### 2. Initialize the Project Environment (First Time Only!)

**You only need to run this command once to set up your environment.** It will automatically create your Conda environment, install all necessary Python packages (including the specific PyTorch version), and set up any custom environment variables needed during the installation process. It also installs the current project in "editable mode" (`pip install -e .`).

```bash
make init
```

- **What to expect:** This command will print various messages as it downloads and installs packages. This process might take several minutes, depending on your internet connection and computer speed.
- **Important:** If you see any `ERROR` messages, please read them carefully. ⚠️ Common issues include Conda not being installed, or `BINND.yml` or `requirements.txt` files being missing.
- **💡 When to re-run `make init`:** You generally do not need to run make init again once it has completed successfully. However, you might consider re-running it if:

    - The `BINND.yml` or `requirements.txt` files are updated in the repository (e.g., to add new dependencies).

    - You make significant changes to the `setup.py` file or the `src/` directory (related to the `pip install -e .` part of the setup). In these cases, it's often sufficient to just reactivate your environment and then run `pip install -e .` directly within the active BINND environment, but `make init` will handle all updates.

### 3. Activate Your New Environment (Every Session!)

After `make init` finishes, the environment is created, but it's not automatically active in your current terminal session. You need to activate it manually.

**You must run this command in your terminal every time you open a new terminal session to work on this project:**

```bash
conda activate BINND
```

- **`conda activate BINND`**: This command switches your terminal to use the Python and packages from the `BINND` Conda environment. You'll usually see `(BINND)` appear at the beginning of your terminal prompt when it's active.

<!-- ### 4. Load Custom Environment Variables

Your project requires specific environment variables (like `PYTHONPATH`) to be set for the code to find its modules correctly.

**Run this command *after* activating your Conda environment, and every time you open a new terminal session to work on this project:**

```bash
. set_env.sh
```

- **`. set_env.sh`**: This command loads the necessary environment variables for your project. The `.` (dot) at the beginning is important! -->

## Using BINND for Prediction 🚀

Now that your environment is fully set up and activated, you're ready to leverage BINND's predictive power. The `inference_demo` directory provides an example of how to use the trained model to predict DNA-DNA interactions.

This step demonstrates the core functionality of BINND: taking DNA sequences as input and predicting their binding behavior with high accuracy.

Navigate to the `inference_demo` directory and run the sample inference script:

```bash
cd inference_demo
python sample_inference.py
```

- **What to expect:** Upon execution, the model will process the sample DNA sequences and output its predictions directly to your terminal. 

## Expanding BINND

While BINND comes with pre-trained models for immediate use, our framework is designed to be fully extensible. The BINND framework is designed with extensive modularity, allowing researchers and developers to easily modify and experiment with different components of the training and evaluation pipelines. For those interested in customizing BINND, you have the flexibility to train a new model from scratch and rigorously evaluate its performance.

This capability allows you to:
- Explore new deep learning architectures and encoding schemes
- Measure time and memory consumption of different stages of the deep learning pipeline
- Benchmark BINND's performance on your own data.
- Perform deeper analyses

Comprehensive, step-by-step instructions on how to prepare your data, configure training parameters, evaluate your custom BINND model and perform deeper analyses are available in our dedicated 📚 [docs](docs) and [examples](examples) directories.

**Our Recommendation 🌟** 
To kick things off and get the most out of this repository, we highly recommend starting with the `BINND_demo.ipynb` notebook located in the `examples` directory. This demo notebook provides a streamlined walkthrough of the core functionalities and will give you a great foundation before diving into more advanced topics or customizing your own workflows. Happy exploring!