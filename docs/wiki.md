# BINND: Training and Testing Guide

This guide provides detailed instructions on how to train and test the **BINND (Binding and Interaction Neural Network for DNA)** model using the provided scripts. This covers setting up configuration, executing training runs, and performing evaluations.

<!-- ## Table of Contents

- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Configuration](https://www.google.com/search?q=%23configuration)
    - [Network Architectures](https://www.google.com/search?q=%23network-architectures)
    - [Encoder Functions](https://www.google.com/search?q=%23encoder-functions)
    - [Paths](https://www.google.com/search?q=%23paths)
    - [Run Parameters](https://www.google.com/search?q=%23run-parameters)
- [Running Training](https://www.google.com/search?q=%23running-training)
- [Running Testing (Inference)](https://www.google.com/search?q=%23running-testing-inference)
- [Understanding Outputs](https://www.google.com/search?q=%23understanding-outputs)
- [Troubleshooting](https://www.google.com/search?q=%23troubleshooting) -->

## Overview

The primary script for training and testing is `main.py`, which is orchestrated by a separate configuration script `run_experiment.py`  found in the sub-directory `scripts`. This guide will walk you through setting up the parameters for your experiments and executing them.

## Prerequisites

Before proceeding, ensure you have completed the [main project setup](https://github.ncsu.edu/dna-based-storage/BINND) (including cloning the repository, creating the Conda environment, and loading custom environment variables). Your Conda environment (`BINND`) must be activated, and `set_env.sh` must be sourced in your current terminal session.

```bash
conda activate BINND
. set_env.sh
```

## Configuration

The `run_experiment.py` script is where you define the parameters for your training or testing run. Open this file to customize your experiment.

### Network Architectures

The `BINND` framework supports several CNN architectures. You can select one by setting the `network_name` variable in `run_experiment.py`.

- **`VALID_NETWORK_NAMES`**:
    - `"BINND"`: The primary, robust BINND CNN architecture.
    - `"BINNDLite"`: A lightweight version of the BINND CNN.
    
    **Example in `run_experiment.py`:**
    
    ```bash
    network_name = "BINNDLite" # Choose your desired network
    ```
    

### Encoder Functions

The input DNA sequences are processed by specific encoder functions. Ensure the chosen `encoder_name` is compatible with your selected network architecture. 

- **`VALID_ENCODER_NAMES`**:
    - `4xn_v2`: The encoding scheme compatible with BINND's CNNs.
    
    **Example in `run_experiment.py`:**
    
    ```
    encoder_name = "4xn_v2" # Choose your desired encoder
    ```
    

### Paths

The script defines key directories for data, model checkpoints, and logs. These are relative to your `ROOT_DIR` (the project's main directory).

- **`data_dir`**: Location of your input dataset.
    - **Expected Structure:** This directory should contain the following CSV files:
        - `train.csv`: Your training dataset.
        - `val.csv`: Your validation dataset.
        - `test.csv`: Your testing dataset.
    - Each file should contain three columns: Seq1, Seq2, and Label. For example,   

        | Seq1               | Seq2               | Label |
        |--------------------|--------------------|-------|
        | GTCAGCTCTTAGCGTCACTC | AGATGAGTACTCATGGGCAT | 1 |
        |CTCCATTAGGGTAAGGCTCC|GCCGAGGGGCAGTGGGGGGG|0|
    

    - *Default:* `[ROOT_DIR]/data/demo`
- **`checkpoint_dir`**: Where trained model weights will be saved.
    - *Default:* `[ROOT_DIR]/experiments/checkpoints/demo`
- **`log_dir`**: Where training and validation logs (e.g., `train_log.csv`, `val_log.csv`) will be stored.
    - *Default:* `[ROOT_DIR]/experiments/logs/demo`
    
    **Example in `run_experiment.py`:**
    
    ```bash
    data_dir = os.path.join(ROOT_DIR, "data", "my_custom_dataset")
    ```
    

### Run Parameters

Adjust these parameters in `run_experiment.py` to control the training or testing process:

- **`is_train` (boolean)**: Set to `True` to run training; `False` for inference/testing.
- **`is_test` (boolean)**: Set to `True` to run testing/inference; `False` for only training.
    - **Note:** To train and test together, set `is_train` to `True` AND `is_test` to `True`.
- **`max_seq_length` (int)**: The fixed length of input DNA sequences the model expects. The current implementation only supports 20 nucleotide long sequences.
- **`batch_size` (int)**: Number of samples processed in one go.
- **`log_interval` (int)**: How frequently (in batches) to print intermediate training/validation progress and log to files.
- **`learning_rate` (float)**: (Only for `is_train=True`) The step size for the optimizer during training.
- **`num_epochs` (int)**: (Only for `is_train=True`) The total number of full passes over the training dataset.
- **`patience` (int)**: (Only for `is_train=True`) For early stopping; number of epochs to wait for validation loss improvement before stopping training.

## Running Training

To train a new model:

1. Open `run_experiment.py`.
2. Set `is_train = True` and `is_test = False`.
3. Adjust `network_name`, `encoder_name`, paths, and training parameters (`learning_rate`, `num_epochs`, `patience`, `batch_size`, `log_interval`) as desired.
4. Save the file.
5. Ensure your `BINND` Conda environment is activated and `set_env.sh` is sourced.
6. Run the script:
    
    ```bash
    python run_experiment.py
    ```
    
    - **What to expect:** The script will print training progress, including loss and accuracy at specified `log_interval`s. The best performing model (based on validation loss) will be saved to `checkpoint_dir`, and detailed logs will be written to `log_dir`.

## Running Testing (Inference)

To perform inference using a pre-trained model:

1. Open `run_experiment.py`.
2. Set `is_train = False` and `is_test = True`.
3. Ensure `checkpoint_dir` points to the location of your `best_model.pt` file.
4. Adjust `network_name`, `encoder_name`, `data_dir`, `batch_size`, and `log_interval` as needed for testing. (Note: `learning_rate`, `num_epochs`, `patience` are ignored during testing.)
5. Save the file.
6. Ensure your `BINND` Conda environment is activated and `set_env.sh` is sourced.
7. Run the script:
    
    ```bash
    python run_experiment.py
    ```
    
    - **What to expect:** The script will load the model from `checkpoint_dir` and run it on the data specified by `data_dir/test.csv`. It will print evaluation metrics (e.g., test loss, accuracy) and prediction results will be written to `log_dir`.

## Understanding Outputs

- **Console Output:** Provides real-time feedback during training/testing, including batch-wise and epoch-wise metrics.
- **`log_dir`:** This directory will contain detailed logs and plots generated during training. After a training run, you can expect to find the following files:
    - `runtime_info.json`: A JSON file containing information about the training run, such as execution time and memory consumption.
    - `train_log.csv`: A CSV file logging the training loss and accuracy per  `log_interval`.
    - `val_log.csv`: A CSV file logging the validation loss and accuracy per  `log_interval`.
    - `train_loss.png`: A plot visualizing the training loss over time.
    - `train_accuracy.png`: A plot visualizing the training accuracy over time.
    - `val_loss.png`: A plot visualizing the validation loss over time.
    - `val_accuracy.png`: A plot visualizing the validation accuracy over time.
- **`checkpoint_dir`:** Stores `best_model.pt`, which is the PyTorch state dictionary of the model that achieved the lowest validation loss during training.