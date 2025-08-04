import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from dataloader.dataset import HTPDataset
from training.train import train_model
from training.test import test_model
from visualize.plot_training import plot_logs
from training import evaluation_metrics
from utils.stats import get_max_memory_usage_mb
import torch.nn as nn
import os
import pandas as pd
import networks
import logging
import time
import json
from torchsummary import summary
import argparse

def main(data_directory, network_name, encode_fn_name, max_seq_length, batch_size, learning_rate, num_epochs, patience, checkpoint_dir, log_dir, is_train, is_test, log_interval):
    
    #print function parameters
    print(f"Data Directory: {data_directory}")
    print(f"Network Name: {network_name}")
    print(f"Encode Function Name: {encode_fn_name}")
    print(f"Max Sequence Length: {max_seq_length}")
    print(f"Batch Size: {batch_size}")
    print(f"Checkpoint Directory: {checkpoint_dir}")
    print(f"Log Directory: {log_dir}")
    print(f"Is Train: {is_train}")
    print(f"Is Test: {is_test}")
    
    if is_train:
        print(f"Learning Rate: {learning_rate}")
        print(f"Number of Epochs: {num_epochs}")
        print(f"Patience: {patience}")
        print(f"Log Interval: {log_interval}")

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

     # Model initialization
    if network_name == "CNNBinaryClassifierV3":
        model = networks.CNNBinaryClassifierV3().to(device)
    elif network_name == "CNNBinaryClassifier2V1":
        model = networks.CNNBinaryClassifier2V1().to(device)
    elif network_name == "BINND":
        model = networks.BINND().to(device)
    elif network_name == "BINNDLite":
        model = networks.BINNDLite().to(device)
    else:
        raise ValueError("Invalid network name")
    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params}")
    print(summary(model, input_size=(1, 4, 40)))
    
    runtime_info = {}

    if is_train:
        
        start_time = time.time()
        # Initialize dataset and dataloaders
        train_dataset = HTPDataset(data_path=os.path.join(
            data_directory, 'train.csv'), max_seq_length=max_seq_length, encode_fn_name=encode_fn_name)
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

        val_dataset = HTPDataset(data_path=os.path.join(
            data_directory, 'val.csv'), max_seq_length=max_seq_length, encode_fn_name=encode_fn_name)
        val_dataloader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

        end_time = time.time()
        training_data_loading_time = end_time - start_time
        runtime_info["training_data_loading_time"] = training_data_loading_time
       
        # Loss function and optimizer
        criterion = nn.BCELoss()  # Binary Cross-Entropy loss
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        # Train the model
        start_time = time.time()
        train_model(model, train_loader, val_dataloader, criterion, optimizer, device, checkpoint_dir=checkpoint_dir,
                    log_dir=log_dir, num_epochs=num_epochs, patience=patience, log_interval=log_interval)
        end_time = time.time()
        training_time = end_time - start_time
        runtime_info["training_time"] = training_time
        runtime_info["total_training_time"] = training_data_loading_time + training_time

        plot_logs(f"{log_dir}/train_log.csv",
                  f"{log_dir}/val_log.csv", log_interval, log_dir)

    if is_test:
        # Initialize dataset and dataloaders
        start_time = time.time()
        test_dataset = HTPDataset(data_path=os.path.join(
            data_directory, 'test.csv'), max_seq_length=max_seq_length, encode_fn_name=encode_fn_name)
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
        end_time = time.time()
        test_data_loading_time = end_time - start_time
        runtime_info["test_data_loading_time"] = test_data_loading_time

        # Load the model
        start_time = time.time()
        model.load_state_dict(torch.load(f"{checkpoint_dir}/best_model.pt", map_location=device))
        end_time = time.time()
        model_loading_time = end_time - start_time
        runtime_info["model_loading_time"] = model_loading_time

        start_time = time.time()
        test_model(model, test_loader, device, log_dir)
        end_time = time.time()
        test_time = end_time - start_time
        runtime_info["test_time"] = test_time
        runtime_info["total_test_time"] = test_data_loading_time + model_loading_time + test_time

        # Model evaluation
        test_df = pd.read_csv(f'{log_dir}/test_log.csv')
        evaluation_metrics.compute_eval_metrics(
            test_df['Label'], test_df['Probability'], save_path=f'{log_dir}/test_metrics.json')
        
        evaluation_metrics.plot_roc_curve(test_df['Label'], test_df['Probability'], save_path=f'{log_dir}/roc_curve.png')

        evaluation_metrics.plot_confusion_matrix(f"{log_dir}/test_metrics.json", f"{log_dir}/confusion_matrix.png")

    # get max memory usage
    max_memory_usage = get_max_memory_usage_mb()
    runtime_info["max_memory_usage(MB)"] = max_memory_usage

    # Save runtime information
    runtime_file_path = f"{log_dir}/runtime_info.json"
    if os.path.exists(runtime_file_path):
        with open(runtime_file_path, "r") as f:
            try:
                data = json.load(f)
                if not isinstance(data, list):
                    data = [data]
            except json.JSONDecodeError:
                data = []
    else:
        data = []
    data.append(runtime_info)
    with open(runtime_file_path, "w") as f:
        json.dump(data, f, indent=4)
    print("Runtime information saved to runtime_info.json")

def parse_args():
    parser = argparse.ArgumentParser(description="Run training or testing for a sequence model.")

    parser.add_argument("--data_dir", type=str, required=True, help="Path to the dataset directory")
    parser.add_argument("--network_name", type=str, required=True, help="Name of the neural network architecture to use")
    parser.add_argument("--encoder_name", type=str, required=True, help="Name of the encoder to use")
    parser.add_argument("--max_seq_length", type=int, default=20, help="Maximum sequence length")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training/testing")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate for optimizer")
    parser.add_argument("--num_epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--patience", type=int, default=2, help="Early stopping patience")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Directory to save model checkpoints")
    parser.add_argument("--log_dir", type=str, default="logs", help="Directory to save training logs")
    parser.add_argument("--is_train", action="store_true", help="Flag to run training")
    parser.add_argument("--is_test", action="store_true", help="Flag to run testing")
    parser.add_argument("--log_interval", type=int, default=500, help="Interval (in steps) for logging during training")

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    main(
        args.data_dir,
        args.network_name,
        args.encoder_name,
        args.max_seq_length,
        args.batch_size,
        args.learning_rate,
        args.num_epochs,
        args.patience,
        args.checkpoint_dir,
        args.log_dir,
        args.is_train,
        args.is_test,
        args.log_interval
    )
 