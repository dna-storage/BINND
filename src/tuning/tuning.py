import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import os
from ray import train, tune
from ray.tune.schedulers import ASHAScheduler
from networks.cnn import GeneralCNNBinaryClassifier
from src.dataloader.dataset import HTPDataset


def main(data_directory, tune_log_dir, max_num_epochs=6, patience=2, reduction_factor=2, num_trials=10):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    conv2d_config = [
        {
            'out_channels': tune.choice([16, 32, 64, 128, 256, 512]),
            'kernel_size': (4, tune.randint(2, 12)),
            'activation': 'ReLU',
            'batch_norm': True,
            'dropout': tune.uniform(0.0, 0.5)
        },
    ]
    
    conv1d_config = [
        {'out_channels': tune.choice([16, 32, 64, 128, 256]), 
         'kernel_size': (tune.randint(2, 12)), 
         'activation': 'ReLU', 
         'batch_norm': True, 
         'dropout': tune.uniform(0.0, 0.5)},
    ]
    
    linear_config = [
        {'out_features': tune.choice([16, 32, 64, 128, 256])},
        {'out_features': 1}
    ]

    config = {
        'input_channels': 1,
        'conv2d_config': conv2d_config,
        'conv1d_config': conv1d_config,
        'linear_config': linear_config,
        'lr': tune.loguniform(1e-4, 1e-1),
        'batch_size': tune.choice([32, 64, 128, 256]),
        'encode_fn_name': tune.choice(['4xn_v2'])
    }

    os.makedirs(tune_log_dir, exist_ok=True)
    run_config = tune.RunConfig(
        storage_path=tune_log_dir,
        name="my_experiment_name" # Optional, but good practice
    )

    train_fn_with_params = tune.with_parameters(
        train_binnd, 
        data_directory=data_directory,
        max_num_epochs=max_num_epochs,
        max_seq_length=20,
        device=device)

    scheduler = ASHAScheduler(
        time_attr="training_iteration",
        max_t=max_num_epochs,
        grace_period=patience,
        reduction_factor=reduction_factor
    )
    
    tuner = tune.Tuner(
        tune.with_resources(
            train_fn_with_params,
            resources={'cpu':4, 'gpu':1}
        ),
        tune_config=tune.TuneConfig(
            metric="loss",
            mode="min",
            scheduler=scheduler,
            num_samples=num_trials,
        ),
        param_space=config,
        run_config=run_config
    )
    
    results = tuner.fit()

    best_result = results.get_best_result("loss", "min")

    print(f"Best trial config: {best_result.config}")
    print(f"Best trial final validation loss: {best_result.metrics['loss']}")
    print(f"Best trial final validation accuracy: {best_result.metrics['accuracy']}")

def train_binnd(config, data_directory, max_num_epochs, max_seq_length, device):
    batch_size = config['batch_size']
    encode_fn_name = config['encode_fn_name']
    learning_rate = config['lr']

    train_dataset = HTPDataset(data_path=os.path.join(
        data_directory, 'train.csv'), max_seq_length=max_seq_length, encode_fn_name=encode_fn_name)
    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    val_dataset = HTPDataset(data_path=os.path.join(
        data_directory, 'val.csv'), max_seq_length=max_seq_length, encode_fn_name=encode_fn_name)
    val_dataloader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)


    model = GeneralCNNBinaryClassifier(
        input_channels=config['input_channels'],
        conv2d_layers_config=config['conv2d_config'],
        conv1d_layers_config=config['conv1d_config'],
        linear_layers_config=config['linear_config']
    ).to(device)

    print(model)
    criterion = nn.BCELoss()  # Binary Cross-Entropy loss
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    for epoch in range(max_num_epochs):
        
        model.train()
        for i_batch, sample_batched in enumerate(train_dataloader):
            inputs, labels = sample_batched['matrix'].to(device), sample_batched['label'].to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), labels)
            loss.backward()
            optimizer.step()
        
        valid_loss = 0.0
        correct = 0
        total_samples = 0
        model.eval()
        
        for i_batch, sample_batched in enumerate(val_dataloader):
            inputs, labels = sample_batched['matrix'].to(device), sample_batched['label'].to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), labels)
            valid_loss += loss.item() * inputs.size(0)

            # Compute accuracy
            preds = (outputs.squeeze() >= 0.5).float()  # Convert logits to binary predictions
            correct += (preds == labels).sum().item()
            total_samples += inputs.size(0)
        
        metrics = {
            "loss": valid_loss / len(val_dataloader),
            "accuracy": correct / total_samples,
        }
        tune.report(metrics)
    print("Finished Training!")

if __name__ == '__main__':
    main(
        "/gpfs_backup/tuck_data/gbrihad/DNABindML/data/tuning",
        "/gpfs_backup/tuck_data/gbrihad/DNABindML/outputs/ray_test",
        max_num_epochs=10,
        patience=2,
        reduction_factor=2,
        num_trials=2
        )
