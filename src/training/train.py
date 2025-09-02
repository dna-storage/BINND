import torch
import torch.optim as optim
import torch.nn as nn
from sklearn.metrics import accuracy_score
import numpy as np
import os

def train_model(model, train_dataloader, val_dataloader, criterion, optimizer, device, checkpoint_dir, log_dir, num_epochs=5, patience=2, log_interval=500):
    # Open log files
    train_log_path = os.path.join(log_dir, "train_log.csv")
    val_log_path = os.path.join(log_dir, "val_log.csv")
    try:
        train_log = open(train_log_path, "w")
        val_log = open(val_log_path, "w")
        train_log.write("epoch,batch,loss,accuracy\n")
        val_log.write("epoch,batch,loss,accuracy\n")
        print(f"Log files opened: {train_log_path}, {val_log_path}")
    except IOError as e:
        print(f"ERROR: Could not open log files. Please check directory permissions: {e}")
        return # Exit function if logs cannot be written
    
    valid_loss_min = np.inf # track change in validation loss
    epochs_since_improvement = 0  # track epochs since last improvement

    for epoch in range(num_epochs):
        print(f"\n--- Epoch {epoch}/{num_epochs} ---") # Epochs are 0-indexed internally, but 1-indexed for display
        # Track cumulative loss and accuracy for log_interval-batch intervals
        cumulative_loss = 0.0
        cumulative_correct = 0
        cumulative_samples = 0
        
        ######################
        # Training the model #
        ######################
        model.train()
        print("Training phase started...")
        total_batches = len(train_dataloader)
        for i_batch, sample_batched in enumerate(train_dataloader):
            inputs, labels = sample_batched['matrix'].to(device), sample_batched['label'].to(device)
            
            # Clearing the gradients of all optimized variables
            optimizer.zero_grad()
            
            # Forward pass: Computing predicted outputs by passing inputs to the model
            _, outputs = model(inputs)
            
            # Calculating the batch loss
            loss = criterion(outputs.squeeze(), labels)
            
            # Backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()
            
            # Perform a single optimization step (parameter update)
            optimizer.step()

            # Compute accuracy
            preds = (outputs.squeeze() >= 0.5).float()  # Convert to binary predictions
            correct = (preds == labels).sum().item()
            
            # Accumulate loss and accuracy
            cumulative_loss += loss.item() * inputs.size(0)
            cumulative_correct += correct
            cumulative_samples += inputs.size(0)

            if (i_batch + 1) % log_interval == 0:
                avg_loss = cumulative_loss / cumulative_samples
                avg_accuracy = cumulative_correct / cumulative_samples
                print(f"Epoch {epoch}, Batch {i_batch+1}/{total_batches}, Avg. Loss: {avg_loss:.4f}, Avg. Accuracy: {avg_accuracy:.4f}")

                train_log.write(f"{epoch},{i_batch+1},{avg_loss:.4f},{avg_accuracy:.4f}\n")
                train_log.flush()

                # Reset cumulative metrics
                cumulative_loss = 0.0
                cumulative_correct = 0
                cumulative_samples = 0  
            

        ########################
        # Validating the model #
        ########################
        cumulative_loss = 0.0
        cumulative_correct = 0
        cumulative_samples = 0
        valid_loss = 0.0

        model.eval()
        print("Validation phase started...")
        total_batches = len(val_dataloader)
        for i_batch, sample_batched in enumerate(val_dataloader):
            inputs, labels = sample_batched['matrix'].to(device), sample_batched['label'].to(device)
            
            _, outputs = model(inputs)
            loss = criterion(outputs.squeeze(), labels)
            valid_loss += loss.item() * inputs.size(0)

            # Compute accuracy
            preds = (outputs.squeeze() >= 0.5).float()  # Convert logits to binary predictions
            correct = (preds == labels).sum().item()
            
            # Accumulate loss and accuracy
            cumulative_loss += loss.item() * inputs.size(0)
            cumulative_correct += correct
            cumulative_samples += inputs.size(0)

            if (i_batch + 1) % log_interval == 0:
                avg_loss = cumulative_loss / cumulative_samples
                avg_accuracy = cumulative_correct / cumulative_samples
                print(f"Epoch {epoch}, Batch {i_batch+1}/{total_batches}, Avg. Loss: {avg_loss:.4f}, Avg. Accuracy: {avg_accuracy:.4f}")
                val_log.write(f"{epoch},{i_batch+1},{avg_loss:.4f},{avg_accuracy:.4f}\n")
                val_log.flush()

                # Reset cumulative metrics
                cumulative_loss = 0.0
                cumulative_correct = 0
                cumulative_samples = 0
        
        # calculate average losses
        valid_loss = valid_loss/len(val_dataloader.dataset)

        # Saving model if validation loss has decreased
        if valid_loss <= valid_loss_min:
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
                valid_loss_min,valid_loss))
            torch.save(model.state_dict(),  f'{checkpoint_dir}/best_model.pt')
            valid_loss_min = valid_loss
            epochs_since_improvement = 0
        else:
            epochs_since_improvement += 1

        if epochs_since_improvement >= patience:
            print(f'Early stopping triggered after {patience} epochs of no improvement in validation loss.')
            break

    # --- Final Prints ---
    print(f"\n--- Training Finished ---")
    print(f"Best validation loss achieved: {valid_loss_min:.6f}")
    print(f"Model saved to: {checkpoint_dir}/best_model.pt")
    
    train_log.close()
    val_log.close()
    print("Log files closed.")