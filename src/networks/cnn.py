import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNBinaryClassifier2V1(nn.Module):
    def __init__(self):
        super(CNNBinaryClassifier2V1, self).__init__()
        
        # Convolutional layers (no padding)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=0)  # Output: 18x18
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=0)  # Output: 16x16
        
        # Fully connected layers
        self.fc1 = nn.Linear(32 * 16 * 16, 128)  # Flattened input
        self.fc2 = nn.Linear(128, 1)  # Binary classification
        
    def forward(self, x):
        x = F.relu(self.conv1(x))  # Adding channel dimension
        x = F.relu(self.conv2(x))
        
        x = x.view(x.shape[0], -1)  # Flatten the feature maps
        x = F.relu(self.fc1(x))
        return torch.sigmoid(self.fc2(x))  # Binary classification output



class CNNBinaryClassifierV3(nn.Module):
    def __init__(self):
        super(CNNBinaryClassifierV3, self).__init__()

         # 2D Convolutional Block: Extracts spatial features from input
        self.conv2d_block = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=1024, kernel_size=(4, 9)),  # Large feature extraction
            nn.ReLU(),  # Non-linearity
            nn.BatchNorm2d(1024),  # Normalize activations to stabilize training
            nn.Dropout2d(0.30)  # Regularization to prevent overfitting
        )

        # 1D Convolutional Block: Further refines extracted features
        self.conv1d_block = nn.Sequential(
            nn.Conv1d(in_channels=1024, out_channels=256, kernel_size=7),  
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.168),

            nn.Conv1d(in_channels=256, out_channels=512, kernel_size=5),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.138),

            nn.Conv1d(in_channels=512, out_channels=256, kernel_size=5),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.07),

            nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.426)
        )

        # Fully Connected (Linear) Block: Maps extracted features to output
        # self.lin_block = nn.Sequential(
        #     nn.Linear(256 * 16, 256),  # Flattened input projected to 256 neurons
        #     nn.ReLU(),
        #     nn.Linear(256, 128),  # Further reduction
        #     nn.ReLU(),
        #     nn.Dropout(0.2),  # Dropout to prevent overfitting
        #     nn.Linear(128, 1)  # Final output neuron (for binary classification)
        # )
        self.lin_block = nn.Sequential(
            nn.Linear(256 * 16, 512),  # Added layer: Expand to 512 neurons
            nn.ReLU(),
            nn.Linear(512, 256),  # Added layer: Reduce to 256 neurons
            nn.ReLU(),
            nn.Linear(256, 128),  # Existing reduction layer
            nn.ReLU(),
            nn.Dropout(0.2),  # Dropout to prevent overfitting
            nn.Linear(128, 64),  # Added layer: Further reduction to 64 neurons
            nn.ReLU(),
            nn.Linear(64, 1)  # Final output neuron (for binary classification)
        )

        # Sigmoid Activation: Converts logits to probability scores
        self.sigmoid = nn.Sigmoid()
        

    def forward(self, x):
        x = self.conv2d_block(x)
        
        # Remove the third dimension (squeeze along dimension 2) 
        # to reshape the output for 1D convolutions
        x = self.conv1d_block(torch.squeeze(x, 2))
        
        # Flatten the tensor for the fully connected layers
        x = x.view(x.size(0), -1)  # Reshapes to (batch_size, flattened_features)
        
        # Pass through the fully connected (linear) block
        x = self.lin_block(x)
        
        # Apply sigmoid activation for binary classification
        x = self.sigmoid(x)
        
        return x

class CNNBinaryClassifierLite(nn.Module):
    def __init__(self):
        super(CNNBinaryClassifierLite, self).__init__()

         # 2D Convolutional Block: Extracts spatial features from input
        self.conv2d_block = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=256, kernel_size=(4, 9)),  # Large feature extraction
            nn.ReLU(),  # Non-linearity
            nn.BatchNorm2d(256),  # Normalize activations to stabilize training
            nn.Dropout2d(0.30)  # Regularization to prevent overfitting
        )

        # 1D Convolutional Block: Further refines extracted features
        self.conv1d_block = nn.Sequential(
            nn.Conv1d(in_channels=256, out_channels=128, kernel_size=7),  
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.168),
        )

        # 👇 Dummy forward pass to compute the flattened size
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, 4, 40)  # Shape: (batch_size=1, channels=1, height=4, width=40)
            out = self.conv2d_block(dummy_input)
            out = out.squeeze(2)  # Remove height dim (1)
            out = self.conv1d_block(out)
            self.flattened_size = out.view(1, -1).shape[1]

        # Fully Connected (Linear) Block: Maps extracted features to output
        self.lin_block = nn.Sequential(
            nn.Linear(self.flattened_size, 256),  # Flattened input projected to 256 neurons
            nn.ReLU(),
            nn.Linear(256, 1)  # Final output neuron (for binary classification)
        )
     
        # Sigmoid Activation: Converts logits to probability scores
        self.sigmoid = nn.Sigmoid()
        

    def forward(self, x):
        x = self.conv2d_block(x)
        
        # Remove the third dimension (squeeze along dimension 2) 
        # to reshape the output for 1D convolutions
        x = self.conv1d_block(torch.squeeze(x, 2))
        
        # Flatten the tensor for the fully connected layers
        x = x.view(x.size(0), -1)  # Reshapes to (batch_size, flattened_features)
        
        # Pass through the fully connected (linear) block
        x = self.lin_block(x)
        
        # Apply sigmoid activation for binary classification
        x = self.sigmoid(x)
        
        return x

class CNNBinaryClassifierV3Original(nn.Module):
    def __init__(self):
        super(CNNBinaryClassifierV3Original, self).__init__()

         # 2D Convolutional Block: Extracts spatial features from input
        self.conv2d_block = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=1024, kernel_size=(4, 9)),  # Large feature extraction
            nn.ReLU(),  # Non-linearity
            nn.BatchNorm2d(1024),  # Normalize activations to stabilize training
            nn.Dropout2d(0.30)  # Regularization to prevent overfitting
        )

        # 1D Convolutional Block: Further refines extracted features
        self.conv1d_block = nn.Sequential(
            nn.Conv1d(in_channels=1024, out_channels=256, kernel_size=7),  
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.168),

            nn.Conv1d(in_channels=256, out_channels=512, kernel_size=5),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.138),

            nn.Conv1d(in_channels=512, out_channels=256, kernel_size=5),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.07),

            nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.426)
        )

        # Fully Connected (Linear) Block: Maps extracted features to output
        self.lin_block = nn.Sequential(
            nn.Linear(256 * 16, 256),  # Flattened input projected to 256 neurons
            nn.ReLU(),
            nn.Linear(256, 128),  # Further reduction
            nn.ReLU(),
            nn.Dropout(0.2),  # Dropout to prevent overfitting
            nn.Linear(128, 1)  # Final output neuron (for binary classification)
        )
     
        # Sigmoid Activation: Converts logits to probability scores
        self.sigmoid = nn.Sigmoid()
        

    def forward(self, x):
        x = self.conv2d_block(x)
        
        # Remove the third dimension (squeeze along dimension 2) 
        # to reshape the output for 1D convolutions
        x = self.conv1d_block(torch.squeeze(x, 2))
        
        # Flatten the tensor for the fully connected layers
        x = x.view(x.size(0), -1)  # Reshapes to (batch_size, flattened_features)
        
        # Pass through the fully connected (linear) block
        x = self.lin_block(x)
        
        # Apply sigmoid activation for binary classification
        x = self.sigmoid(x)
        
        return x