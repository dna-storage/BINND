import torch
import torch.nn as nn
import torch.nn.functional as F


class GeneralCNNBinaryClassifier(nn.Module):
    """_summary_
    A general purpose Convolutional Neural Network class with both 2D and 1D convolutions.

    This class allows for flexible definition of:
    - 2D Convolutional layers (Conv2d, ReLU, BatchNorm2d, Dropout2d)
    - 2D Pooling layers (MaxPool2d, AvgPool2d)
    - 1D Convolutional layers (Conv1d, ReLU, BatchNorm1d, Dropout)
    - 1D Pooling layers (MaxPool1d, AvgPool1d)
    - Fully Connected / Linear layers (Linear, ReLU, Dropout)

    The architecture is defined by passing lists of dictionaries for each
    block type, allowing for an arbitrary number of layers within each block.
    """

    def __init__(self,
                 input_channels,
                 conv2d_layers_config,
                 pooling2d_layers_config=None,
                 conv1d_layers_config=None,
                 pooling1d_layers_config=None,
                 linear_layers_config=None,
                 ):

        super(GeneralCNNBinaryClassifier, self).__init__()

    
        # --- Build 2D Convolutional Block ---
        conv2d_modules = []
        current_in_channels_2d = input_channels
        for config in conv2d_layers_config:
            out_channels = config['out_channels']
            kernel_size = config['kernel_size']
            stride = config.get('stride', 1)
            padding = config.get('padding', 0)

            conv2d_modules.append(nn.Conv2d(in_channels=current_in_channels_2d,
                                            out_channels=out_channels,
                                            kernel_size=kernel_size,
                                            stride=stride,
                                            padding=padding)
                                  )
            conv2d_modules.append(nn.ReLU())
            
            if ('batch_norm' in config and config['batch_norm'] == True):
                conv2d_modules.append(nn.BatchNorm2d(out_channels))
            
            if ('dropout' in config and config['dropout'] > 0):
                conv2d_modules.append(nn.Dropout2d(config['dropout']))
            
            current_in_channels_2d = out_channels
            
        self.conv2d_block = nn.Sequential(*conv2d_modules)
        
    
        
            


class CNNBinaryClassifier2V1(nn.Module):
    def __init__(self):
        super(CNNBinaryClassifier2V1, self).__init__()

        # Convolutional layers (no padding)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16,
                               kernel_size=3, stride=1, padding=0)  # Output: 18x18
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32,
                               kernel_size=3, stride=1, padding=0)  # Output: 16x16

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
            nn.Conv2d(in_channels=1, out_channels=1024,
                      kernel_size=(4, 9)),  # Large feature extraction
            nn.ReLU(),  # Non-linearity
            # Normalize activations to stabilize training
            nn.BatchNorm2d(1024),
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
        # Reshapes to (batch_size, flattened_features)
        x = x.view(x.size(0), -1)

        # Pass through the fully connected (linear) block
        x = self.lin_block(x)

        # Apply sigmoid activation for binary classification
        x = self.sigmoid(x)

        return x


class BINNDLite(nn.Module):
    def __init__(self):
        super(BINNDLite, self).__init__()

        # 2D Convolutional Block: Extracts spatial features from input
        self.conv2d_block = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=256,
                      kernel_size=(4, 9)),  # Large feature extraction
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
            # Shape: (batch_size=1, channels=1, height=4, width=40)
            dummy_input = torch.zeros(1, 1, 4, 40)
            out = self.conv2d_block(dummy_input)
            out = out.squeeze(2)  # Remove height dim (1)
            out = self.conv1d_block(out)
            self.flattened_size = out.view(1, -1).shape[1]

        # Fully Connected (Linear) Block: Maps extracted features to output
        self.lin_block = nn.Sequential(
            # Flattened input projected to 256 neurons
            nn.Linear(self.flattened_size, 256),
            nn.ReLU(),
            # Final output neuron (for binary classification)
            nn.Linear(256, 1)
        )

        # Sigmoid Activation: Converts logits to probability scores
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv2d_block(x)

        # Remove the third dimension (squeeze along dimension 2)
        # to reshape the output for 1D convolutions
        x = self.conv1d_block(torch.squeeze(x, 2))

        # Flatten the tensor for the fully connected layers
        # Reshapes to (batch_size, flattened_features)
        x = x.view(x.size(0), -1)

        # Pass through the fully connected (linear) block
        x = self.lin_block(x)

        # Apply sigmoid activation for binary classification
        x = self.sigmoid(x)

        return x


class BINND(nn.Module):
    def __init__(self):
        super(BINND, self).__init__()

        # 2D Convolutional Block: Extracts spatial features from input
        self.conv2d_block = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=1024,
                      kernel_size=(4, 9)),  # Large feature extraction
            nn.ReLU(),  # Non-linearity
            # Normalize activations to stabilize training
            nn.BatchNorm2d(1024),
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
            # Flattened input projected to 256 neurons
            nn.Linear(256 * 16, 256),
            nn.ReLU(),
            nn.Linear(256, 128),  # Further reduction
            nn.ReLU(),
            nn.Dropout(0.2),  # Dropout to prevent overfitting
            # Final output neuron (for binary classification)
            nn.Linear(128, 1)
        )

        # Sigmoid Activation: Converts logits to probability scores
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv2d_block(x)

        # Remove the third dimension (squeeze along dimension 2)
        # to reshape the output for 1D convolutions
        x = self.conv1d_block(torch.squeeze(x, 2))

        # Flatten the tensor for the fully connected layers
        # Reshapes to (batch_size, flattened_features)
        x = x.view(x.size(0), -1)

        # Pass through the fully connected (linear) block
        x = self.lin_block(x)

        # Apply sigmoid activation for binary classification
        x = self.sigmoid(x)

        return x


if __name__ == "__main__":
    conv2d_config = [
        {'out_channels': 32, 'kernel_size': (4, 9), 'batch_norm': True, 'dropout': 0.2}, # Kernel (4,9) on H=4 input -> H=1 output
        {'out_channels': 64, 'kernel_size': (1, 5), 'padding': (0, 0)}, # Example: another 2D layer reducing width
    ]
    
    model = GeneralCNNBinaryClassifier(1, conv2d_layers_config=conv2d_config)
    print(model)