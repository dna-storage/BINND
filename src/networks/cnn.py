import torch
import torch.nn as nn
import torch.nn.functional as F


class GeneralCNNBinaryClassifier(nn.Module):
    """_summary_
    A general purpose Convolutional Neural Network class with both 2D and 1D convolutions.

    This class allows for flexible definition of:
    - 2D Convolutional layers 
    - 1D Convolutional layers
    - Fully Connected / Linear layers

    The architecture is defined by passing lists of dictionaries for each
    block type, allowing for an arbitrary number of layers within each block.
    """

    def __init__(self,
                 input_channels,
                 conv2d_layers_config,
                 conv1d_layers_config=None,
                 linear_layers_config=None,
                 ):

        super(GeneralCNNBinaryClassifier, self).__init__()

        def _get_activation(config):
            activation_map = {
                'relu': nn.ReLU,
                'leakyrelu': nn.LeakyReLU,
                'elu': nn.ELU,
                'gelu': nn.GELU,
                'tanh': nn.Tanh,
                'sigmoid': nn.Sigmoid,
                # Add more as needed
            }

            activation = config.get('activation')
            if activation:
                activation = activation.lower()
                if activation in activation_map:
                    return activation_map[activation]()
                else:
                    raise ValueError(
                        f"Unsupported activation: {activation}")
            else:
                return None

        # --- Build 2D Convolutional Block ---
        conv2d_modules = []
        current_in_channels = input_channels
        for config in conv2d_layers_config:
            out_channels = config['out_channels']
            kernel_size = config['kernel_size']
            stride = config.get('stride', 1)
            padding = config.get('padding', 0)

            conv2d_modules.append(nn.Conv2d(in_channels=current_in_channels,
                                            out_channels=out_channels,
                                            kernel_size=kernel_size,
                                            stride=stride,
                                            padding=padding)
                                  )

            activation_layer = _get_activation(config)
            if activation_layer:
                conv2d_modules.append(activation_layer)

            if config.get('batch_norm'):
                conv2d_modules.append(nn.BatchNorm2d(out_channels))

            dropout = config.get('dropout', 0)
            if dropout > 0:
                conv2d_modules.append(nn.Dropout2d(config['dropout']))

            current_in_channels = out_channels

        self.conv2d_block = nn.Sequential(*conv2d_modules)

        # --- Build 1D Convolutional Block ---
        conv1d_modules = []
        if conv1d_layers_config:
            for config in conv1d_layers_config:
                out_channels = config['out_channels']
                kernel_size = config['kernel_size']
                stride = config.get('stride', 1)
                padding = config.get('padding', 0)

                conv1d_modules.append(nn.Conv1d(in_channels=current_in_channels,
                                                out_channels=out_channels,
                                                kernel_size=kernel_size,
                                                stride=stride,
                                                padding=padding))

                activation_layer = _get_activation(config)
                if activation_layer:
                    conv1d_modules.append(activation_layer)

                if config.get('batch_norm'):
                    conv1d_modules.append(nn.BatchNorm1d(out_channels))

                dropout = config.get('dropout', 0)
                if dropout > 0:
                    conv1d_modules.append(nn.Dropout(config['dropout']))

                current_in_channels = out_channels

        self.conv1d_block = nn.Sequential(
            *conv1d_modules) if conv1d_modules else nn.Identity()

        # --- Build Fully Connected (Linear) Block ---

        linear_modules = []
        if linear_layers_config:

            with torch.no_grad():
                # Shape: (batch_size=1, channels=1, height=4, width=40)
                dummy_input = torch.zeros(1, 1, 4, 40)
                out = self.conv2d_block(dummy_input)
                out = out.squeeze(2)  # Remove height dim (1)
                out = self.conv1d_block(out)
                self.flattened_size = out.view(1, -1).shape[1]

            current_in_features = self.flattened_size

            for config in linear_layers_config:
                out_features = config['out_features']

                linear_modules.append(nn.Linear(in_features=current_in_features,
                                                out_features=out_features))

                activation_layer = _get_activation(config)
                if activation_layer:
                    linear_modules.append(activation_layer)

                if config.get('batch_norm'):
                    linear_modules.append(nn.BatchNorm1d(out_features))

                dropout = config.get('dropout', 0)
                if dropout > 0:
                    linear_modules.append(nn.Dropout(config['dropout']))

                current_in_features = out_features

        self.linear_block = nn.Sequential(
            *linear_modules) if linear_modules else nn.Identity()

        # Sigmoid Activation: Converts logits to probability scores
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv2d_block(x)

        # Assuming height becomes 1, remove the third dimension (squeeze along dimension 2)
        # to reshape the output for 1D convolutions
        x = x.squeeze(2)
        x = self.conv1d_block(x)

        # Flatten the tensor for the fully connected layers
        # Reshapes to (batch_size, flattened_features)
        x = x.view(x.size(0), -1)

        # Pass through the fully connected (linear) block
        x = self.linear_block(x)

        # Apply sigmoid activation for binary classification
        x = self.sigmoid(x)

        return x


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
        {'out_channels': 256, 'kernel_size': (
            4, 9), 'activation': 'ReLU', 'batch_norm': True, 'dropout': 0.3},
    ]

    conv1d_config = [
        {'out_channels': 128, 'kernel_size': (
            7), 'activation': 'ReLU', 'batch_norm': True, 'dropout': 0.168},
    ]

    linear_config = [
        {'out_features': 256},
        {'out_features': 1}
    ]

    model = GeneralCNNBinaryClassifier(
        1, conv2d_layers_config=conv2d_config, conv1d_layers_config=conv1d_config, linear_layers_config=linear_config)
    print(model)
