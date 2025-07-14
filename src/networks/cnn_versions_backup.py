"""
=== 1 === model_name: CNNBinaryClassifierV3_0.1_encoder4xn_v2_htp_oct_2024_cond2
{
    "accuracy": 0.8342391558046693,
    "AUC": 0.8841219791830559,
    "precision": 0.9300190182443101,
    "recall": 0.7228601731956208,
    "F1-score": 0.8134577676153594,
    "true_positive": 5226094,
    "false_positive": 393247,
    "true_negative": 6836980,
    "false_negative": 2003650
}
   
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

"""