import torch
import torch.nn as nn
from torchvision import models


class teff42(nn.Module):
    def __init__(self, dropout_rate=0.5, l2_weight_decay=1e-4):
        super(teff42, self).__init__()
        # Load the pre-trained EfficientNet-B0
        self.efficientnet = models.efficientnet_b0(pretrained=True)

        # Modify the first convolutional layer to accept grayscale (1-channel) images
        self.efficientnet.features[0][0] = nn.Conv2d(
            in_channels=1,  # Change input channels to 1
            out_channels=self.efficientnet.features[0][0].out_channels,  # Keep output channels the same
            kernel_size=self.efficientnet.features[0][0].kernel_size,
            stride=self.efficientnet.features[0][0].stride,
            padding=self.efficientnet.features[0][0].padding,
            bias=False
        )

        # Adjust the classifier to directly take the feature map output (without pooling)
        self.fc1 = nn.Linear(1280 * 7 * 7, 256)  # Input size = channels * feature map size (7x7)
        self.dropout1 = nn.Dropout(p=dropout_rate)
        self.fc2 = nn.Linear(256, 64)
        self.dropout2 = nn.Dropout(p=dropout_rate)
        self.fc3 = nn.Linear(64, 2)

        # Store L2 regularization weight decay
        self.l2_weight_decay = l2_weight_decay

    def forward(self, x):
        # Pass input through EfficientNet feature extractor
        x = self.efficientnet.features(x)  # Feature map output with shape [batch_size, 1280, 7, 7]

        # Flatten the feature map before passing through the classifier
        x = torch.flatten(x, 1)  # Flatten to [batch_size, 1280 * 7 * 7]

        # Pass through custom classifier
        x = nn.functional.relu(self.fc1(x))
        x = self.dropout1(x)
        x = nn.functional.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)

        return x

# Example usage with L2 regularization in the optimizer
model = teff42(dropout_rate=0.5)

# Define optimizer with L2 regularization (weight decay)
import torch.optim as optim
optimizer = optim.Adam(
    model.parameters(),  # Optimize all parameters since weight freezing is removed
    lr=1e-4,
    weight_decay=model.l2_weight_decay  # L2 regularization
)
