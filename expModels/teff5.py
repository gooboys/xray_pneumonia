import torch
import torch.nn as nn
from torchvision import models


class teff5(nn.Module):
    def __init__(self, dropout_rate=0.5, freeze_base=True, l2_weight_decay=1e-4):
        super(teff5, self).__init__()
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

        # Optionally freeze the base model layers
        if freeze_base:
            for param in self.efficientnet.features.parameters():
                param.requires_grad = False

        # Replace the classifier with Global Average Pooling + custom classifier
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))  # GAP layer
        self.fc1 = nn.Linear(1280, 128)  # EfficientNet-B0 output channels = 1280
        self.dropout1 = nn.Dropout(p=dropout_rate)
        self.fc2 = nn.Linear(128, 64)
        self.dropout2 = nn.Dropout(p=dropout_rate)
        self.fc3 = nn.Linear(64, 2)

        # Store L2 regularization weight decay
        self.l2_weight_decay = l2_weight_decay

    def forward(self, x):
        # Pass input through EfficientNet feature extractor
        x = self.efficientnet.features(x)

        # Apply Global Average Pooling
        x = self.global_avg_pool(x)
        x = torch.flatten(x, 1)  # Flatten the GAP output

        # Pass through custom classifier
        x = nn.functional.relu(self.fc1(x))
        x = self.dropout1(x)
        x = nn.functional.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)

        return x

# Example usage with L2 regularization in the optimizer
model = teff5(dropout_rate=0.5, freeze_base=True)

# Define optimizer with L2 regularization (weight decay)
import torch.optim as optim
optimizer = optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),  # Only optimize unfrozen parameters
    lr=1e-4,
    weight_decay=model.l2_weight_decay  # L2 regularization
)
