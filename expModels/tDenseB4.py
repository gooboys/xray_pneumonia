import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import densenet169, DenseNet169_Weights

class tDenseB4(nn.Module):
    def __init__(self, dropout_rate=0.5, freeze_base=True, l2_weight_decay=1e-4):
        super(tDenseB4, self).__init__()
        # Using a pre-trained DenseNet-169
        self.densenet = densenet169(weights=DenseNet169_Weights.IMAGENET1K_V1)

        # Modify the first convolutional layer to accept grayscale (1-channel) images
        self.densenet.features.conv0 = nn.Conv2d(
            in_channels=1,  # Change input channels to 1
            out_channels=self.densenet.features.conv0.out_channels,  # Keep output channels the same
            kernel_size=self.densenet.features.conv0.kernel_size,
            stride=self.densenet.features.conv0.stride,
            padding=self.densenet.features.conv0.padding,
            bias=False
        )

        # Optionally freeze the base model layers
        if freeze_base:
            for param in self.densenet.features.parameters():
                param.requires_grad = False

        # Replace classifier with Global Average Pooling + custom classifier
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))  # GAP layer
        self.fc1 = nn.Linear(1664, 832)  # DenseNet-169 output channels = 1664
        self.dropout1 = nn.Dropout(p=dropout_rate)
        self.fc2 = nn.Linear(832, 208)
        self.dropout2 = nn.Dropout(p=dropout_rate)
        self.fc3 = nn.Linear(208, 2)

        # Store L2 regularization weight decay
        self.l2_weight_decay = l2_weight_decay

    def forward(self, x):
        # Pass input through DenseNet feature extractor
        x = self.densenet.features(x)

        # Apply Global Average Pooling
        x = self.global_avg_pool(x)
        x = torch.flatten(x, 1)  # Flatten the GAP output

        # Pass through custom classifier
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)

        return x

# Example usage with L2 regularization in the optimizer
model = tDenseB4(dropout_rate=0.5, freeze_base=True)

# Define optimizer with L2 regularization (weight decay)
import torch.optim as optim
optimizer = optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),  # Only optimize unfrozen parameters
    lr=1e-4,
    weight_decay=model.l2_weight_decay  # L2 regularization
)
