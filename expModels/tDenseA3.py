import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class tDenseA3(nn.Module):
    def __init__(self, dropout_rate=0.5, freeze_base=True, l2_weight_decay=1e-4):
        super(tDenseA3, self).__init__()
        # Using a pre-trained DenseNet-121
        self.densenet = models.densenet121(pretrained=True)
        
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
        self.fc1 = nn.Linear(1024, 256)  # DenseNet-121 output channels = 1024
        self.dropout1 = nn.Dropout(p=dropout_rate)
        self.fc2 = nn.Linear(256, 128)
        self.dropout2 = nn.Dropout(p=dropout_rate)
        self.fc3 = nn.Linear(128, 2)

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