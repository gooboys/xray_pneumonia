import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class eff2(nn.Module):
    def __init__(self):
        super(eff2, self).__init__()
        # Load the pre-trained EfficientNet-B0
        self.efficientnet = models.efficientnet_b0(pretrained=False)

        # Modify the first convolutional layer to accept grayscale (1-channel) images
        self.efficientnet.features[0][0] = nn.Conv2d(
            in_channels=1,  # Change input channels to 1
            out_channels=self.efficientnet.features[0][0].out_channels,  # Keep output channels the same
            kernel_size=self.efficientnet.features[0][0].kernel_size,
            stride=self.efficientnet.features[0][0].stride,
            padding=self.efficientnet.features[0][0].padding,
            bias=False
        )

        # Replace the classifier for the number of classes (2 for binary classification)
        self.efficientnet.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(self.efficientnet.classifier[1].in_features, 512),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(128, 2)  # Binary classification
        )

    def forward(self, x):
        x = self.efficientnet(x)
        return x
