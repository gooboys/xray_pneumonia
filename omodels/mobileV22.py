import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class mobileV22(nn.Module):
    def __init__(self):
        super(mobileV22, self).__init__()
        # Load pre-trained MobileNetV2
        self.mobilenet = models.mobilenet_v2(pretrained=True)
        
        # Modify the first convolutional layer to accept grayscale images
        self.mobilenet.features[0][0] = nn.Conv2d(
            in_channels=1,
            out_channels=self.mobilenet.features[0][0].out_channels,
            kernel_size=self.mobilenet.features[0][0].kernel_size,
            stride=self.mobilenet.features[0][0].stride,
            padding=self.mobilenet.features[0][0].padding,
            bias=False
        )
        
        # Replace the classifier for binary classification
        self.mobilenet.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(self.mobilenet.last_channel, 128),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        x = self.mobilenet(x)
        return x
