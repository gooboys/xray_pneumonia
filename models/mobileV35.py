import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class mobileV35(nn.Module):
    def __init__(self, version='large'):
        super(mobileV35, self).__init__()
        # Load pre-trained MobileNetV3 (Large or Small)
        if version == 'large':
            self.mobilenet = models.mobilenet_v3_large(pretrained=False)
        else:
            self.mobilenet = models.mobilenet_v3_small(pretrained=False)
        
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
            nn.Linear(self.mobilenet.classifier[0].in_features, 128),
            nn.BatchNorm1d(128),  # BatchNorm applied to stabilize activations
            nn.ReLU(),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        x = self.mobilenet(x)
        return x
