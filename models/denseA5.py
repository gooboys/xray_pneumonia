import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class denseA5(nn.Module):
    def __init__(self,dropout_rate=0.5):
        super(denseA5, self).__init__()
        # Using a pre-trained DenseNet-121
        self.densenet = models.densenet121(pretrained=False)

        # Modify the first convolutional layer to accept grayscale (1-channel) images
        self.densenet.features.conv0 = nn.Conv2d(
            in_channels=1,  # Change input channels to 1
            out_channels=self.densenet.features.conv0.out_channels,  # Keep output channels the same
            kernel_size=self.densenet.features.conv0.kernel_size,
            stride=self.densenet.features.conv0.stride,
            padding=self.densenet.features.conv0.padding,
            bias=False
        )

        # Adjust the classifier to match the number of classes
        self.densenet.classifier = nn.Linear(1024, 128)
        self.dropout1 = nn.Dropout(p=dropout_rate)
        self.fc1 = nn.Linear(128,64)
        self.dropout2 = nn.Dropout(p=dropout_rate)
        self.fc2 = nn.Linear(64,2)

    def forward(self, x):
        x = F.relu(self.densenet(x))
        x = self.dropout1(x)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)

        return x
