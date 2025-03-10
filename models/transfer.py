import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class transfer(nn.Module):
    def __init__(self):
        super(transfer, self).__init__()
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

        # Adjust the classifier to match the number of classes
        self.densenet.classifier = nn.Linear(1024, 128)
        self.dropout1 = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(128,2)

    def forward(self, x):
        x = F.relu(self.densenet(x))
        x = self.dropout1(x)
        x = self.fc1(x)
        return x
