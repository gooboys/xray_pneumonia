import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import densenet169, DenseNet169_Weights

class denseB4(nn.Module):
    def __init__(self):
        super(denseB4, self).__init__()
        # Using a pre-trained DenseNet-169
        self.densenet = densenet169(weights=DenseNet169_Weights.IMAGENET1K_V1)  # Use pre-trained weights

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
        self.densenet.classifier = nn.Linear(1664, 832)  # DenseNet-169 has 1664 output features
        self.dropout1 = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(832, 208)
        self.dropout2 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(208, 2)

    def forward(self, x):
        x = F.relu(self.densenet(x))
        x = self.dropout1(x)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)

        return x
