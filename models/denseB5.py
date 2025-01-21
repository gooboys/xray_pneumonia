import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import densenet169, DenseNet169_Weights

class denseB5(nn.Module):
    def __init__(self, dropout_rate=0.5):
        super(denseB5, self).__init__()
        # Using a pre-trained DenseNet-169
        self.densenet = densenet169(weights=DenseNet169_Weights.IMAGENET1K_V1)  # Use pre-trained weights DenseNet169_Weights.IMAGENET1K_V1

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
        self.densenet.classifier = nn.Linear(1664, 128)  # DenseNet-169 has 1664 output features
        self.dropout1 = nn.Dropout(p=dropout_rate)
        self.fc1 = nn.Linear(128, 2)

    def forward(self, x):
        x = F.relu(self.densenet(x))
        x = self.dropout1(x)
        x = self.fc1(x)

        return x
