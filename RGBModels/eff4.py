import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class eff4(nn.Module):
    def __init__(self,dropout_rate=0.5):
        super(eff4, self).__init__()
        # Load the pre-trained EfficientNet-B0
        self.efficientnet = models.efficientnet_b0(pretrained=True)

        # Replace the classifier for the number of classes (2 for binary classification)
        self.efficientnet.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(self.efficientnet.classifier[1].in_features, 2),
        )

    def forward(self, x):
        x = self.efficientnet(x)
        return x
