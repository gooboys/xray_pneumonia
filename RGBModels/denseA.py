import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class denseA(nn.Module):
    def __init__(self):
        super(denseA, self).__init__()
        # Using a pre-trained DenseNet-121
        self.densenet = models.densenet121(pretrained=True)

        # Adjust the classifier to match the number of classes
        self.densenet.classifier = nn.Linear(1024, 128)
        self.dropout1 = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(128,2)

    def forward(self, x):
        x = F.relu(self.densenet(x))
        x = self.dropout1(x)
        x = self.fc1(x)
        return x
