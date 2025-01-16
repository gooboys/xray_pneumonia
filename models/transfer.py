import torch.nn as nn
import torchxrayvision as xrv

class transfer(nn.Module):
    def __init__(self):
        super(transfer, self).__init__()
        # Load the pre-trained DenseNet model
        self.base_model = xrv.models.DenseNet(weights="all")
        # Replace the fully connected (fc) layer
        self.base_model.classifier = nn.Sequential(
            nn.Linear(self.base_model.classifier.in_features, 1),  # 1 output for binary classification
        )

    def forward(self, x):
        return self.base_model(x)