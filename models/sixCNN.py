import torch.nn as nn
import torch.nn.functional as F

class sixCNN(nn.Module):
    def __init__(self):
        super(sixCNN, self).__init__()
       
        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3)  # Input is grayscale (1 channel)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
       
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
    
       
        # Fully connected layers
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(54*54*64, 512)  # Flattened size from last pooling layer
        self.dropout1 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(512, 128)
        self.dropout2 = nn.Dropout(p=0.5)
        self.fc3 = nn.Linear(128, 2)  # Output layer for 3 classes
       
    def forward(self, x):
        # Forward pass through convolutional and pooling layers
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
       
        # Flatten and pass through fully connected layers
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x