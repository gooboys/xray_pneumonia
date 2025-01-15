import torch.nn as nn
import torch.nn.functional as F

class oneCNN(nn.Module):
    def __init__(self):
        super(oneCNN, self).__init__()
       
        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3)  # Input is grayscale (1 channel)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
       
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
       
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
       
        # Fully connected layers
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(30 * 30 * 128, 256)  # Flattened size from last pooling layer
        self.dropout1 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(256, 128)
        self.dropout2 = nn.Dropout(p=0.5)
        self.fc3 = nn.Linear(128, 3)  # Output layer for 3 classes
       
    def forward(self, x):
        # Forward pass through convolutional and pooling layers
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
       
        # Flatten and pass through fully connected layers
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x