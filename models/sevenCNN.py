import torch.nn as nn
import torch.nn.functional as F

class sevenCNN(nn.Module):
    def __init__(self):
        super(sevenCNN, self).__init__()
       
        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3)  # Input is grayscale (1 channel)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
       
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
       
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
       
        # Fully connected layers
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(30 * 30 * 128, 2048)  # Flattened size from last pooling layer
        self.dropout1 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(2048, 1024)  # Output layer for 3 classes
        self.dropout2 = nn.Dropout(p=0.5)
        self.fc3 = nn.Linear(1024, 512)
        self.dropout3 = nn.Dropout(p=0.5)
        self.fc4 = nn.Linear(512, 256)
        self.dropout4 = nn.Dropout(p=0.5)
        self.fc5 = nn.Linear(256, 128)
        self.dropout5 = nn.Dropout(p=0.5)
        self.fc6 = nn.Linear(128, 64)
        self.dropout6 = nn.Dropout(p=0.5)
        self.fc7 = nn.Linear(64,3)
       
    def forward(self, x):
        # Forward pass through convolutional and pooling layers
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
       
        # Flatten the output from the convolutional layers
        x = self.flatten(x)
        
        # Fully connected layers with ReLU activation and dropout
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        
        x = F.relu(self.fc3(x))
        x = self.dropout3(x)
        
        x = F.relu(self.fc4(x))
        x = self.dropout4(x)
        
        x = F.relu(self.fc5(x))
        x = self.dropout5(x)
        
        x = F.relu(self.fc6(x))
        x = self.dropout6(x)

        x = self.fc7(x)
        return x