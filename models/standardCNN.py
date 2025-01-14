import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image, ImageEnhance

transform = transforms.ToTensor()  # Convert images to PyTorch tensors
# Check if a GPU is available and set the device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class standardCNN(nn.Module):
    def __init__(self):
        super(standardCNN, self).__init__()
       
        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3)  # Input is grayscale (1 channel)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
       
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
       
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
       
        # Fully connected layers
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(30 * 30 * 128, 128)  # Flattened size from last pooling layer
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(128, 3)  # Output layer for 3 classes
       
    def forward(self, x):
        # Forward pass through convolutional and pooling layers
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
       
        # Flatten and pass through fully connected layers
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
   
    def generate_cam(self, image_path, class_idx):
        """
        Generates the Class Activation Map (CAM) for a given class index.
        """
        self.to(device)
        # Load and preprocess the image
        original_image = Image.open(image_path).convert("L")  # Load as grayscale
    
        # Apply transformations and convert to tensor
        input_image = transform(original_image).unsqueeze(0).to(device)  # Add batch dimension and move to GPU

        # Forward pass to get feature maps and logits
        self.eval()
        with torch.no_grad():
            features = self.pool3(F.relu(self.conv3(
                        self.pool2(F.relu(self.conv2(
                            self.pool1(F.relu(self.conv1(input_image))))
                        ))
            )))
            logits = self.fc2(F.relu(self.fc1(features.view(features.size(0), -1))))  # Same as your previous forward pass

        # Get the weights for the target class from the final fully connected layer
        fc_weights = self.fc2.weight[class_idx].detach()

        # Generate CAM by weighted sum of feature maps
        cam = torch.zeros(features.size(2), features.size(3), device=features.device)
        for i in range(features.size(1)):  # Iterate over channels
            cam += fc_weights[i] * features[0, i, :, :]

        cam = torch.relu(cam)  # Apply ReLU to remove negative values

        # Normalize CAM for visualization
        cam -= cam.min()
        cam /= cam.max()
        cam = cam.cpu().numpy()

        # Resize CAM to the original image dimensions
        cam_image = Image.fromarray((cam * 255).astype('uint8')).resize(original_image.size, Image.LANCZOS)

        # Convert CAM to heatmap
        heatmap = cam_image.convert("L")

        # Overlay CAM onto the original image
        overlay = Image.blend(original_image.convert("L"), heatmap, alpha=0.5)

        # Enhance the overlay to make it more visible
        enhancer = ImageEnhance.Contrast(overlay)
        overlay = enhancer.enhance(1.5)

        return overlay, heatmap