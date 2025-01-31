import torch
import torch.nn.functional as F
from torch import nn
from torchvision import models
import matplotlib.pyplot as plt


class denseA3(nn.Module):
    def __init__(self, dropout_rate=0.5, hook_layer="7x7"):
        super(denseA3, self).__init__()
        # Using a pre-trained DenseNet-121
        self.densenet = models.densenet121(pretrained=True)

        # Modify the first convolutional layer to accept grayscale (1-channel) images
        self.densenet.features.conv0 = nn.Conv2d(
            in_channels=1,
            out_channels=self.densenet.features.conv0.out_channels,
            kernel_size=self.densenet.features.conv0.kernel_size,
            stride=self.densenet.features.conv0.stride,
            padding=self.densenet.features.conv0.padding,
            bias=False
        )

        # Adjust the classifier to match the number of classes
        self.densenet.classifier = nn.Linear(1024, 256)
        self.dropout1 = nn.Dropout(p=dropout_rate)
        self.fc1 = nn.Linear(256, 128)
        self.dropout2 = nn.Dropout(p=dropout_rate)
        self.fc2 = nn.Linear(128, 2)

        # Placeholder for feature maps
        self.feature_maps = None
        self.hook_handle = None  # Store the handle for hook removal
        self.hook_layer = hook_layer

        # Register the hook dynamically
        self._register_hook()

    def _hook_fn(self, module, input, output):
        """Hook function to capture feature maps."""
        self.feature_maps = output

    def _register_hook(self):
        """Registers the forward hook based on the chosen layer."""
        if self.hook_handle:
            self.hook_handle.remove()  # Remove previous hook if it exists

        if self.hook_layer == "7x7":
            target_layer = self.densenet.features[-1]  # Deepest feature layer (default)
        elif self.hook_layer == "14x14":
            target_layer = self.densenet.features.denseblock3  # Intermediate layer
        elif self.hook_layer == "28x28":
            target_layer = self.densenet.features.denseblock2  # Earlier layer for higher spatial resolution
        else:
            raise ValueError("Invalid hook_layer. Choose '7x7', '14x14', or '28x28'.")

        self.hook_handle = target_layer.register_forward_hook(self._hook_fn)

    def forward(self, x):
        features = self.densenet.features(x)  # Extract features from DenseNet
        x = F.relu(features)
        x = F.adaptive_avg_pool2d(x, (1, 1)).view(x.size(0), -1)
        x = self.densenet.classifier(x)
        x = self.dropout1(x)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        return x

    def generate_cam(self, target_class, hook_layer=None):
        """
        Generate a Class Activation Map (CAM) for the specified target class.
        Allows switching hook layers dynamically.
        """
        if hook_layer and hook_layer != self.hook_layer:
            self.hook_layer = hook_layer
            self._register_hook()

        if self.feature_maps is None:
            raise ValueError("Feature maps are not available. Ensure a forward pass is completed first.")

        # Ensure the feature maps and weights are on the same device as the model
        device = next(self.parameters()).device
        self.feature_maps = self.feature_maps.to(device)

        # Get the weights of the final fully connected layer and move them to the correct device
        fc_weights = self.fc2.weight.data.to(device)

        # Extract the weights for the target class
        target_weights = fc_weights[target_class]

        # Compute the CAM
        cam = torch.zeros(self.feature_maps.shape[2:], dtype=torch.float32, device=device)  # Initialize the CAM
        for i, weight in enumerate(target_weights):
            cam += weight * self.feature_maps[0, i, :, :]

        # Apply ReLU to the CAM
        cam = F.relu(cam)

        # Normalize the CAM
        cam = cam - cam.min()
        cam = cam / cam.max()

        return cam.cpu().detach().numpy()  # Move to CPU for visualization
