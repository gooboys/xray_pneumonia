import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import densenet169, DenseNet169_Weights
import matplotlib.pyplot as plt


class denseB5(nn.Module):
    def __init__(self, dropout_rate=0.5, hook_layer="7x7"):
        super(denseB5, self).__init__()
        # Using a pre-trained DenseNet-169
        self.densenet = densenet169(weights=DenseNet169_Weights.IMAGENET1K_V1)

        # Adjust the classifier to match the number of classes
        self.densenet.classifier = nn.Linear(1664, 128)  # DenseNet-169 has 1664 output features
        self.dropout1 = nn.Dropout(p=dropout_rate)
        self.fc1 = nn.Linear(128, 2)  # Final classification layer

        # Placeholder for feature maps
        self.feature_maps = None
        self.hook_handle = None  # Store the hook handle for removal
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
            target_layer = self.densenet.features[-1]  # Deepest feature layer
        elif self.hook_layer == "14x14":
            target_layer = self.densenet.features.denseblock3  # Intermediate layer
        elif self.hook_layer == "28x28":
            target_layer = self.densenet.features.denseblock2  # Earlier layer for higher spatial resolution
        else:
            raise ValueError("Invalid hook_layer. Choose '7x7', '14x14', or '28x28'.")

        self.hook_handle = target_layer.register_forward_hook(self._hook_fn)

    def forward(self, x):
        features = self.densenet.features(x)  # Extract convolutional features
        x = F.relu(features)
        x = F.adaptive_avg_pool2d(x, (1, 1)).view(x.size(0), -1)  # Global average pooling
        x = self.densenet.classifier(x)
        x = self.dropout1(x)
        x = self.fc1(x)  # Final classification layer
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

        # Ensure feature maps are on the same device as model parameters
        device = next(self.parameters()).device
        self.feature_maps = self.feature_maps.to(device)

        # Get weights of the last fully connected layer
        fc_weights = self.fc1.weight.data.to(device)

        # Extract the weights corresponding to the target class
        target_weights = fc_weights[target_class]

        # Compute the CAM
        cam = torch.zeros(self.feature_maps.shape[2:], dtype=torch.float32, device=device)
        for i, weight in enumerate(target_weights):
            cam += weight * self.feature_maps[0, i, :, :]

        # Apply ReLU to filter out negative values
        cam = F.relu(cam)

        # Normalize CAM values between 0 and 1
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)

        return cam.cpu().detach().numpy()  # Move to CPU for visualization
