import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class eff4(nn.Module):
    def __init__(self, dropout_rate=0.5, hook_layer="7x7"):
        super(eff4, self).__init__()
        # Load the pre-trained EfficientNet-B0
        self.efficientnet = models.efficientnet_b0(pretrained=True)

        # Modify the first convolutional layer to accept grayscale (1-channel) images
        self.efficientnet.features[0][0] = nn.Conv2d(
            in_channels=1,  # Change input channels to 1
            out_channels=self.efficientnet.features[0][0].out_channels,  # Keep output channels the same
            kernel_size=self.efficientnet.features[0][0].kernel_size,
            stride=self.efficientnet.features[0][0].stride,
            padding=self.efficientnet.features[0][0].padding,
            bias=False
        )

        # Replace the classifier for binary classification (2 output classes)
        self.efficientnet.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(self.efficientnet.classifier[1].in_features, 2),
        )

        # Placeholder for feature maps
        self.feature_maps = None
        self.hook_handle = None  # Store the handle for hook removal
        self.hook_layer = hook_layer

        # Register the hook dynamically
        self._register_hook()

    def _hook_fn(self, module, input, output):
        """Hook function to capture feature maps."""
        self.feature_maps = output  # Stores feature maps dynamically during the forward pass
        print(f"Captured feature map shape: {output.shape}")  # Debugging

    def _register_hook(self):
        """Registers the forward hook based on the chosen layer."""
        if self.hook_handle:
            self.hook_handle.remove()  # Remove previous hook if it exists

        # Identify correct layers for different resolutions
        layer_dict = {
            "7x7": self.efficientnet.features[-1],  # Deepest feature layer
            "14x14": self.efficientnet.features[-4],  # MBConv block
            "28x28": self.efficientnet.features[3],  # Early convolutional layer
        }

        if self.hook_layer not in layer_dict:
            raise ValueError("Invalid hook_layer. Choose '7x7', '14x14', or '28x28'.")

        target_layer = layer_dict[self.hook_layer]
        print(f"Hook registered at: {self.hook_layer}")  # Debugging
        self.hook_handle = target_layer.register_forward_hook(self._hook_fn)

    def forward(self, x):
        x = self.efficientnet(x)  # Forward pass through EfficientNet
        return x

    def _get_final_conv_weights(self, layer):
        """
        Extracts weights from the final projection layer inside MBConv blocks or Conv2D layers.
        """
        # If this is an MBConv block
        if isinstance(layer, nn.Sequential):
            for sub_layer in layer:
                if hasattr(sub_layer, "block"):  # Check if it's an MBConv block
                    block = sub_layer.block
                    if isinstance(block, nn.Sequential) and len(block) > 3:
                        if isinstance(block[3][0], nn.Conv2d):  # Final projection layer
                            return block[3][0].weight.data  # Extract final pointwise conv weights

        # If it's an early Conv2D layer (like features[1])
        if isinstance(layer[0], nn.Conv2d):
            return layer[0].weight.data

        raise ValueError(f"Could not find the final Conv2d layer inside: {layer}")

    def generate_cam(self, target_class, hook_layer=None, input_tensor=None):
        """
        Generate a Class Activation Map (CAM) for the specified target class.
        Ensures correct feature map dimensions.
        """
        if hook_layer and hook_layer != self.hook_layer:
            self.hook_layer = hook_layer
            self._register_hook()

            # Run a new forward pass to get updated feature maps
            if input_tensor is not None:
                _ = self.forward(input_tensor)

        if self.feature_maps is None:
            raise ValueError("Feature maps are not available. Ensure a forward pass is completed first.")

        # Ensure the feature maps and weights are on the same device as the model
        device = next(self.parameters()).device
        self.feature_maps = self.feature_maps.to(device)

        # Dynamically determine the appropriate weights based on feature map shape
        num_channels = self.feature_maps.shape[1]

        if num_channels == 1280:  # 7x7 layer (final output)
            fc_weights = self.efficientnet.classifier[1].weight.data.to(device)
        elif num_channels == 112:  # 14x14 layer (MBConv block)
            fc_weights =  self.efficientnet.features[-4][-1].block[-1][0].weight.data.to(device)
        elif num_channels == 40:  # 28x28 layer (MBConv block in `features[1]`)
            fc_weights = self.efficientnet.features[3][-1].block[-1][0].weight.data.to(device)
        else:
            raise ValueError(
                f"Feature maps have {num_channels} channels, and no corresponding weights were found."
            )

        num_fc_weights = fc_weights.shape[1]
        # print(fc_weights.shape)

        if num_channels != num_fc_weights:
            if num_channels > num_fc_weights:
                print(f"Warning: Feature maps have {num_channels} channels, but extracted weights expect {num_fc_weights}.")
            fc_weights = fc_weights[:, :num_channels]  # Trim weights to match feature map channels

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

