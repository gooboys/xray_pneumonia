import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split as sk_train_test_split
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
from camModels import denseA3, denseB5

transform = transforms.ToTensor()  # Convert images to PyTorch tensors
# Check if a GPU is available and set the device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def camtest(modelType, image_path, model_path, resolution):
    image = Image.open(image_path).convert("L")
    input_tensor = transform(image).unsqueeze(0).to(device)

    model = modelType(dropout_rate=0.5,hook_layer=resolution).to(device)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()

    logits = model(input_tensor)
    cam = model.generate_cam(1, resolution)

    # Ensure tensors are moved to CPU before visualization
    plt.figure(figsize=(10, 10))
    plt.imshow(input_tensor.squeeze().cpu().numpy(), cmap='gray', alpha=0.5)  # Original grayscale image
    plt.imshow(cam, cmap='jet', alpha=0.5)  # Overlay CAM
    plt.colorbar()
    plt.show()



if __name__ == "__main__":

    image_path = 'NormalizedXRays/image_3.jpeg'
    model_path1 = 'testModel/denseA3.pth'
    model_path2 = 'TrainedModels/denseB51.pth'
    model_path3 = 'TrainedModels/eff31.pth'
    model_path4 = 'TrainedModels/eff41.pth'
    # inputs for resolution to get different sized CAM maps: '7x7' '14x14' '28x28'
    resolution = '28x28'

    camtest(denseA3, image_path, model_path4, resolution)