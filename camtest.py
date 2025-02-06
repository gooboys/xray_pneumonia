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
from camModels import denseA3, denseB5, eff4
import csv

def get_label(csv_file, target_path):
    with open(csv_file, newline='', encoding='utf-8') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header
        for row in reader:
            if row[0] == target_path:
                return int(row[1])
    return None  # Return None if the path is not found

transform = transforms.ToTensor()  # Convert images to PyTorch tensors
# Check if a GPU is available and set the device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def camtest(modelType, image_path, model_path, resolution):
    image = Image.open(image_path).convert("L")
    input_tensor = transform(image).unsqueeze(0).to(device)

    model = modelType(dropout_rate=0.5,hook_layer=resolution).to(device)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()

    imageClass = get_label('Normalized_Image_Paths.csv', image_path)
    print(imageClass)

    logits = model(input_tensor)
    cam = model.generate_cam(imageClass, resolution)

    # Ensure tensors are moved to CPU before visualization
    plt.figure(figsize=(10, 10))
    plt.imshow(input_tensor.squeeze().cpu().numpy(), cmap='gray', alpha=0.5)  # Original grayscale image
    plt.imshow(cam, cmap='jet', alpha=0.5)  # Overlay CAM
    plt.colorbar()
    plt.show()



if __name__ == "__main__":

    image_path = 'NormalizedXRays/image_6.jpeg'
    model_path1 = 'testModel/denseA3.pth'
    model_path2 = 'TrainedModels/denseB51.pth'
    # model_path3 = 'TrainedModels/eff31.pth'
    model_path4 = 'TrainedModels/eff41.pth'
    # inputs for resolution to get different sized CAM maps: '7x7' '14x14' '28x28'
    resolution = '28x28'

    camtest(eff4, image_path, model_path4, '7x7')
    camtest(eff4, image_path, model_path4, '14x14')
    camtest(eff4, image_path, model_path4, resolution)