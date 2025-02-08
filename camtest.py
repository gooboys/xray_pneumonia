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

# Finds the label of the sample in question from the CSV file containing paths and labels
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

# This function runs the function to generate a CAM model. These are inbuilt functions exclusive to the 
# models stored in the camModels folder
def camtest(modelType, image_path, model_path, resolution):
    # Taking the image and turning it into a tensor for the model
    image = Image.open(image_path).convert("L")
    input_tensor = transform(image).unsqueeze(0).to(device)

    # Declares the model, uses the tag 'resolution' to determine which layer the CAM will be generated from
    model = modelType(dropout_rate=0.5,hook_layer=resolution).to(device)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()

    # Getting the class label to run generate_cam function
    imageClass = get_label('Normalized_Image_Paths.csv', image_path)
    print(imageClass)

    # Running the tensor through the model
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
    model_path1 = 'TrainedModels/denseA31.pth'
    model_path2 = 'TrainedModels/denseB51.pth'
    model_path3 = 'TrainedModels/eff41.pth'
    # inputs for resolution to get different sized CAM maps: '7x7' '14x14' '28x28'
    resOne = '7x7'
    resTwo = '14x14'
    resolution = '28x28'

    # Tests all three infection type models on a single image for all different resolutions.
    # No interpolation or image smoothing is done here
    camtest(denseA3, image_path, model_path1, resOne)
    camtest(denseA3, image_path, model_path1, resTwo)
    camtest(denseA3, image_path, model_path1, resolution)
    camtest(denseB5, image_path, model_path2, resOne)
    camtest(denseB5, image_path, model_path2, resTwo)
    camtest(denseB5, image_path, model_path2, resolution)
    camtest(eff4, image_path, model_path3, resOne)
    camtest(eff4, image_path, model_path3, resTwo)
    camtest(eff4, image_path, model_path3, resolution)