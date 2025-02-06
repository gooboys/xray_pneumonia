import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split as sk_train_test_split
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
from PIL import ImageEnhance
from camModels import denseA3, denseB5, eff4
from models import standardCNN
import cv2

# Check if a GPU is available and set the device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def show_menu():
    print("===========================================")
    print("    Basic pneumonia disease state model    ")
    print("            By: Thierry Juang")
    print("===========================================")
    print("1. Run model prediction on CT Scan")
    print("2. List all models incorporated")
    print("3. View model structure")
    print("4. View model performance reports")
    print("5. Exit the program")
    print("===========================================")
    return

transform = transforms.ToTensor()

def run_it(model_paths, model_types):
    image_path = input("What is the number of the image you would like to analyze? ")
    cont1 = True
    while cont1:
        if image_path.isdigit():
            if int(image_path)<5856:  
                cont1 = False
            else:
                print("That doesn't seem to be an integer or an image we posses")
                image_path = input("What is the number of the image you would like to analyze? ")
        elif not image_path.isdigit():
            print("That doesn't seem to be an integer or an image we posses")
            image_path = input("What is the number of the image you would like to analyze? ")        
    image_path = "NormalizedXRays/image_" + image_path + ".jpeg"
    cont = True
    printer = False
    print_heatmaps = input("Would you like to see the class activation map for this image? (Y/N) ")
    while cont:
        if print_heatmaps == "Y" or print_heatmaps == "y":
            printer = True
            cont = False
        elif print_heatmaps == "N" or print_heatmaps == "n":
            printer = False
            cont = False
        else:
            print("That was not a valid input, please try again")
            print_heatmaps = input("Would you like to see the class activation map for this image? (Y/N) ")
    # run_model(CNN, ensemble, image_path, transform, device, printer)
    return

def generate_cam(model, image_path, class_idx, make_graphs=True):
    """
    Generates and visualizes a Class Activation Map (CAM) for a specific class index.

    Args:
        model (nn.Module): The trained CNN model.
        image_path (str): Path to the input image.
        class_idx (int): The class index for which to generate CAM.

    Returns:
        PIL.Image: Image with the CAM overlay.
    """
    # Load and preprocess the image
    original_image = Image.open(image_path).convert("L")  # Load as grayscale
   
    # Apply transformations and convert to tensor
    input_image = transform(original_image).unsqueeze(0).to(device)  # Add batch dimension and move to GPU

    # Forward pass to get feature maps and logits
    model.eval()
    with torch.no_grad():
        features = model.pool3(F.relu(model.conv3(
                    model.pool2(F.relu(model.conv2(
                        model.pool1(F.relu(model.conv1(input_image))))
                    ))
        )))
        logits = model.fc2(F.relu(model.fc1(features.view(features.size(0), -1))))  # Same as your previous forward pass

    # Get the weights for the target class from the final fully connected layer
    fc_weights = model.fc2.weight[class_idx].detach()

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

def camtest(modelType, image_path, model_path, imageClass, resolution):
    image = Image.open(image_path).convert("L")
    input_tensor = transform(image).unsqueeze(0).to(device)

    model = modelType(dropout_rate=0.5,hook_layer=resolution).to(device)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()

    logits = model(input_tensor)
    cam = model.generate_cam(imageClass, resolution)

    return cam

def run_ensemble(model_paths, model_types, image_path, CAM, resolution):
    image = Image.open(image_path).convert("L")
    input_tensor = transform(image).unsqueeze(0).to(device)
   
    heatmaps = []
    ensemble_outputs = []
    df = pd.read_csv('Normalized_Image_Paths.csv')
    label = int(df.loc[df['Paths'] == image_path, 'Labels'].values[0])
    for path in model_paths['infection_present_models']:
        model_class = model_types[path]
        model = model_class().to(device)
        model.load_state_dict(torch.load(path, weights_only=True))
        model.eval()
        with torch.no_grad():
            outputs = model(input_tensor)
            ensemble_outputs.append(F.softmax(outputs, dim=1).cpu().numpy())
        overlay, heatmap = generate_cam(model, image_path, label)
        heatmaps.append(heatmap)

    ensemble_outputs = np.mean(ensemble_outputs, axis=0)
    predicted_class = np.argmax(ensemble_outputs)
    
    if predicted_class == 0:
        class_mapping = {0: "normal", 1: "bacterial", 2: "viral"}
        print("Predicted Class (Ensemble): normal")
        print(f"Actual Class: {class_mapping.get(label, 'Unknown')}")
        if CAM:
            showCAM(heatmaps, [])
        return
    
    # Classifying if not no-infection
    type_heatmaps = []
    ensemble_outputs_type = []
    for weights_path in model_paths['infection_type_models']:
        modeltype = model_types[weights_path]
        model = modeltype().to(device)
        model.load_state_dict(torch.load(weights_path, weights_only=True))
        model.eval()
        with torch.no_grad():
            outputs = model(input_tensor)
            ensemble_outputs_type.append(F.softmax(outputs, dim=1).cpu().numpy())
        type_heatmaps.append(camtest(modeltype, image_path, weights_path, label-1, resolution))

    avg_outputs_type = np.mean(ensemble_outputs_type, axis=0)

    # Switch between argmax and custom threshhold, comment out one
    # type_predictions = np.argmax(avg_outputs_type, axis=1)
    print(avg_outputs_type.shape)
    type_predictions = (avg_outputs_type[:, 1] > 0.54).astype("int32")
    print(type_predictions.shape)
    
    final_prediction = int(type_predictions[0]) + 1  # Assign correct class labels

    class_mapping = {0: "normal", 1: "bacterial", 2: "viral"}
    print(f"Predicted Class (Ensemble): {class_mapping.get(final_prediction, 'Unknown')}")
    print(f"Actual Class: {class_mapping.get(label, 'Unknown')}")
    if CAM:
        showCAM(heatmaps, type_heatmaps, image_path)
    return


def showCAM(disease_present_CAMS, disease_type_CAMS, image_path):
    # Load the original image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB format
    
    # Compute mean heatmaps
    mean_present_CAM = np.mean(disease_present_CAMS, axis=0)
    mean_type_CAM = np.mean(disease_type_CAMS, axis=0)
    
    # Normalize heatmaps to [0, 1] for proper visualization
    mean_present_CAM = (mean_present_CAM - np.min(mean_present_CAM)) / (np.max(mean_present_CAM) - np.min(mean_present_CAM) + 1e-8)
    mean_type_CAM = (mean_type_CAM - np.min(mean_type_CAM)) / (np.max(mean_type_CAM) - np.min(mean_type_CAM) + 1e-8)

    # Resize heatmaps to match image dimensions
    heatmap_size = (image.shape[1], image.shape[0])
    mean_present_CAM = cv2.resize(mean_present_CAM, heatmap_size)
    mean_type_CAM = cv2.resize(mean_type_CAM, heatmap_size)

    # Overlay heatmaps onto the image
    heatmap_present = cv2.applyColorMap((mean_present_CAM * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap_type = cv2.applyColorMap((mean_type_CAM * 255).astype(np.uint8), cv2.COLORMAP_JET)

    overlay_present = cv2.addWeighted(image, 0.5, heatmap_present, 0.5, 0)
    overlay_type = cv2.addWeighted(image, 0.5, heatmap_type, 0.5, 0)

    # Plot results
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(image)
    axes[0].set_title("Original Image")
    axes[0].axis("off")
    
    axes[1].imshow(overlay_present)
    axes[1].set_title("Mean Disease Presence CAM")
    axes[1].axis("off")

    axes[2].imshow(overlay_type)
    axes[2].set_title("Mean Disease Type CAM")
    axes[2].axis("off")

    plt.show()
    return

if __name__ == "__main__":
    run = True
    model_paths = {
        'infection_present_models': [
            'TrainedModels/standard1.pth',
            'TrainedModels/standard2.pth',
            'TrainedModels/standard3.pth',
            'TrainedModels/standard4.pth',
            'TrainedModels/standard5.pth'
            ],
        'infection_type_models': [
            'TrainedModels/denseA31.pth','TrainedModels/denseA32.pth','TrainedModels/denseA33.pth','TrainedModels/denseA34.pth',
            'TrainedModels/eff41.pth','TrainedModels/eff42.pth','TrainedModels/eff43.pth','TrainedModels/eff44.pth',
            'TrainedModels/denseB51.pth','TrainedModels/denseB52.pth','TrainedModels/denseB53.pth','TrainedModels/denseB54.pth'
        ]
    }
    model_types = {
        'TrainedModels/standard1.pth': standardCNN,'TrainedModels/standard2.pth': standardCNN,'TrainedModels/standard3.pth': standardCNN,'TrainedModels/standard4.pth': standardCNN,'TrainedModels/standard5.pth': standardCNN,
        'TrainedModels/denseA31.pth': denseA3,'TrainedModels/denseA32.pth': denseA3,'TrainedModels/denseA33.pth': denseA3,'TrainedModels/denseA34.pth': denseA3,
        'TrainedModels/eff41.pth': eff4,'TrainedModels/eff42.pth': eff4,'TrainedModels/eff43.pth': eff4,'TrainedModels/eff44.pth': eff4,
        'TrainedModels/denseB51.pth': denseB5,'TrainedModels/denseB52.pth': denseB5,'TrainedModels/denseB53.pth': denseB5,'TrainedModels/denseB54.pth': denseB5
    }
    run_ensemble(model_paths, model_types, 'NormalizedXRays/image_0.jpeg', True, '14x14')
    # while run:
    #     show_menu()
    #     choice = input("Enter your choice (1-5): ").strip()
    #     if choice == "1":
    #         run_it(model_paths, model_types)
    #     if choice == "2":
    #         print("To be listed")
    #     if choice == "3":
    #         print("List model options:")
    #         model_number = input("What model would you like to see?")
    #         if model_number == "1":
    #             print('the structure')
    #         if model_number == "2":
    #             print('the structure')
    #         if model_number == "3":
    #             print('the structure')
    #     if choice == "4":
    #         print("List model performance options:")
    #         model_number = input("What model performance would you like to see?")
    #         if model_number == "1":
    #             print('the structure')
    #         if model_number == "2":
    #             print('the structure')
    #         if model_number == "3":
    #             print('the structure')
    #     if choice == "5":
    #         print("Thanks for using our model!")
    #         run = False