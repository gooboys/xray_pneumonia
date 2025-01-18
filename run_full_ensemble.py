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

# Check if a GPU is available and set the device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def show_menu():
    print("===========================================")
    print("    Basic pneumonia disease state model    ")
    print("By: Thierry Juang, James Couch, Saha Udassi")
    print("===========================================")
    print("1. Run model prediction on CT Scan")
    print("2. List all models incorporated")
    print("3. View model structure")
    print("4. View model performance reports")
    print("5. Exit the program")
    print("===========================================")
    return

transform = transforms.ToTensor()

def run_it(ensemble):
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

'''
pneumonia_present_models = [] # list of models predicting if pneumonia is present
pneumonia_present_model_type = eff3 # type of model predicting pneumonia

pneumonia_type_models = {
    eff3: ['model_1.pth','model_2.pth']
    eff4: ['model_3.pth','model_4.pth']
}
'''
# Runs the model with an image from image path, also takes in a model if it is not a string.
def run_model(pneumonia_present_models, pneumonia_present_model_type, pneumonia_type_models, image_path, transform, device, showCAM):

    image = Image.open(image_path).convert("L")
    input_tensor = transform(image).unsqueeze(0).to(device)

    actual_predicted_class = 0
   
    heatmaps = []
    overlays = []
    ensemble_outputs = []
    df = pd.read_csv('Normalized_Image_Paths.csv')
    label = int(df.loc[df['Paths'] == image_path, 'Labels'].values[0])
    for path in pneumonia_present_models:
        model = pneumonia_present_model_type().to(device)
        model.load_state_dict(torch.load(path, weights_only=True))
        model.eval()
        with torch.no_grad():
            outputs = model(input_tensor)
            ensemble_outputs.append(F.softmax(outputs, dim=1).cpu().numpy())
        overlay, heatmap = model.generate_cam(model, image_path, label, make_graphs=showCAM)
        overlays.append(overlay)
        heatmaps.append(heatmap)
   
    # Average predictions
    ensemble_outputs = np.mean(ensemble_outputs, axis=0)
    predicted_class = np.argmax(ensemble_outputs)
    
    if predicted_class == 1:
        ensemble_outputs2 = []
        for model_class in pneumonia_type_models.keys():
            for model_path in pneumonia_type_models.get(model_class):
                model = model_class().to(device)
                model.load_state_dict(torch.load(model_path, weights_only=True))
                model.eval()
                with torch.no_grad():
                    outputs = model(input_tensor)
                    ensemble_outputs2.append(F.softmax(outputs, dim=1).cpu().numpy())
                overlay, heatmap = model.generate_cam(model, image_path, label, make_graphs=showCAM)
                overlays.append(overlay)
                heatmaps.append(heatmap)
        # Average predictions
        ensemble_outputs2 = np.mean(ensemble_outputs2, axis=0)
        predicted_class = np.argmax(ensemble_outputs2)
        if predicted_class == 0:
            actual_predicted_class = 1
        else:
            actual_predicted_class = 2
    actual_class_mapping = {0: "normal", 1: "bacterial", 2: "viral"}
    print(f"Predicted Class (Ensemble): {actual_class_mapping.get(actual_predicted_class, 'Unknown')}")
    print(f"Actual Class: {actual_class_mapping.get(label, 'Unknown')}")
   
    # Convert all heatmaps to NumPy arrays and ensure they are 2D
    heatmap_arrays = [np.squeeze(np.array(heatmap)) for heatmap in heatmaps]

    # Check if the resulting arrays are 2D
    for i, heatmap in enumerate(heatmap_arrays):
        if len(heatmap.shape) != 2:
            raise ValueError(f"Heatmap at index {i} is not 2D. Shape: {heatmap.shape}")

    # Stack the heatmaps along a new axis and compute the mean
    mean_heatmap = np.mean(np.stack(heatmap_arrays, axis=0), axis=0)

    # Normalize the mean heatmap to [0, 255]
    mean_heatmap -= mean_heatmap.min()
    mean_heatmap /= mean_heatmap.max()
    mean_heatmap = (mean_heatmap * 255).astype('uint8')
   
    if predicted_class == 0:
        class_name = "Normal"
    elif predicted_class == 1:
        class_name = "Bacterial"
    else:
        class_name = "Viral"
       
    if showCAM == True:
        # Display both the heatmap and overlay
        plt.figure(figsize=(12, 6))
   
        # Display Heatmap
        plt.subplot(1, 2, 1)
        plt.imshow(mean_heatmap)
        plt.title(f'Class Activation Map Heatmap for {class_name} Class')
        plt.axis("off")
   
        # Display Overlay
        plt.subplot(1, 2, 2)
        plt.imshow(overlay)
        plt.title(f'Overlay for {class_name} Class')
        plt.axis("off")
   
        # Adjust layout to prevent title cut off
        plt.subplots_adjust(top=0.85, bottom=0.1, left=0.1, right=0.9)
   
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    run = True
    while run:
        show_menu()
        ensemble = ["model_split_0_best.pth","model_split_1_best.pth","model_split_2_best.pth","model_split_3_best.pth","model_split_4_best.pth"]
        choice = input("Enter your choice (1-5): ").strip()
        if choice == "1":
            run_it(ensemble)
        if choice == "2":
            print("To be listed")
        if choice == "3":
            print("List model options:")
            model_number = input("What model would you like to see?")
            if model_number == "1":
                print('the structure')
            if model_number == "2":
                print('the structure')
            if model_number == "3":
                print('the structure')
        if choice == "4":
            print("List model performance options:")
            model_number = input("What model performance would you like to see?")
            if model_number == "1":
                print('the structure')
            if model_number == "2":
                print('the structure')
            if model_number == "3":
                print('the structure')
        if choice == "5":
            print("Thanks for using our model!")
            run = False