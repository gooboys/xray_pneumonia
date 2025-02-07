import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import ImageEnhance
from camModels import denseA3, denseB5, eff4
from models import standardCNN
import cv2
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Check if a GPU is available and set the device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = transforms.ToTensor()

heatmap_prompt = """
Please select the heatmap size you would like to use:

1)  7×7  → Very High Abstraction (Final Layer)
    - Highlights global discriminative areas
    - Best for explaining model decisions

2)  14×14 → Medium Abstraction (Mid-Level Features)
    - Highlights textures, shapes, and regions
    - Best for understanding feature localization

3)  28×28 → Low Abstraction (Early Feature Extraction)
    - Highlights edges, fine details, and textures
    - Best for debugging feature extraction

Enter the corresponding number (1, 2, or 3) to select an option:
"""
models_listed = """
Model Structure:
----------------
This is an ensemble model performing two stages of binary classification.

1. **Bagging Stage:**
   - A single deep learning model performs the first binary classification.
   - This model aggregates multiple predictions to improve robustness.
   - **Model Used:** Custom Deep Learning Model.
   - This model distinguishes between infection and no infection cases.

2. **Transfer Learning Stage:**
   - Three different types of transfer learning models are used.
   - Each type consists of 4 separate models, resulting in a total of 12 models.
   - **Models Used:**
     - DenseNet-169 (4 models)
     - DenseNet-121 (4 models)
     - EfficientNet-B0 (4 models)
   - The final prediction is obtained by combining the outputs of all 12 models.
   - This model distinguishes between viral and bacterial infection cases.

Overall, this ensemble approach leverages deep learning for bagging, followed by 
multiple transfer learning models to refine classification performance.
"""
model_structure_options = """
Select a Model Structure to View:
---------------------------------
1 - Deep Learning Model (Bagging Stage)
2 - DenseNet-169 (Transfer Learning)
3 - DenseNet-121 (Transfer Learning)
4 - EfficientNet-B0 (Transfer Learning)

Enter the number corresponding to the model structure you would like to see.
"""
cnn_structure = """
Deep Learning Model Structure:
-----------------------------
Input: Grayscale image (1 channel)

1. **Convolutional & Pooling Layers**
   - Conv1: 32 filters, 3x3 kernel, ReLU activation
   - MaxPool1: 2x2 kernel, stride 2
   - Conv2: 64 filters, 3x3 kernel, ReLU activation
   - MaxPool2: 2x2 kernel, stride 2
   - Conv3: 128 filters, 3x3 kernel, ReLU activation
   - MaxPool3: 2x2 kernel, stride 2

2. **Fully Connected Layers**
   - Flatten layer (reshapes feature map to vector)
   - FC1: 128 neurons, ReLU activation
   - Dropout (p=0.5)
   - FC2: 2 output neurons (for binary classification)

Forward pass follows Conv -> ReLU -> Pooling -> Fully Connected Layers -> Output.
"""
denseB5_structure = """
DenseNet-169 Transfer Learning Model Structure:
------------------------------------------------
1. **Feature Extraction (DenseNet-169 Backbone)**
   - Uses a pre-trained DenseNet-169 model.
   - First convolutional layer modified to accept grayscale (1-channel) images.
   - Extracts deep hierarchical features through multiple DenseBlocks.

2. **Classifier Modification**
   - Original classifier replaced with:
     - Linear layer (1664 -> 128)
     - Dropout layer (p=0.5)
     - Fully connected layer (128 -> 2) for binary classification.

Overall, this model integrates DenseNet-169 with custom modifications for grayscale input 
and binary classification.
"""
denseA3_structure = """
DenseNet-121 Transfer Learning Model Structure:
------------------------------------------------
1. **Feature Extraction (DenseNet-121 Backbone)**
   - Uses a pre-trained DenseNet-121 model.
   - First convolutional layer modified to accept grayscale (1-channel) images.
   - Extracts deep hierarchical features through multiple DenseBlocks.

2. **Classifier Modification**
   - Original classifier replaced with:
     - Linear layer (1024 -> 256)
     - Dropout layer (p=0.5)
     - Linear layer (256 -> 128), ReLU activation
     - Dropout layer (p=0.5)
     - Fully connected layer (128 -> 2) for binary classification.

Overall, this model integrates DenseNet-121 with custom modifications for grayscale input 
and binary classification.
"""
eff4_structure = """
EfficientNet-B0 Transfer Learning Model StructureEfficientNet-B0:
------------------------------------------------
1. **Feature Extraction (EfficientNet-B0 Backbone)**
   - Uses a pre-trained EfficientNet-B0 model.
   - First convolutional layer modified to accept grayscale (1-channel) images.
   - Extracts deep hierarchical features through Mobile Inverted Bottleneck (MBConv) blocks.

2. **Classifier Modification**
   - Original classifier replaced with:
     - Dropout layer (p=0.5)
     - Fully connected layer (1280 -> 2) for binary classification.

Overall, this model integrates EfficientNet-B0 with custom modifications for grayscale input 
and binary classification.
"""
press_to_continue = "Input anything to continue..."

list_model_options = """
List of Model Performance Options:
----------------------------------
1 - DenseNet-121 Transfer Learning Model
2 - DenseNet-169 Transfer Learning Model
3 - EfficientNet-B0 Transfer Learning Model
4 - Ensembled Transfer Learning Model (DenseNet-121, DenseNet-169, EfficientNet-B0)
5 - Disease Presence Classification Model
6 - Full Ensembled Pneumonia Classification Model (Normal vs. Bacterial vs. Viral)

Enter the number corresponding to the model performance report you would like to view.
"""
denseA3_performance = """
DenseNet-121 Transfer Learning Model Report
----------------------------------------------
Dataset: Pneumonia Classification (Bacterial vs. Viral)

Confusion Matrix (Threshold = 0.53):
-----------------------------------
       Predicted
         0    1
      -----------
    0 | 112   37  (Bacterial)
    1 |  30  120  (Viral)

Classification Report:
----------------------
               Precision    Recall  F1-Score   Support
------------------------------------------------------
Bacterial (0)    0.7887     0.7517    0.7698       149
Viral (1)        0.7643     0.8000    0.7818       150

Overall Model Performance:
--------------------------
- Accuracy: **77.59%**
- ROC-AUC Score: **84.16%**
- Macro Avg: Precision = 0.7765, Recall = 0.7758, F1-Score = 0.7758
- Weighted Avg: Precision = 0.7765, Recall = 0.7759, F1-Score = 0.7758

Thresholds & Decision Boundary:
-------------------------------
- Default classification threshold: **0.5**
- Predictions greater than **0.53** are classified as **Viral (1)**
- Predictions less than or equal to **0.53** are classified as **Bacterial (0)**

Summary:
--------
The EfficientNet-B0 transfer learning model achieves a balanced classification 
between bacterial and viral pneumonia, with an accuracy of **77.59%** and an 
**ROC-AUC score of 84.16%**, indicating strong overall discrimination capability.

The bacterial class exhibits slightly higher precision (**78.87%**), reducing 
false positives, while the viral class achieves higher recall (**80.00%**), 
capturing more true cases. The model maintains a well-calibrated balance 
between both classes.

Using a **threshold of 0.5**, the model classifies cases effectively, but 
adjusting the threshold could optimize performance further based on specific 
application needs.
"""
ensemble_report = """
Ensembled Transfer Learning Model Report
----------------------------------------
Models Used: DenseNet-121, DenseNet-169, EfficientNet-B0
Dataset: Pneumonia Classification (Bacterial vs. Viral)

Confusion Matrix (Threshold = 0.52):
-----------------------------------
       Predicted
         0    1
      -----------
    0 | 120   29  (Bacterial)
    1 |  30  120  (Viral)

Classification Report:
----------------------
               Precision    Recall  F1-Score   Support
------------------------------------------------------
Bacterial (0)    0.8000     0.8054    0.8027       149
Viral (1)        0.8054     0.8000    0.8027       150

Overall Model Performance:
--------------------------
- Accuracy: **80.27%**
- ROC-AUC Score: **86.63%**
- Macro Avg: Precision = 0.8027, Recall = 0.8027, F1-Score = 0.8027
- Weighted Avg: Precision = 0.8027, Recall = 0.8027, F1-Score = 0.8027

Thresholds & Decision Boundary:
-------------------------------
- Default classification threshold: **0.5**
- Predictions greater than **0.52** are classified as **Viral (1)**
- Predictions less than or equal to **0.52** are classified as **Bacterial (0)**

Summary:
--------
The ensembled model combines **DenseNet-121, DenseNet-169, and EfficientNet-B0**, leveraging 
their individual strengths to enhance classification performance. It achieves a balanced 
classification between bacterial and viral pneumonia, with an accuracy of **80.27%** and an 
**ROC-AUC score of 86.63%**, indicating strong overall discrimination capability.

Both bacterial and viral classes exhibit near-identical precision, recall, and F1-scores, 
suggesting a well-calibrated model. The threshold of **0.5** was used for classification, 
but adjustments could be explored for further optimization.

Ensembling these transfer learning models improves overall robustness and generalization 
compared to individual models.
"""
denseB5_report = """
DenseNet-169 Transfer Learning Model Report
-------------------------------------------
Dataset: Pneumonia Classification (Bacterial vs. Viral)

Confusion Matrix (Threshold = 0.52):
-----------------------------------
       Predicted
         0    1
      -----------
    0 | 109   40  (Bacterial)
    1 |  28  122  (Viral)

Classification Report:
----------------------
               Precision    Recall  F1-Score   Support
------------------------------------------------------
Bacterial (0)    0.7956     0.7315    0.7622       149
Viral (1)        0.7531     0.8133    0.7821       150

Overall Model Performance:
--------------------------
- Accuracy: **77.26%**
- ROC-AUC Score: **85.45%**
- Macro Avg: Precision = 0.7744, Recall = 0.7724, F1-Score = 0.7721
- Weighted Avg: Precision = 0.7743, Recall = 0.7726, F1-Score = 0.7722

Thresholds & Decision Boundary:
-------------------------------
- Default classification threshold: **0.5**
- Predictions greater than **0.52** are classified as **Viral (1)**
- Predictions less than or equal to **0.52** are classified as **Bacterial (0)**

Summary:
--------
The DenseNet-169 transfer learning model achieves **77.26% accuracy** and an 
**ROC-AUC score of 85.45%**, showing strong overall performance in distinguishing 
bacterial and viral pneumonia.

The bacterial class has a slightly higher precision (**79.56%**), minimizing false 
positives, while the viral class has a higher recall (**81.33%**), ensuring more 
cases are correctly identified. This suggests the model is slightly more cautious 
when predicting viral cases.

Using a **threshold of 0.5**, the model effectively classifies cases, though 
threshold tuning could further optimize performance for specific applications.
"""
eff4_report = """
EfficientNet-B0 Transfer Learning Model Report
----------------------------------------------
Dataset: Pneumonia Classification (Bacterial vs. Viral)

Confusion Matrix (Threshold = 0.52):
------------------------------------
       Predicted
         0    1
      -----------
    0 | 120   29  (Bacterial)
    1 |  35  115  (Viral)

Classification Report:
----------------------
               Precision    Recall  F1-Score   Support
------------------------------------------------------
Bacterial (0)    0.7742     0.8054    0.7895       149
Viral (1)        0.7986     0.7667    0.7823       150

Overall Model Performance:
--------------------------
- Accuracy: **78.60%**
- ROC-AUC Score: **85.93%**
- Macro Avg: Precision = 0.7864, Recall = 0.7860, F1-Score = 0.7859
- Weighted Avg: Precision = 0.7864, Recall = 0.7860, F1-Score = 0.7859

Thresholds & Decision Boundary:
-------------------------------
- Default classification threshold: **0.50**
- Adjusted classification threshold: **0.52**
- Predictions greater than **0.52** are classified as **Viral (1)**
- Predictions less than or equal to **0.52** are classified as **Bacterial (0)**

Summary:
--------
The EfficientNet-B0 transfer learning model achieves **78.60% accuracy** and an 
**ROC-AUC score of 85.93%**, showing strong overall discrimination capability.

Compared to the default **0.50 threshold**, an adjusted threshold of **0.52** was 
used, slightly increasing the model's bias toward bacterial classification. This 
change resulted in a higher precision for the viral class (**79.86%**) but a slightly 
lower recall (**76.67%**), indicating fewer false positives at the cost of missing 
some viral cases.

This suggests that **threshold tuning** can help optimize the model’s performance 
depending on the specific clinical or research needs.
"""
standardCNN_report = """
Disease Presence Classification Model Report
--------------------------------------------
Dataset: Pneumonia Detection (Disease vs. No Disease)

Confusion Matrix (Threshold = 0.5):
-----------------------------------
       Predicted
         0    1
      -----------
    0 | 142    7  (No Disease)
    1 |   5  144  (Disease Present)

Classification Report:
----------------------
               Precision    Recall  F1-Score   Support
------------------------------------------------------
No Disease (0)   0.9660     0.9530    0.9595       149
Disease (1)      0.9536     0.9664    0.9600       149

Overall Model Performance:
--------------------------
- Accuracy: **95.97%**
- ROC-AUC Score: **99.04%**
- Macro Avg: Precision = 0.9598, Recall = 0.9597, F1-Score = 0.9597
- Weighted Avg: Precision = 0.9598, Recall = 0.9597, F1-Score = 0.9597

Thresholds & Decision Boundary:
-------------------------------
- Predictions greater than **0.50** are classified as **Disease Present (1)**
- Predictions less than or equal to **0.50** are classified as **No Disease (0)**

Summary:
--------
The disease presence classification model demonstrates **exceptionally high accuracy (95.97%)** 
and a **ROC-AUC score of 99.04%**, indicating strong predictive capability. 

Both classes exhibit high precision and recall, with **minimal false positives and false negatives**. 
The slightly higher recall for the **Disease Present** class (**96.64%**) suggests the model prioritizes 
correctly identifying diseased cases, making it well-suited for medical screening applications.

Given the **ROC-AUC of 99.04%**, this model provides highly reliable discrimination between diseased 
and non-diseased cases, though further threshold tuning could refine performance for specific use cases.
"""
fullEnsemble_report = """
Full Pneumonia Classification Model Report
------------------------------------------
Dataset: Pneumonia Diagnosis (Normal vs. Bacterial vs. Viral)

Confusion Matrix:
-----------------
       Predicted
         NORMAL  BACTERIA  VIRUS
      --------------------------------
  NORMAL   |  142      1       6
BACTERIA   |    1    123      25
   VIRUS   |    6     33     111

Classification Report:
----------------------
              Precision    Recall  F1-Score   Support
------------------------------------------------------
     NORMAL     0.9530     0.9530    0.9530       149
   BACTERIA     0.7834     0.8255    0.8039       149
      VIRUS     0.7817     0.7400    0.7603       150

Overall Model Performance:
--------------------------
- Accuracy: **83.93%**
- Macro Avg: Precision = 0.8394, Recall = 0.8395, F1-Score = 0.8391
- Weighted Avg: Precision = 0.8393, Recall = 0.8393, F1-Score = 0.8389

Thresholds & Decision Boundary:
-------------------------------
- **Infection Presence Classification (Normal vs. Infected)**: Threshold = **0.50**
  - Predictions greater than **0.50** are classified as **Infected (Bacterial or Viral).**
  - Predictions less than or equal to **0.50** are classified as **Normal.**
  
- **Disease Type Classification (Bacterial vs. Viral)**: Threshold = **0.54**
  - If classified as **Infected**, the model uses a second threshold of **0.54**.
  - The class with the highest confidence score above **0.54** is selected.

Summary:
--------
This pneumonia classification model operates in **two stages**:
1. **Infection Presence Classification (Threshold = 0.50)**:
   - Determines whether a patient has pneumonia (Bacterial/Viral) or is Normal.
2. **Disease Type Classification (Threshold = 0.54)**:
   - If pneumonia is detected, the model classifies it as **Bacterial or Viral**.

- **Normal cases** are identified with **95.30% precision and recall**, ensuring high confidence in non-infected predictions.
- **Bacterial cases** show an **82.55% recall**, meaning most bacterial infections are correctly classified.
- **Viral cases** have a slightly lower **74.00% recall**, suggesting some viral cases are misclassified as bacterial.

By using **two different thresholds**, this model optimizes classification performance while maintaining a balance 
between precision and recall. Further fine-tuning of these thresholds may improve classification accuracy depending 
on clinical needs.
"""

def show_menu():
    print("===========================================")
    print("    Basic pneumonia disease state model    ")
    print("            By: Thierry Juang")
    print("===========================================")
    print("1. Run model prediction on CT Scan")
    print("2. List all models incorporated")
    print("3. View model structures")
    print("4. View model performance reports")
    print("5. Exit the program")
    print("===========================================")
    return

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
    if printer:
        while True:
            res = input(heatmap_prompt)
            if res == '1' or res == '2' or res == '3':
                break
            else:
                print('That was not a valid input, please try again')
    options = {
        '1': '7x7',
        '2': '14x14',
        '3': '28x28'
    }
    resolution = options[res]
    run_ensemble(model_paths, model_types, image_path, printer, resolution)
    return

def generate_cam(model, image_path, class_idx):
    """
    Generates and visualizes a Class Activation Map (CAM) for a specific class index.

    Args:
        model (nn.Module): The trained CNN model.
        image_path (str): Path to the input image.
        class_idx (int, optional): The class index for which to generate CAM.

    Returns:
        PIL.Image: Image with the CAM overlay.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    # Load and preprocess the image
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.ToTensor(),
    ])
    original_image = Image.open(image_path).convert("L")  # Convert to grayscale
    input_tensor = transform(original_image).unsqueeze(0).to(device)  # Add batch dimension

    # Forward pass to extract features from conv3
    activation = None
    def hook_fn(module, input, output):
        nonlocal activation
        activation = output.detach()  # Store feature maps safely

    hook = model.conv3.register_forward_hook(hook_fn)
    with torch.no_grad():
        output = model(input_tensor)
    hook.remove()

    # Ensure activation was captured
    if activation is None:
        raise ValueError("Activation hook did not capture any output!")

    # Determine the class index (if not provided, use predicted class)
    if class_idx is None:
        class_idx = torch.argmax(output, dim=1).item()

    # Ensure the class index is within bounds
    class_idx = min(class_idx, model.fc2.out_features - 1)

    # Get the weights for the target class from fc2
    fc_weights = model.fc2.weight[class_idx].detach()

    # Validate fc_weights
    if torch.isnan(fc_weights).any() or torch.isinf(fc_weights).any():
        raise ValueError("fc_weights contain NaN or Inf values!")

    # Generate CAM by weighted sum of feature maps
    cam = torch.zeros(activation.shape[2:], device=device)
    for i in range(activation.shape[1]):  # Iterate over channels
        cam += fc_weights[i] * activation[0, i, :, :]

    cam = torch.relu(cam)  # Apply ReLU to remove negative values

    # Normalize CAM for visualization
    cam = cam.cpu().numpy()

    if cam.max() > cam.min():  # Avoid division by zero
        cam -= cam.min()
        cam /= cam.max()
    else:
        cam = np.zeros_like(cam)  # Default blank heatmap

    # Handle NaN or Inf values safely
    cam = np.nan_to_num(cam, nan=0.0, posinf=1.0, neginf=0.0)

    # Resize CAM to the original image dimensions
    cam_image = Image.fromarray((cam * 255).astype('uint8')).resize(original_image.size, Image.LANCZOS)

    # Convert CAM to heatmap
    heatmap = cam_image.convert("L")

    # Overlay CAM onto the original image
    overlay = Image.blend(original_image.convert("L"), heatmap, alpha=0.5)

    # Enhance contrast to make CAM more visible
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

def run_ensemble(model_paths, model_types, image_path, CAM, resolution = '7x7'):
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
            showCAM(heatmaps, [], image_path)
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
    type_predictions = (avg_outputs_type[:, 1] > 0.54).astype("int32")
    
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
    
    # Compute mean heatmap for disease presence
    mean_present_CAM = np.mean(disease_present_CAMS, axis=0)
    mean_present_CAM = (mean_present_CAM - np.min(mean_present_CAM)) / (np.max(mean_present_CAM) - np.min(mean_present_CAM) + 1e-8)

    # Resize heatmap to match image dimensions
    heatmap_size = (image.shape[1], image.shape[0])
    mean_present_CAM = cv2.resize(mean_present_CAM, heatmap_size)

    # Generate overlayed heatmap for disease presence
    heatmap_present = cv2.applyColorMap((mean_present_CAM * 255).astype(np.uint8), cv2.COLORMAP_JET)
    overlay_present = cv2.addWeighted(image, 0.5, heatmap_present, 0.5, 0)

    # Check if disease_type_CAMS exists
    has_disease_type = len(disease_type_CAMS) > 0

    if has_disease_type:
        # Compute mean heatmap for disease type
        mean_type_CAM = np.mean(disease_type_CAMS, axis=0)
        mean_type_CAM = (mean_type_CAM - np.min(mean_type_CAM)) / (np.max(mean_type_CAM) - np.min(mean_type_CAM) + 1e-8)
        mean_type_CAM = cv2.resize(mean_type_CAM, heatmap_size)

        # Generate overlayed heatmap for disease type
        heatmap_type = cv2.applyColorMap((mean_type_CAM * 255).astype(np.uint8), cv2.COLORMAP_JET)
        overlay_type = cv2.addWeighted(image, 0.5, heatmap_type, 0.5, 0)

    # Create figure with correct number of rows
    num_rows = 3 if has_disease_type else 2
    fig, axes = plt.subplots(num_rows, 1, figsize=(5, 4 * num_rows), constrained_layout=True)

    # Function to add a properly formatted colorbar
    def add_colorbar(ax, im):
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)  # Adjusted padding for vertical layout
        cbar = fig.colorbar(im, cax=cax, shrink=0.8)  # Keep colorbar compact
        cbar.set_label("Activation Intensity", fontsize=9, labelpad=6)
        cbar.ax.tick_params(labelsize=8)
        cbar.mappable.set_clim(0, 1)  # Normalize colorbar values to [0,1]

    # Original Image
    axes[0].imshow(image)
    axes[0].set_title("Original Image", fontsize=12)
    axes[0].axis("off")

    # Disease Presence CAM with colorbar
    im1 = axes[1].imshow(overlay_present, cmap="jet", vmin=0, vmax=1)
    axes[1].set_title("Mean Disease Presence CAM", fontsize=12)
    axes[1].axis("off")
    add_colorbar(axes[1], im1)

    # Only show Disease Type CAM if available
    if has_disease_type:
        im2 = axes[2].imshow(overlay_type, cmap="jet", vmin=0, vmax=1)
        axes[2].set_title("Mean Disease Type CAM", fontsize=12)
        axes[2].axis("off")
        add_colorbar(axes[2], im2)

    plt.show()

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
    while run:
        show_menu()
        choice = input("Enter your choice (1-5): ").strip()
        if choice == "1":
            run_it(model_paths, model_types)
        elif choice == "2":
            print(models_listed)
            input(press_to_continue)
        elif choice == "3":
            print(model_structure_options)
            model_number = input("What model would you like to see? (1-4)").strip()
            if model_number == "1":
                print(cnn_structure)
                input(press_to_continue)
            if model_number == "2":
                print(denseB5_structure)
                input(press_to_continue)
            if model_number == "3":
                print(denseA3_structure)
                input(press_to_continue)
            if model_number == "4":
                print(eff4_structure)
                input(press_to_continue)
            else:
                print('That is not a valid model. Returning to menu.')
        elif choice == "4":
            model_number = input(list_model_options).strip()
            if model_number == "1":
                print(denseA3_performance)
                input(press_to_continue)
            elif model_number == "2":
                print(denseB5_report)
                input(press_to_continue)
            elif model_number == "3":
                print(eff4_report)
                input(press_to_continue)
            elif model_number == "4":
                print(ensemble_report)
                input(press_to_continue)
            elif model_number == "5":
                print(standardCNN_report)
                input(press_to_continue)
            elif model_number == "6":
                print(fullEnsemble_report)
                input(press_to_continue)
            else:
                print('That is not a valid option. Returning to menu.')
        elif choice == "5":
            print("Thanks for using our model!")
            run = False
        else:
            print("Invalid input, please enter a digit 1-5")