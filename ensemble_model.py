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
from PIL import ImageEnhance

# Check if a GPU is available and set the device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
       
        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3)  # Input is grayscale (1 channel)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
       
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
       
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
       
        # Fully connected layers
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(26 * 26 * 128, 128)  # Flattened size from last pooling layer
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
   
    def generate_cam(self, class_idx):
        """
        Generates the Class Activation Map (CAM) for a given class index.
        """
        # Get the weights of the fully connected layer for the target class
        fc_weights = self.fc2.weight[class_idx].detach()
   
        # Initialize the CAM as zeros, with the same spatial dimensions as the feature maps
        cam = torch.zeros(self.feature_maps.size(2), self.feature_maps.size(3), device=self.feature_maps.device)
   
        # Accumulate contributions from each feature map
        for i in range(self.feature_maps.size(1)):  # Iterate over feature map channels
            cam += fc_weights[i] * self.feature_maps[0, i, :, :]
   
        # Apply ReLU to remove negative contributions
        cam = torch.relu(cam)
   
        # Normalize CAM for visualization
        cam -= cam.min()
        cam /= cam.max()
       
        return cam

def train_test_split(file_path):
    # Load the data
    data = pd.read_csv(file_path)

    # Check the label distribution
    label_counts = data['Labels'].value_counts()
    min_class_count = label_counts.min()
   
    # Create a balanced sample for each label to achieve roughly 33.3% of each label in train and test sets
    data_balanced = pd.concat([
        data[data['Labels'] == label].sample(min_class_count, random_state=42)
        for label in data['Labels'].unique()
    ])

    # Split the balanced data into training and testing sets with an 80-20 split
    train_data, test_data = sk_train_test_split(data_balanced, test_size=0.2, stratify=data_balanced['Labels'], random_state=42)

    # Verify the class balance
    print("Training set class distribution:\n", train_data['Labels'].value_counts(normalize=True))
    print("Testing set class distribution:\n", test_data['Labels'].value_counts(normalize=True))

    return train_data, test_data

# Define the custom Dataset class to handle image loading
class ImageDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path = self.data.iloc[idx]['Paths']
        label = self.data.iloc[idx]['Labels']
        image = Image.open(image_path)  # Image is already grayscale
       
        if self.transform:
            image = self.transform(image)

        return image, label
   
transform = transforms.ToTensor()  # Convert images to PyTorch tensors

def monte_carlo_cross_validation(model_class, train_data, test_data, criterion, optimizer_class, num_splits, train_size, num_epochs, batch_size, device, transform):
    train_loss_all = []
    val_loss_all = []
   
    train_accuracy_all = []
    val_accuracy_all = []
   
    model_weights_paths = []  # Store paths to the best model weights for each split
   
    for split in range(num_splits):
        print(f"Monte Carlo Split {split+1}/{num_splits}")
       
        # Initialize split-specific best loss
        best_val_loss = float('inf')
        best_weights_path = f'model_split_{split}_best.pth'
        model_weights_paths.append(best_weights_path)  
       
        # Randomly split the data into training and validation sets
        train_data, val_data = sk_train_test_split(train_data, train_size=train_size, stratify=train_data['Labels'])

        # Initialize a new model for each split
        model = model_class().to(device)
        optimizer = optimizer_class(model.parameters(), lr=0.001)

        # Track training and validation losses
        train_losses = []
        val_losses = []
        split_train_accuracy = []
        split_val_accuracy = []

        # Create DataLoaders
        train_dataset = ImageDataset(train_data, transform=transform)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        val_dataset = ImageDataset(val_data, transform=transform)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        for epoch in range(num_epochs):
            # Training
            model.train()
            epoch_train_loss = 0
            correct_train = 0
            total_train = 0
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                epoch_train_loss += loss.item()
               
                # Calculate accuracy
                _, predicted = torch.max(outputs, 1)
                total_train += labels.size(0)
                correct_train += (predicted == labels).sum().item()

            avg_train_loss = epoch_train_loss / len(train_loader)
            train_losses.append(avg_train_loss)
            train_accuracy = 100 * correct_train / total_train
            split_train_accuracy.append(train_accuracy)

            # Validation
            model.eval()
            epoch_val_loss = 0
            correct_val = 0
            total_val = 0  
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    epoch_val_loss += loss.item()
                   
                    # Calculate accuracy
                    _, predicted = torch.max(outputs, 1)
                    total_val += labels.size(0)
                    correct_val += (predicted == labels).sum().item()

            avg_val_loss = epoch_val_loss / len(val_loader)
            val_losses.append(avg_val_loss)
            val_accuracy = 100 * correct_val / total_val
            split_val_accuracy.append(val_accuracy)
           
            print(f"Epoch [{epoch+1}/{num_epochs}] - Train Loss: {avg_train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, Val Loss: {avg_val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")
           
            # Save best weights for the current split
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(model.state_dict(), best_weights_path)

        # Append split results
        train_loss_all.append(train_losses)
        val_loss_all.append(val_losses)
        train_accuracy_all.append(split_train_accuracy)
        val_accuracy_all.append(split_val_accuracy)
   
    # Compute average loss over all splits
    avg_train_loss = [sum(epoch_losses) / num_splits for epoch_losses in zip(*train_loss_all)]
    avg_val_loss = [sum(epoch_losses) / num_splits for epoch_losses in zip(*val_loss_all)]
   
    # Compute average accuracy over all splits
    avg_train_accuracy = [sum(epoch_accuracies) / num_splits for epoch_accuracies in zip(*train_accuracy_all)]
    avg_val_accuracy = [sum(epoch_accuracies) / num_splits for epoch_accuracies in zip(*val_accuracy_all)]
   
    # Plot mccv results
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, num_epochs+1), avg_train_loss, label='Average Training Loss')
    plt.plot(range(1, num_epochs+1), avg_val_loss, label='Average Validation Loss')
    plt.ylim(0.0, 1.0)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training vs Validation Loss (Monte Carlo Cross-Validation)')
    plt.legend()
    plt.grid(True)
    plt.show()
   
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, num_epochs+1), avg_train_accuracy, label='Average Training Accuracy')
    plt.plot(range(1, num_epochs+1), avg_val_accuracy, label='Average Validation Accuracy')
    plt.ylim(0, 100)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training vs Validation Accuracy (Monte Carlo Cross-Validation)')
    plt.legend()
    plt.grid(True)
    plt.show()
   
    # Create dataset and dataloader for the test data
    test_dataset = ImageDataset(test_data, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Aggregate predictions across all models
    all_labels = []
    all_predictions = []
    all_probabilities = []

    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)

        # Collect outputs from each model
        ensemble_outputs = []
        for weights_path in model_weights_paths:
            model = model_class().to(device)
            model.load_state_dict(torch.load(weights_path, weights_only=True))
            model.eval()
            with torch.no_grad():
                outputs = model(images)
                ensemble_outputs.append(F.softmax(outputs, dim=1).cpu().numpy())

        # Average predictions across the ensemble
        ensemble_outputs = np.mean(ensemble_outputs, axis=0)
        all_probabilities.extend(ensemble_outputs)

        # Calculate final predictions
        predicted = np.argmax(ensemble_outputs, axis=1)
        all_predictions.extend(predicted)
        all_labels.extend(labels.cpu().numpy())

    # Confusion Matrix and Classification Report
    class_names = ['NORMAL', 'BACTERIA', 'VIRUS']
    conf_matrix = confusion_matrix(all_labels, all_predictions)
    class_report = classification_report(all_labels, all_predictions, target_names=class_names)

    # AUC-ROC for multi-class classification
    all_labels_one_hot = np.eye(len(class_names))[all_labels]
    auc_roc_score = roc_auc_score(all_labels_one_hot, np.array(all_probabilities), multi_class='ovr')

    # Print results
    print(f"Confusion Matrix:\n{conf_matrix}")
    print(f"Classification Report:\n{class_report}")
    print(f"AUC-ROC Score: {auc_roc_score:.4f}")
   
    # ROC Curve plotting
    fpr, tpr, _ = roc_curve(all_labels_one_hot.ravel(), np.array(all_probabilities).ravel())
    plt.figure(figsize=(10, 8))
   
    # One line for each class
    for i, class_name in enumerate(class_names):
        fpr_class, tpr_class, _ = roc_curve(all_labels_one_hot[:, i], np.array(all_probabilities)[:, i])
        plt.plot(fpr_class, tpr_class, label=f'ROC curve for {class_name} (AUC = {roc_auc_score(all_labels_one_hot[:, i], np.array(all_probabilities)[:, i]):.2f})')
   
    # Plotting settings
    plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line (random model)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.show()

    return model_weights_paths


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


# Runs the model with an image from image path, also takes in a model if it is not a string.
def run_model(model_class, model_paths, image_path, transform, device, showCAM):

    image = Image.open(image_path).convert("L")
    input_tensor = transform(image).unsqueeze(0).to(device)
   
    heatmaps = []
    ensemble_outputs = []
    df = pd.read_csv('Normalized_Image_Paths.csv')
    label = int(df.loc[df['Paths'] == image_path, 'Labels'].values[0])
    for path in model_paths:
        model = model_class().to(device)
        model.load_state_dict(torch.load(path, weights_only=True))
        model.eval()
        with torch.no_grad():
            outputs = model(input_tensor)
            ensemble_outputs.append(F.softmax(outputs, dim=1).cpu().numpy())
        overlay, heatmap = generate_cam(model, image_path, label, make_graphs=showCAM)
        heatmaps.append(heatmap)
   
    # Average predictions
    ensemble_outputs = np.mean(ensemble_outputs, axis=0)
    predicted_class = np.argmax(ensemble_outputs)
   
    class_mapping = {0: "normal", 1: "bacterial", 2: "viral"}
    print(f"Predicted Class (Ensemble): {class_mapping.get(predicted_class, 'Unknown')}")
    print(f"Actual Class: {class_mapping.get(label, 'Unknown')}")
   
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
   
    # csv file containing [Path, Label] for each normalized image
    csv_file = 'Normalized_Image_Paths.csv'
   
    # Split the data into training and testing (80-20) while maintaining balanced classes
    train_data, test_data = train_test_split(csv_file)
   
    # Initialize the model and move it to the GPU if available
    model = CNN().to(device)
   
    # Initialize hyperparameters
    num_splits = 5
    train_size = 0.8
    num_epochs = 10
    batch_size = 32
   
    # Perform MCCV
    model_paths = monte_carlo_cross_validation(
        model_class=CNN,
        train_data=train_data,
        test_data=test_data,
        criterion=nn.CrossEntropyLoss(),
        optimizer_class=torch.optim.Adam,
        num_splits=num_splits,
        train_size=train_size,
        num_epochs=num_epochs,
        batch_size=batch_size,
        device=device,
        transform=transform
    )
   
    # model_paths = ["model_split_0_best.pth","model_split_1_best.pth","model_split_2_best.pth","model_split_3_best.pth","model_split_4_best.pth"]

    # Run model on single case
    image_path = 'NormalizedXRays/image_3883.jpeg'
    run_model(CNN, model_paths, image_path, transform, device, True)