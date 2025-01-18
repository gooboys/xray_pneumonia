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
from models import eff3, eff4, denseA3, denseB5, standardCNN

transform = transforms.ToTensor()  # Convert images to PyTorch tensors
# Check if a GPU is available and set the device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# This train_test_split creates 2 sets of train/test for the binary classification breakdowns, takes in 'Normalized_Image_Paths.csv'
def train_test_split(file_path,test_size):
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
    train_data, test_data = sk_train_test_split(data_balanced, test_size=test_size, stratify=data_balanced['Labels'], random_state=42)

    # Copying data for disease type split, removing non-infected cases
    disease_type_train = train_data[train_data['Labels'] != 0].copy()
    disease_type_test = test_data[test_data['Labels'] != 0].copy()

    # Changing labels from 1->0 2->1
    label_mapping = {1:0,2:1}
    disease_type_train['Labels'] = disease_type_train['Labels'].map(label_mapping)
    disease_type_test['Labels'] = disease_type_test['Labels'].map(label_mapping)

    # Copying data for disease present split
    disease_train = train_data.copy()
    disease_test = test_data.copy()

    # Turning both disease cases into one class
    disease_train['Labels'] = disease_train['Labels'].apply(lambda x: 1 if x == 2 else x)
    disease_test['Labels'] = disease_test['Labels'].apply(lambda x: 1 if x == 2 else x)

    # Check the label distribution
    disease_train_min = disease_train['Labels'].value_counts().min()
    disease_test_min = disease_test['Labels'].value_counts().min()
    
    # Create a balanced sample for each label to achieve roughly 50% of each label in train and test sets
    disease_train_balanced = pd.concat([
        disease_train[disease_train['Labels'] == label].sample(disease_train_min, random_state=42)
        for label in disease_train['Labels'].unique()
    ])
    disease_test_balanced = pd.concat([
        disease_test[disease_test['Labels'] == label].sample(disease_test_min, random_state=42)
        for label in disease_test['Labels'].unique()
    ])

    # Verify the class balance
    print("Training set (disease present) class distribution:\n", disease_type_train['Labels'].value_counts(normalize=True))
    print("Testing set (disease present) class distribution:\n", disease_type_test['Labels'].value_counts(normalize=True))
    print("Training set (disease type) class distribution:\n", disease_train_balanced['Labels'].value_counts(normalize=True))
    print("Testing set (disease type) class distribution:\n", disease_test_balanced['Labels'].value_counts(normalize=True))

    return disease_type_train, disease_type_test, disease_train_balanced, disease_test_balanced


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

'''
data = {
    'train': train_data,
    'test': test_data,
    'transform': transform
}
training_config = {
    'criterion': criterion,
    'optimizer_class': optimizer_class,
    'model_paths': model_paths, #(list of model names)
    'train_size': train_size, # (decimal of test data ratio)
    'num_epochs': num_epochs,
    'batch_size': batch_size
}
meta_data = {
    'model_type': 0, # (0 for if disease is present, 1 for disease type classification)
    'graphs': 0, # (1 to print graphs, 0 to not print graphs)
}
'''
def monte_carlo_cross_validation(model_class, data, training_config, meta_data, device):
    train_data = data.get('train')
    test_data = data.get('test')
    transform = data.get('transform')

    criterion = training_config.get('criterion')
    optimizer_class = training_config.get('optimizer_class')
    model_paths = training_config.get('model_paths')
    train_size = training_config.get('train_size')
    num_epochs = training_config.get('num_epochs')
    batch_size = training_config.get('batch_size')

    model_type = meta_data.get('model_type')
    graphs = meta_data.get('graphs')
    
    train_loss_all = []
    val_loss_all = []
   
    train_accuracy_all = []
    val_accuracy_all = []
    
    num_splits = len(model_paths)

    for model_path in model_paths:       
        split = 1
        print(f"Monte Carlo Split {split}/{num_splits}")
        split += 1

        # Initialize split-specific best loss
        best_val_loss = float('inf')
       
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
                torch.save(model.state_dict(), model_path)

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
    if graphs:
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
        for weights_path in model_paths:
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
    class_names = []
    if model_type:
        class_names = ['Bacterial','Viral']
    else:
        class_names = ['No Infection','Infection']
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
   
    if graphs:
        # Plotting settings
        plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line (random model)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc='lower right')
        plt.show()

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
    # csv file containing [Path, Label] for each normalized image
    csv_file = 'Normalized_Image_Paths.csv'
   
    # Split the data into training and testing (80-20) while maintaining balanced classes
    disease_type_train, disease_type_test, disease_train, disease_test = train_test_split(csv_file, 0.1)
    
    data = {
        'train': disease_train,
        'test': disease_test,
        'transform': transform
    }

    # For model eff3
    training_config1 = {
        'criterion': nn.CrossEntropyLoss(),
        'optimizer_class': torch.optim.Adam,
        'model_paths': ['TrainedModels/eff31.pth','TrainedModels/eff32.pth', 'TrainedModels/eff33.pth', 'TrainedModels/eff34.pth'],
        'train_size': 0.1,
        'num_epochs': 10,
        'batch_size': 32
    }
    meta_data1 = {
        'model_type': 1,
        'graphs': 0,
    }

    monte_carlo_cross_validation(eff3, data, training_config1, meta_data1, device)

    # For model eff4
    training_config1 = {
        'criterion': nn.CrossEntropyLoss(),
        'optimizer_class': torch.optim.Adam,
        'model_paths': ['TrainedModels/eff41.pth','TrainedModels/eff42.pth','TrainedModels/eff43.pth','TrainedModels/eff44.pth'],
        'train_size': 0.1,
        'num_epochs': 10,
        'batch_size': 32
    }
    meta_data1 = {
        'model_type': 1,
        'graphs': 0,
    }

    monte_carlo_cross_validation(eff4, data, training_config1, meta_data1, device)

    # For model denseA3
    training_config1 = {
        'criterion': nn.CrossEntropyLoss(),
        'optimizer_class': torch.optim.Adam,
        'model_paths': ['TrainedModels/denseA31.pth','TrainedModels/denseA32.pth','TrainedModels/denseA33.pth','TrainedModels/denseA34.pth'],
        'train_size': 0.1,
        'num_epochs': 10,
        'batch_size': 32
    }
    meta_data1 = {
        'model_type': 1,
        'graphs': 0,
    }

    monte_carlo_cross_validation(denseA3, data, training_config1, meta_data1, device)

    # For model denseB5
    training_config1 = {
        'criterion': nn.CrossEntropyLoss(),
        'optimizer_class': torch.optim.Adam,
        'model_paths': ['TrainedModels/denseB51.pth','TrainedModels/denseB52.pth','TrainedModels/denseB53.pth','TrainedModels/denseB54.pth'],
        'train_size': 0.1,
        'num_epochs': 10,
        'batch_size': 32
    }
    meta_data1 = {
        'model_type': 1,
        'graphs': 0,
    }

    monte_carlo_cross_validation(denseB5, data, training_config1, meta_data1, device)

    # For standardCNN
    training_config1 = {
        'criterion': nn.CrossEntropyLoss(),
        'optimizer_class': torch.optim.Adam,
        'model_paths': ['TrainedModels/standard1.pth','TrainedModels/standard2.pth','TrainedModels/standard3.pth','TrainedModels/standard4.pth'],
        'train_size': 0.1,
        'num_epochs': 10,
        'batch_size': 32
    }
    meta_data1 = {
        'model_type': 0,
        'graphs': 0,
    }

    monte_carlo_cross_validation(standardCNN, data, training_config1, meta_data1, device)