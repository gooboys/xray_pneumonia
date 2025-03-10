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
from models import (
    standardCNN, oneCNN, twoCNN, threeCNN, fourCNN, fiveCNN, sixCNN, sevenCNN, transfer,
    denseA1, denseA2, denseA3, denseA4, denseA5,
    denseB1, denseB2, denseB3, denseB4, denseB5,
    eff1, eff2, eff3, eff4, eff5,
    mobileV21, mobileV22, mobileV23, mobileV24, mobileV25,
    mobileV31, mobileV32, mobileV33, mobileV34, mobileV35
)

transform = transforms.ToTensor()  # Convert images to PyTorch tensors
# Check if a GPU is available and set the device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# This train_test_split creates 2 sets of train/test for the binary classification breakdowns, takes in 'Normalized_Image_Paths.csv'
def train_test_split(file_path,test_size, printy=False):
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

    # Split the balanced data into training and testing sets with a split based on test_size
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
    if printy:
        print("Training set (disease present) class distribution:\n", disease_type_train['Labels'].value_counts(normalize=True))
        print("Testing set (disease present) class distribution:\n", disease_type_test['Labels'].value_counts(normalize=True))
        print("Training set (disease type) class distribution:\n", disease_train_balanced['Labels'].value_counts(normalize=True))
        print("Testing set (disease type) class distribution:\n", disease_test_balanced['Labels'].value_counts(normalize=True))

        # Display sample rows from each dataset
        print("\nSample rows from disease_type_train:")
        print(disease_type_train.sample(5, random_state=42))

        print("\nSample rows from disease_type_test:")
        print(disease_type_test.sample(5, random_state=42))

        print("\nSample rows from disease_train_balanced:")
        print(disease_train_balanced.sample(5, random_state=42))

        print("\nSample rows from disease_test_balanced:")
        print(disease_test_balanced.sample(5, random_state=42))

    return disease_type_train, disease_type_test

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

def montecarlo(model_class, train_data, test_data, criterion, optimizer_class, num_splits, train_size, num_epochs, batch_size, device, transform, limit, plot=False):
    train_loss_all = []
    val_loss_all = []
    train_accuracy_all = []
    val_accuracy_all = []
    
    models = []  # Store models in memory

    for split in range(num_splits):
        print(f"Monte Carlo Split {split+1}/{num_splits}")
        
        # Initialize split-specific best loss
        best_val_loss = float('inf')
        
        # Randomly split the data into training and validation sets
        train_data, val_data = sk_train_test_split(train_data, train_size=train_size, stratify=train_data['Labels'])

        # Initialize a new model for each split
        model = model_class(0.6196310613745626).to(device)
        optimizer = optimizer_class(model.parameters(), lr=0.000420675036631029)

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

        counter = 0

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
            
            # Save the best model for the current split
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_model = model_class().to(device)
                best_model.load_state_dict(model.state_dict())
                counter = 0
            if counter > limit:
                break
            counter += 1

        # Append split results
        train_loss_all.append(train_losses)
        val_loss_all.append(val_losses)
        train_accuracy_all.append(split_train_accuracy)
        val_accuracy_all.append(split_val_accuracy)

        # Store the best model in memory
        models.append(best_model)
    
    # Compute average loss over all splits
    avg_train_loss = [sum(epoch_losses) / num_splits for epoch_losses in zip(*train_loss_all)]
    avg_val_loss = [sum(epoch_losses) / num_splits for epoch_losses in zip(*val_loss_all)]
    
    # Compute average accuracy over all splits
    avg_train_accuracy = [sum(epoch_accuracies) / num_splits for epoch_accuracies in zip(*train_accuracy_all)]
    avg_val_accuracy = [sum(epoch_accuracies) / num_splits for epoch_accuracies in zip(*val_accuracy_all)]
    
    if plot:
        # Plot results
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

    # Testing on the test data
    test_dataset = ImageDataset(test_data, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    all_labels = []
    all_predictions = []
    all_probabilities = []

    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        
        ensemble_outputs = []
        for model in models:
            model.eval()
            with torch.no_grad():
                outputs = model(images)
                ensemble_outputs.append(F.softmax(outputs, dim=1).cpu().numpy())
        
        # Average predictions across the ensemble
        ensemble_outputs = np.mean(ensemble_outputs, axis=0)
        all_probabilities.extend(ensemble_outputs)

        # Apply custom probability threshold for classification
        custom_threshold = 0.64   # Adjust based on your needs
        predicted = (ensemble_outputs[:, 1] > custom_threshold).astype("int32")  # Class 1 if prob > 0.6

        # Calculate final predictions
        # predicted = np.argmax(ensemble_outputs, axis=1)
        all_predictions.extend(predicted)
        all_labels.extend(labels.cpu().numpy())

    # Confusion Matrix and Classification Report
    class_names = ['BACTERIAL', 'VIRAL']
    conf_matrix = confusion_matrix(all_labels, all_predictions)
    class_report = classification_report(all_labels, all_predictions, target_names=class_names)

    # AUC-ROC for multi-class classification
    all_labels_one_hot = np.eye(len(class_names))[all_labels]
    auc_roc_score = roc_auc_score(all_labels_one_hot, np.array(all_probabilities), multi_class='ovr')

    print(f"Confusion Matrix:\n{conf_matrix}")
    print(f"Classification Report:\n{class_report}")
    print(f"AUC-ROC Score: {auc_roc_score:.4f}")
    
    return class_report, auc_roc_score



if __name__ == "__main__":
    # csv file containing [Path, Label] for each normalized image
    csv_file = 'Normalized_Image_Paths.csv'
   
    # Split the data into training and testing (90-10) while maintaining balanced classes
    train_data, test_data = train_test_split(csv_file, 0.1)
   
    # Initialize hyperparameters
    num_splits = 4
    train_size = 0.9
    num_epochs = 15
    batch_size = 16
    
    # Initialize other parameters
    criterion = nn.CrossEntropyLoss()
    optimizer_class = torch.optim.Adam
    limit = 10

    txt_file = "xModels.txt"

    # List of model objects
    models = [
        denseA3
    ]

    # List of model names as strings
    model_names = [
        'denseA3'
    ]

    models_and_names = zip(models,model_names)
    
    for model, name in models_and_names:
        report, ROC_score = montecarlo(model, train_data, test_data, criterion, optimizer_class, num_splits, train_size, num_epochs, batch_size, device, transform, limit)
        with open(txt_file, "a") as file:
            file.write(name + ":\n")
            file.write(report)
            file.write(f"ROC Score: {ROC_score}")
            file.write("\n")