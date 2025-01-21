import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split as sk_train_test_split
from sklearn.metrics import roc_auc_score, classification_report
from models import tdense121
from omodels import (
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

        return image, float(label)

# Training function with train and validation accuracy, saving the best model to memory
def train_model(model, trainloader, valloader, criterion, optimizer, num_epochs, device):
    model.train()  # Set the model to training mode
    best_val_loss = float("inf")  # Initialize best validation loss to infinity
    best_model_state = None  # To store the best model state

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        epoch_train_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs).squeeze(1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            epoch_train_loss += loss.item()
            
            # Calculate training accuracy
            predicted = (outputs >= 0.5).float()  # Binary prediction based on logits
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
        
        avg_train_loss = epoch_train_loss / len(trainloader)
        train_accuracy = 100 * correct_train / total_train
        
        # Validation phase
        model.eval()
        epoch_val_loss = 0.0
        correct_val = 0
        total_val = 0
        
        with torch.no_grad():
            for inputs, labels in valloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs).squeeze(1)
                loss = criterion(outputs, labels)
                
                epoch_val_loss += loss.item()
                
                # Calculate validation accuracy
                predicted = (outputs >= 0.5).float()
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()
        
        avg_val_loss = epoch_val_loss / len(valloader)
        val_accuracy = 100 * correct_val / total_val

        # Save the model state if the validation loss improves
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict()  # Save the current model state
            print(f"Best model updated with validation loss: {best_val_loss:.4f}")

        # Print metrics for the epoch
        print(f"Epoch {epoch + 1}/{num_epochs} - "
              f"Train Loss: {avg_train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, "
              f"Val Loss: {avg_val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")

    # Load the best model state into the current model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print("Loaded the best model state into the current model.")


# Testing function with ROC-AUC score and classification report
def test_model(model, dataloader, device):
    model.eval()  # Set the model to evaluation mode
    y_true = []
    y_pred = []
    y_scores = []  # To store raw logits for ROC-AUC calculation
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs).squeeze(1)
            predictions = (torch.sigmoid(outputs) >= 0.5).float()  # Apply threshold
            
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predictions.cpu().numpy())
            y_scores.extend(torch.sigmoid(outputs).cpu().numpy())  # Use sigmoid for probabilities
    
    # Calculate metrics
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_scores = np.array(y_scores)
    
    accuracy = (y_true == y_pred).sum() / len(y_true)
    roc_auc = roc_auc_score(y_true, y_scores)  # Use raw probabilities for ROC-AUC
    class_report = classification_report(y_true, y_pred, target_names=['Class 0', 'Class 1'])
    
    # Print results
    print(f"Accuracy: {accuracy:.4f}")
    print(f"ROC-AUC Score: {roc_auc:.4f}")
    print("\nClassification Report:\n", class_report)
    return accuracy, roc_auc, y_true, y_pred

def monte_carlo(splits, data_path, test_size, model_type, criterion, optimizer, num_epochs, device):
    accuracies = []
    rocs = []
    y_preds = []
    y_trues = []
    for split in range(splits):
        print(f"Monte Carlo Split {split+1}/{num_splits}")

        train_data, test_data, fill1, fill2 =  train_test_split(data_path, test_size)
        model = model_type()
        model = model.to(device)
        optimize = optimizer(model.parameters(), lr=0.001)
        # Create DataLoaders
        train_dataset = ImageDataset(train_data, transform=transform)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        val_dataset = ImageDataset(test_data, transform=transform)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        train_model(model, train_loader, val_loader, criterion, optimize, num_epochs, device)
        accuracy, roc, y_true, y_pred = test_model(model, val_loader, device)

        accuracies.append(accuracy)
        rocs.append(roc)
        y_trues.extend(y_true)
        y_preds.extend(y_pred)
    class_report = classification_report(y_trues, y_preds, target_names=['Bacterial', 'Viral'])
    acc = sum(accuracies)/len(accuracies)
    ROCs = sum(rocs)/len(rocs)
    print(f"Classification Report:\n{class_report}")
    print(f"Accuracy: {acc:.4f}")
    print(f"AUC-ROC Score: {ROCs:.4f}")
    return class_report, ROCs



if __name__ == "__main__":
    # csv file containing [Path, Label] for each normalized image
    csv_file = 'Normalized_Image_Paths.csv'
   
    # Initialize hyperparameters
    num_splits = 20
    train_size = 0.9
    num_epochs = 15
    batch_size = 32


    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam

    models = [
        standardCNN, oneCNN, twoCNN, threeCNN, fourCNN, fiveCNN, sixCNN, sevenCNN, transfer,
        denseA1, denseA2, denseA3, denseA4, denseA5,
        denseB1, denseB2, denseB3, denseB4, denseB5,
        eff1, eff2, eff3, eff4, eff5,
        mobileV21, mobileV22, mobileV23, mobileV24, mobileV25,
        mobileV31, mobileV32, mobileV33, mobileV34, mobileV35
    ]
    model_names = [
        "standardCNN","oneCNN", "twoCNN", "threeCNN", "fourCNN", "fiveCNN", "sixCNN", "sevenCNN", "transfer",
        "denseA1", "denseA2", "denseA3", "denseA4", "denseA5",
        "denseB1", "denseB2", "denseB3", "denseB4", "denseB5",
        "eff1", "eff2", "eff3", "eff4", "eff5",
        "mobileV21", "mobileV22", "mobileV23", "mobileV24", "mobileV25",
        "mobileV31", "mobileV32", "mobileV33", "mobileV34", "mobileV35"
    ]
    models_and_names = zip(models,model_names)

    txt_file = 'one_nodes.txt'

    for model, name in models_and_names:
        report, ROC_score = monte_carlo(num_splits, csv_file, 0.1, model, criterion, optimizer, num_epochs, device)
        with open(txt_file, "a") as file:
            file.write(name + ":\n")
            file.write(report)
            file.write(f"ROC Score: {ROC_score}")
            file.write("\n")

    
       
    # # Split the data into training and testing (80-20) while maintaining balanced classes
    # train_data, test_data, a, b = train_test_split(csv_file, 0.1)
    
    # if torch.cuda.is_available():
    #     device = torch.device("cuda:0")
    #     print("running on the GPU")
    # else:
    #     device = torch.device("cpu")
    #     print("running on the CPU")

    # model = tdense121()
    # model = model.to(device)
    # criterion = nn.BCEWithLogitsLoss()
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # # Create DataLoaders
    # train_dataset = ImageDataset(train_data, transform=transform)
    # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # val_dataset = ImageDataset(test_data, transform=transform)
    # val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device)
    # test_model(model, val_loader, device)