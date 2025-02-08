import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split as sk_train_test_split
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
import optuna
from threshholding import train_test_split, ImageDataset
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

    # # Verify the class balance
    # print("Training set (disease present) class distribution:\n", disease_type_train['Labels'].value_counts(normalize=True))
    # print("Testing set (disease present) class distribution:\n", disease_type_test['Labels'].value_counts(normalize=True))
    # print("Training set (disease type) class distribution:\n", disease_train_balanced['Labels'].value_counts(normalize=True))
    # print("Testing set (disease type) class distribution:\n", disease_test_balanced['Labels'].value_counts(normalize=True))

    # # Display sample rows from each dataset
    # print("\nSample rows from disease_type_train:")
    # print(disease_type_train.sample(5, random_state=42))

    # print("\nSample rows from disease_type_test:")
    # print(disease_type_test.sample(5, random_state=42))

    # print("\nSample rows from disease_train_balanced:")
    # print(disease_train_balanced.sample(5, random_state=42))

    # print("\nSample rows from disease_test_balanced:")
    # print(disease_test_balanced.sample(5, random_state=42))

    return disease_type_train, disease_type_test



def montecarlo(model_class, data_path, specs, custom_threshold, learn, dropout=0, plot=False):

    criterion = specs['criterion']
    optimizer_class = specs['optimizer']
    num_splits = specs['splits']
    num_epochs = specs['num_epochs']
    batch_size = specs['batch_size']
    device = specs['device']
    transform = specs['transform']
    limit = specs['limit']
    
    # Initialize accumulators for Monte Carlo runs
    conf_matrices = []
    class_reports = []
    auc_roc_scores = []
    accuracy_scores = []

    for split in range(num_splits):
        print(f"Monte Carlo Split {split+1}/{num_splits}")
        
        # Initialize split-specific best loss
        best_val_loss = float('inf')
        
        # Randomly split the data into training and validation sets
        train_data, val_data = train_test_split(data_path, 0.1)

        # Initialize a new model for each split
        model = None
        if dropout:
            model = model_class(dropout).to(device)
        else:
            model = model_class().to(device)
        optimizer = optimizer_class(model.parameters(), lr=learn)

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

        
        # Testing on the test data
        test_dataset = ImageDataset(val_data, transform=transform)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        all_labels = []
        all_predictions = []
        all_probabilities = []

        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            
            model.eval()
            with torch.no_grad():
                outputs = model(images)
                probabilities = F.softmax(outputs, dim=1).cpu().numpy()  # Convert logits to probabilities

            all_probabilities.extend(probabilities)

            # Apply custom probability threshold for classification
            predicted = (probabilities[:, 1] > custom_threshold).astype("int32")  # Class 1 if prob > threshold

            all_predictions.extend(predicted)
            all_labels.extend(labels.cpu().numpy())

        # Compute metrics for this run
        class_names = ['BACTERIAL', 'VIRAL']
        conf_matrix = confusion_matrix(all_labels, all_predictions)
        class_report = classification_report(all_labels, all_predictions, target_names=class_names, output_dict=True)

        # Convert labels to one-hot encoding for AUC-ROC calculation
        all_labels_one_hot = np.eye(len(class_names))[all_labels]
        auc_roc_score_value = roc_auc_score(all_labels_one_hot, np.array(all_probabilities), multi_class='ovr')

        # Store metrics
        conf_matrices.append(conf_matrix)
        # class_reports.append(class_report)
        class_reports.append({k: v for k, v in class_report.items() if isinstance(v, dict)})
        auc_roc_scores.append(auc_roc_score_value)
         # Separate accuracy from other metrics
        accuracy_scores.append(class_report["accuracy"])

    # === Compute Average Results Over All Splits ===

    # Average Confusion Matrix
    avg_conf_matrix = np.mean(conf_matrices, axis=0).astype(int)
    

    # Average Classification Report (Per-Class Metrics)
    avg_class_report = {}
    for key in class_reports[0]:  # Iterate over class names and global metrics
        avg_class_report[key] = {}
        for metric in class_reports[0][key]:  # e.g., 'precision', 'recall', 'f1-score', 'support'
            avg_class_report[key][metric] = np.mean([report[key][metric] for report in class_reports])

    # Average AUC-ROC Score
    avg_auc_roc_score = np.mean(auc_roc_scores)
    avg_accuracy = np.mean(accuracy_scores)

    # === Print Final Monte Carlo Results ===
    print(f"\n=== Final Monte Carlo Cross-Validation Results ===")
    print(f"Average Confusion Matrix:\n{avg_conf_matrix}")
    print(f"\nAverage Classification Report:")
    for label, metrics in avg_class_report.items():
        print(f"{label}: {metrics}")
    
    print(f'\nAverage Accuracy: {avg_accuracy:.4f}')
    print(f"\nAverage AUC-ROC Score: {avg_auc_roc_score:.4f}")

    return avg_auc_roc_score, avg_class_report, avg_conf_matrix

def objective(trial, model_class):
    # Define the hyperparameter search space
    lr = trial.suggest_loguniform('lr', 1e-5, 1e-3)
    drop = trial.suggest_uniform('dropout', 0.3, 0.7)
    threshhold = trial.suggest_uniform('threshhold', 0.4,0.6)

    # Initialize hyperparameters
    num_splits = 4
    train_size = 0.9
    num_epochs = 15
    batch_size = 32
    
    # Initialize other parameters
    criterion = nn.CrossEntropyLoss()
    optimizer_class = torch.optim.Adam
    limit = 3

    datapath = 'Normalized_Image_Paths.csv'
    specs = {
        'criterion': criterion,
        'optimizer': optimizer_class,
        'splits': num_splits,
        'train_size': train_size,
        'num_epochs': num_epochs,
        'batch_size': batch_size,
        'device': device,
        'transform': transform,
        'limit': limit
    }

    roc_auc, avg_class_report, avg_conf = montecarlo(eff4, datapath, specs, threshhold, lr, dropout= drop)

    return roc_auc  # Maximize ROC-AUC

def log_best_trial(study, trial, model_name):
    """Callback function to log best results when a new best trial is found."""
    if study.best_trial == trial:
        with open('bayes2.txt', 'a') as file:
            file.write(f'{model_name} (New Best Trial):\n')
            file.write(f'Best Hyperparameters: {study.best_params}\n')
            file.write(f'Best ROC-AUC Score: {study.best_value}\n\n')
        print(f'Updated best result for {model_name}: {study.best_value}')

if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(trial, model_class=eff4), n_trials=50,
                   callbacks=[lambda study, trial: log_best_trial(study, trial, eff4)]
                   )
    with open('bayes2.txt', 'a') as file:
        file.write('eff4:\n')  # Write the model name
        file.write(f'Best Hyperparameters: {study.best_params}\n')  # Write best parameters
        file.write(f'Best ROC-AUC Score: {study.best_value}\n\n')  # Write best score

    study1 = optuna.create_study(direction="maximize")
    study1.optimize(lambda trial: objective(trial, model_class=eff3),
                     n_trials=50,
                     callbacks=[lambda study, trial: log_best_trial(study, trial, eff3)])
    with open('bayes2.txt', 'a') as file:
        file.write('eff3:\n')  # Write the model name
        file.write(f'Best Hyperparameters: {study1.best_params}\n')  # Write best parameters
        file.write(f'Best ROC-AUC Score: {study1.best_value}\n\n')  # Write best score

    study2= optuna.create_study(direction="maximize")
    study2.optimize(lambda trial: objective(trial, model_class=denseB5), n_trials=50)
    with open('bayes2.txt', 'a') as file:
        file.write('denseB5:\n')  # Write the model name
        file.write(f'Best Hyperparameters: {study2.best_params}\n')  # Write best parameters
        file.write(f'Best ROC-AUC Score: {study2.best_value}\n\n')  # Write best score
