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
from models import eff3, eff4, denseB5, denseA3

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

    # Split the balanced data into training and testing sets with a split determined by test_size
    train_data, test_data = sk_train_test_split(data_balanced, test_size=test_size, stratify=data_balanced['Labels'], random_state=42)

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

def montecarlo(model_class, train_data, test_data, criterion, optimizer_class, num_splits, train_size, num_epochs, batch_size, device, transform, limit, learn_rate, dropout_rate):
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
        model = model_class(dropout_rate).to(device)
        optimizer = optimizer_class(model.parameters(), lr=learn_rate)

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

        # Calculate final predictions
        predicted = np.argmax(ensemble_outputs, axis=1)
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
    
    return auc_roc_score

def objective(trial, model_class):
    # Define the hyperparameter search space
    lr = trial.suggest_loguniform('lr', 1e-5, 1e-3)
    dropout = trial.suggest_uniform('dropout', 0.3, 0.7)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])

    # csv file containing [Path, Label] for each normalized image
    csv_file = 'infection_type_labels.csv'
   
    # Split the data into training and testing (80-20) while maintaining balanced classes
    train_data, test_data = train_test_split(csv_file, 0.1)
   
    # Initialize hyperparameters
    num_splits = 4
    train_size = 0.9
    num_epochs = 15
    
    # Initialize other parameters
    criterion = nn.CrossEntropyLoss()
    optimizer_class = torch.optim.Adam
    limit = 3

    # Train the model with these hyperparameters
    roc_auc = montecarlo(model_class, train_data, test_data, criterion, optimizer_class, num_splits, train_size, num_epochs, batch_size, device, transform, limit, learn_rate=lr, dropout_rate=dropout)

    return roc_auc  # Maximize ROC-AUC

if __name__ == "__main__":
    # BELOW ARE THE TRIALS TESTING FOR LEARNING RATE, DROPOUT, AND BATCH SIZE
    # study_a = optuna.create_study(direction="maximize")
    # study_a.optimize(lambda trial: objective(trial, model_class=eff3), n_trials=50)
    # with open('bayes.txt', 'w') as file:
    #     file.write('eff3:\n')  # Write the model name
    #     file.write(f'Best Hyperparameters: {study_a.best_params}\n')  # Write best parameters
    #     file.write(f'Best ROC-AUC Score: {study_a.best_value}\n\n')  # Write best score

    # study_b = optuna.create_study(direction="maximize")
    # study_b.optimize(lambda trial: objective(trial, model_class=eff4), n_trials=50)
    # with open('bayes.txt', 'w') as file:
    #     file.write('eff4:\n')  # Write the model name
    #     file.write(f'Best Hyperparameters: {study_b.best_params}\n')  # Write best parameters
    #     file.write(f'Best ROC-AUC Score: {study_b.best_value}\n\n')  # Write best score

    # study_c = optuna.create_study(direction="maximize")
    # study_c.optimize(lambda trial: objective(trial, model_class=denseA3), n_trials=50)
    # with open('bayes.txt', 'a') as file:
    #     file.write('denseA3}:\n')  # Write the model name
    #     file.write(f'Best Hyperparameters: {study_c.best_params}\n')  # Write best parameters
    #     file.write(f'Best ROC-AUC Score: {study_c.best_value}\n\n')  # Write best score

    study_d = optuna.create_study(direction="maximize")
    study_d.optimize(lambda trial: objective(trial, model_class=denseB5), n_trials=50)
    with open('bayes.txt', 'a') as file:
        file.write('denseB5:\n')  # Write the model name
        file.write(f'Best Hyperparameters: {study_d.best_params}\n')  # Write best parameters
        file.write(f'Best ROC-AUC Score: {study_d.best_value}\n\n')  # Write best score