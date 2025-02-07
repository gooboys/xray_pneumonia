import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split as sk_train_test_split
from sklearn.metrics import roc_auc_score, roc_curve
from camModels import denseA3, denseB5, eff4
from sklearn.metrics import confusion_matrix, classification_report
from models import standardCNN

transform = transforms.ToTensor()  # Convert images to PyTorch tensors
# Check if a GPU is available and set the device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_test_split(file_path,test_size, print = False):
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
    if print:
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
    
# Testing function
def test_models(model_classes, model_paths, dataloader, threshhold=0):
    y_true = []
    y_pred = []
    y_scores = []  # Stores predicted probabilities for ROC-AUC

    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        ensemble_outputs_present = []

        for weights_path in model_paths:
            model_class = model_classes[weights_path]
            model = model_class().to(device)
            model.load_state_dict(torch.load(weights_path, weights_only=True))
            model.eval()
            with torch.no_grad():
                outputs = model(images)
                ensemble_outputs_present.append(F.softmax(outputs, dim=1).cpu().numpy())

        # Average predictions for infection presence
        avg_outputs_present = np.mean(ensemble_outputs_present, axis=0)
        y_scores.extend(avg_outputs_present[:, 1].tolist())  # Store positive class probabilities

        if threshhold:
            final_predictions = (avg_outputs_present[:, 1] > threshhold).astype("int32")
        else:
            final_predictions = np.argmax(avg_outputs_present, axis=1)  # Default: non-infected

        y_pred.extend(final_predictions.tolist())
        y_true.extend(labels.cpu().numpy().tolist())

    # Convert lists to numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_scores = np.array(y_scores)  # Probabilities for ROC-AUC

    # Compute accuracy
    accuracy = np.mean(y_true == y_pred)

    # Compute ROC-AUC score
    roc_auc = roc_auc_score(y_true, y_scores)

    # Compute ROC curve
    fpr, tpr, _ = roc_curve(y_true, y_scores)

    # Compute and print the confusion matrix
    conf_matrix = confusion_matrix(y_true, y_pred)
    print("\nConfusion Matrix:\n", conf_matrix)
    print("\nClassification Report:\n", classification_report(y_true, y_pred, digits=4))
    print(f"Accuracy: {accuracy:.4f}")
    print(f"ROC-AUC Score: {roc_auc:.4f}")

if __name__ == "__main__":
    # Data preparation is univeral for all testing:
    # csv file containing [Path, Label] for each normalized image
    csv_file = 'Normalized_Image_Paths.csv'
    # Split the data into training and testing (80-20) while maintaining balanced classes
    train_data, test_data, a, b = train_test_split(csv_file, 0.1)
    val_dataset = ImageDataset(test_data, transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # Dataset for evaluating disease presence model
    d_val_dataset = ImageDataset(b, transform=transform)
    d_val_loader = DataLoader(d_val_dataset, batch_size=32, shuffle=False)
    
    # BELOW FOR entire model ensemble
    model_paths = [
        'TrainedModels/denseA31.pth','TrainedModels/denseA32.pth','TrainedModels/denseA33.pth','TrainedModels/denseA34.pth',
        'TrainedModels/eff41.pth','TrainedModels/eff42.pth','TrainedModels/eff43.pth','TrainedModels/eff44.pth',
        'TrainedModels/denseB51.pth','TrainedModels/denseB52.pth','TrainedModels/denseB53.pth','TrainedModels/denseB54.pth'
        ]
    
    model_types = {
        'TrainedModels/denseA31.pth': denseA3,'TrainedModels/denseA32.pth': denseA3,'TrainedModels/denseA33.pth': denseA3,'TrainedModels/denseA34.pth': denseA3,
        'TrainedModels/eff41.pth': eff4,'TrainedModels/eff42.pth': eff4,'TrainedModels/eff43.pth': eff4,'TrainedModels/eff44.pth': eff4,
        'TrainedModels/denseB51.pth': denseB5,'TrainedModels/denseB52.pth': denseB5,'TrainedModels/denseB53.pth': denseB5,'TrainedModels/denseB54.pth': denseB5
    }
    # BELOW FOR denseNet-121
    model_paths1 = ['TrainedModels/denseA31.pth','TrainedModels/denseA32.pth','TrainedModels/denseA33.pth','TrainedModels/denseA34.pth']    
    model_types1 = {'TrainedModels/denseA31.pth': denseA3,'TrainedModels/denseA32.pth': denseA3,'TrainedModels/denseA33.pth': denseA3,'TrainedModels/denseA34.pth': denseA3}
    # BELOW FOR denseNet-169
    model_paths2 = ['TrainedModels/denseB51.pth','TrainedModels/denseB52.pth','TrainedModels/denseB53.pth','TrainedModels/denseB54.pth']
    model_types2 = {'TrainedModels/denseB51.pth': denseB5,'TrainedModels/denseB52.pth': denseB5,'TrainedModels/denseB53.pth': denseB5,'TrainedModels/denseB54.pth': denseB5}
    # BELOW FOR EfficientNetB-0
    model_paths3 = ['TrainedModels/eff41.pth','TrainedModels/eff42.pth','TrainedModels/eff43.pth','TrainedModels/eff44.pth']
    model_types3 = {'TrainedModels/eff41.pth': eff4,'TrainedModels/eff42.pth': eff4,'TrainedModels/eff43.pth': eff4,'TrainedModels/eff44.pth': eff4}
    # BELOW FOR disease presence
    model_paths4 = ['TrainedModels/standard1.pth','TrainedModels/standard2.pth','TrainedModels/standard3.pth','TrainedModels/standard4.pth','TrainedModels/standard5.pth']
    model_types4 = {'TrainedModels/standard1.pth': standardCNN,'TrainedModels/standard2.pth': standardCNN,'TrainedModels/standard3.pth': standardCNN,'TrainedModels/standard4.pth': standardCNN,'TrainedModels/standard5.pth': standardCNN}

    # Running for entire model ensemble
    test_models(model_types, model_paths, val_loader, threshhold = 0.5)
    test_models(model_types, model_paths, val_loader, threshhold = 0.51)
    test_models(model_types, model_paths, val_loader, threshhold = 0.52)
    test_models(model_types, model_paths, val_loader, threshhold = 0.53)
    test_models(model_types, model_paths, val_loader, threshhold = 0.54)
    # Running for denseNet-121
    test_models(model_types1, model_paths1, val_loader, threshhold = 0.5)
    test_models(model_types1, model_paths1, val_loader, threshhold = 0.51)
    test_models(model_types1, model_paths1, val_loader, threshhold = 0.52)
    test_models(model_types1, model_paths1, val_loader, threshhold = 0.53)
    test_models(model_types1, model_paths1, val_loader, threshhold = 0.54)
    # Running for denseNet-169
    test_models(model_types2, model_paths2, val_loader, threshhold = 0.5)
    test_models(model_types2, model_paths2, val_loader, threshhold = 0.51)
    test_models(model_types2, model_paths2, val_loader, threshhold = 0.52)
    test_models(model_types2, model_paths2, val_loader, threshhold = 0.53)
    test_models(model_types2, model_paths2, val_loader, threshhold = 0.54)
    # Running for EfficientNetB-0
    test_models(model_types3, model_paths3, val_loader, threshhold = 0.5)
    test_models(model_types3, model_paths3, val_loader, threshhold = 0.51)
    test_models(model_types3, model_paths3, val_loader, threshhold = 0.52)
    test_models(model_types3, model_paths3, val_loader, threshhold = 0.53)
    test_models(model_types3, model_paths3, val_loader, threshhold = 0.54)
    # Running for disease presence
    test_models(model_types4, model_paths4, d_val_loader)