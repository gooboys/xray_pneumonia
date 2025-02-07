import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split as sk_train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
# from models import eff3, eff4, denseA3, denseB5
from camModels import denseA3, denseB5, eff4
from models import standardCNN

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

'''
This function validates the models in an ensemble. It runs binary classification twice to separate three classes.
The validation set is created using the same random state when sorting out the test set for training.

validation_set = the set of data for the validation set
model_paths = {
    infection_present_models: ['modelpaths', 'modelpaths', 'etc']
    infection_type_models: ['modelpaths', 'modelpaths', 'etc']
}
model_types = {
    'modelpath': model_type,
    'modelpath': model_type,
    'modelpath': model_type,
    'modelpath': model_type
}
'''

def validate_models(validation_set, model_paths, model_types, threshhold = 0.5, graphs=True):
    # Extract lists for both ensemble stages
    present_model_paths = model_paths['infection_present_models']
    type_model_paths = model_paths['infection_type_models']

    # Create dataset and dataloader for the validation data
    test_dataset = ImageDataset(validation_set, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    all_labels = []
    all_final_predictions = []
    
    # First stage: Predict whether infection is present
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        ensemble_outputs_present = []

        for weights_path in present_model_paths:
            model = standardCNN().to(device)
            model.load_state_dict(torch.load(weights_path, weights_only=True))
            model.eval()
            with torch.no_grad():
                outputs = model(images)
                ensemble_outputs_present.append(F.softmax(outputs, dim=1).cpu().numpy())

        # Average predictions for infection presence
        avg_outputs_present = np.mean(ensemble_outputs_present, axis=0)
        is_infected = (np.argmax(avg_outputs_present, axis=1) == 1)

        # Second stage: Classify type of infection if present
        final_predictions = np.zeros(images.shape[0], dtype=int) # Default: non-infected  np.argmax(avg_outputs_present, axis=1)

        if np.any(is_infected):
            is_infected = torch.tensor(is_infected, device=device, dtype=torch.bool)
            images_infected = images[is_infected]
            ensemble_outputs_type = []

            for weights_path in type_model_paths:
                modeltype = model_types[weights_path]
                model = modeltype().to(device)
                model.load_state_dict(torch.load(weights_path, weights_only=True))
                model.eval()
                with torch.no_grad():
                    outputs = model(images_infected)
                    ensemble_outputs_type.append(F.softmax(outputs, dim=1).cpu().numpy())

            avg_outputs_type = np.mean(ensemble_outputs_type, axis=0)

            # Switch between argmax and custom threshhold, comment out one
            # type_predictions = np.argmax(avg_outputs_type, axis=1)
            type_predictions = (avg_outputs_type[:, 1] > threshhold).astype("int32")
            
            final_predictions[is_infected.cpu().numpy()] = type_predictions + 1  # Assign correct class labels

        all_final_predictions.extend(final_predictions)
        all_labels.extend(labels.cpu().numpy())

    # Convert to NumPy arrays
    all_labels = np.array(all_labels)
    all_final_predictions = np.array(all_final_predictions)


    # Declaring Class Names
    class_names = ['NORMAL', 'BACTERIA', 'VIRUS']

    # Compute Metrics
    conf_matrix = confusion_matrix(all_labels, all_final_predictions)
    class_report = classification_report(all_labels, all_final_predictions, target_names=class_names, digits=4)

    # Print results
    print(f"Confusion Matrix:\n{conf_matrix}")
    print(f"Classification Report:\n{class_report}")


    return {"conf_matrix": conf_matrix, "class_report": class_report}


if __name__ == "__main__":
    file_path = 'Normalized_Image_Paths.csv'

    test_data, train_data = train_test_split(file_path, 0.1)

    model_paths = {
        'infection_present_models': ['TrainedModels/standard1.pth','TrainedModels/standard2.pth','TrainedModels/standard3.pth',
                                     'TrainedModels/standard4.pth','TrainedModels/standard5.pth'],
        'infection_type_models': [
                                'TrainedModels/denseA31.pth','TrainedModels/denseA32.pth','TrainedModels/denseA33.pth','TrainedModels/denseA34.pth',
                                'TrainedModels/eff41.pth','TrainedModels/eff42.pth','TrainedModels/eff43.pth','TrainedModels/eff44.pth',
                                'TrainedModels/denseB51.pth','TrainedModels/denseB52.pth','TrainedModels/denseB53.pth','TrainedModels/denseB54.pth'
        ]
    }

    model_types = {
        'TrainedModels/denseA31.pth': denseA3,'TrainedModels/denseA32.pth': denseA3,'TrainedModels/denseA33.pth': denseA3,'TrainedModels/denseA34.pth': denseA3,
        'TrainedModels/eff41.pth': eff4,'TrainedModels/eff42.pth': eff4,'TrainedModels/eff43.pth': eff4,'TrainedModels/eff44.pth': eff4,
        'TrainedModels/denseB51.pth': denseB5,'TrainedModels/denseB52.pth': denseB5,'TrainedModels/denseB53.pth': denseB5,'TrainedModels/denseB54.pth': denseB5
    }

    validate_models(train_data, model_paths, model_types, 0.5)
    validate_models(train_data, model_paths, model_types, 0.51)
    validate_models(train_data, model_paths, model_types, 0.52)
    validate_models(train_data, model_paths, model_types, 0.53)
    validate_models(train_data, model_paths, model_types, 0.54)