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
    print("            By: Thierry Juang")
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