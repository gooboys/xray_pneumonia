import torch
from torchvision import transforms
from ensemble_model import run_model, CNN

'''
This file runs the ensemble model for the dual-binary classification. Please check the order of files which need
to be run in the README.md file for this container to work.
'''

# Check if a GPU is available and set the device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

performance_report = """
Deep Neural Network Ensembled Pneumonia Classification Model Report
----------------------------------------------------
Dataset: Pneumonia Diagnosis (Normal vs. Bacterial vs. Viral)

Confusion Matrix:
-----------------
       Predicted
         NORMAL  BACTERIA  VIRUS
      --------------------------------
  NORMAL   |  282     5     11
BACTERIA   |    8   229     62
   VIRUS   |   22    76    201

Classification Report:
----------------------
              Precision    Recall  F1-Score   Support
------------------------------------------------------
     NORMAL     0.90       0.95      0.92       298
   BACTERIA     0.74       0.77      0.75       299
      VIRUS     0.73       0.67      0.70       299

Overall Model Performance:
--------------------------
- Accuracy: **79.00%**
- Macro Avg: Precision = 0.79, Recall = 0.79, F1-Score = 0.79
- Weighted Avg: Precision = 0.79, Recall = 0.79, F1-Score = 0.79
- AUC-ROC Score: **0.9249**

Summary:
--------
The full ensembled pneumonia classification model achieves **79.00% accuracy**, 
effectively distinguishing between **Normal, Bacterial, and Viral pneumonia cases**.

- **Normal cases** exhibit the highest precision (**90%**) and recall (**95%**), 
  meaning most non-infected cases are correctly identified.
- **Bacterial pneumonia** shows a **77% recall**, capturing most bacterial cases 
  while maintaining **74% precision**.
- **Viral pneumonia** has the lowest recall (**67%**), meaning some viral cases 
  are misclassified as bacterial, but maintains **73% precision**.

With an **AUC-ROC score of 0.9249**, the model demonstrates **strong overall 
discrimination capability** between classes. Further threshold tuning may 
optimize the balance between precision and recall for different pneumonia types.
"""

def show_menu():
    print("===========================================")
    print("    Basic pneumonia disease state model    ")
    print("By: Thierry Juang, James Couch, Saha Udassi")
    print("===========================================")
    print("1. Run model prediction on CT Scan")
    print("2. View performance report of the model")
    print("3. Exit the program")
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
    run_model(CNN, ensemble, image_path, transform, device, printer)
    return

if __name__ == "__main__":
    run = True
    while run:
        show_menu()
        ensemble = ["model_split_0_best.pth","model_split_1_best.pth","model_split_2_best.pth","model_split_3_best.pth","model_split_4_best.pth"]
        choice = input("Enter your choice (1-2): ").strip()
        if choice == "1":
            run_it(ensemble)
        elif choice == "2":
            print(performance_report)
            input("Press any button to return to the menu.")
        elif choice == "3":
            print("Thanks for using our model!")
            run = False
        else:
            input("That was not a valid input, please press any button to return to the menu.")