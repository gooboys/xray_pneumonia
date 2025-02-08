import torch
from torchvision import transforms
from ensemble_model import run_model, CNN

# Check if a GPU is available and set the device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

performance_report = '''
'''

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