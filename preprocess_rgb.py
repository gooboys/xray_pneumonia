import numpy as np
import pandas as pd
import os
from PIL import Image


def apply_lanczos_with_padding(image, size=(256, 256)):
    # Open the original image in grayscale
    #image = Image.open(input_image_path).convert('L')
    
    # Get the original image size
    original_width, original_height = image.size
    
    # Calculate the aspect ratio
    aspect_ratio = original_width / original_height
    
    # Determine the new dimensions while maintaining the aspect ratio
    if original_width > original_height:
        new_width = size[0]
        new_height = int(size[0] / aspect_ratio)
    else:
        new_height = size[1]
        new_width = int(size[1] * aspect_ratio)
    
    # Resize the image using LANCZOS interpolation
    resized_image = image.resize((new_width, new_height), Image.LANCZOS)
    
    # Create a new image with the target size and black background (padding)
    new_image = Image.new('L', size, color=0)  # 'L' for grayscale, color=0 is black
    
    # Paste the resized image onto the center of the new image
    top = (size[1] - new_height) // 2
    left = (size[0] - new_width) // 2
    new_image.paste(resized_image, (left, top))

    return new_image

def process_images(positive_train_folder, negative_train_folder, positive_test_folder, negative_test_folder, csv_file, normalized_folder):
    # Initialize a list to store data
    image_data = []

    os.makedirs(normalized_folder, exist_ok=True)

    # Function to extract label based on the image filename
    def get_label(image_filename):
        if "virus" in image_filename.lower():
            return 2  # Virus
        elif "bacteria" in image_filename.lower():
            return 1  # Bacteria
        else:
            return 0  # Normal

    # Function to check if the file is an image
    def is_image(filename):
        valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff']
        return any(filename.lower().endswith(ext) for ext in valid_extensions)

    # Process all folders
    folders = [positive_train_folder, negative_train_folder, positive_test_folder, negative_test_folder]

    count = 0

    for folder in folders:
        for image_name in os.listdir(folder):
            # Construct image path
            image_path = os.path.join(folder, image_name)
            
            # Check if file is an image based on extension
            if not is_image(image_name):
                continue  # Skip non-image files

            try:
                # Open image
                with Image.open(image_path) as img:
                    normalized_image = apply_lanczos_with_padding(img, size=(224, 224))
                    
                    # Get label based on the image filename
                    label = get_label(image_name)
            
                    # Save the normalized image with a new filename
                    count += 1
                    saved_file_name = f"{normalized_folder}/image_{count}.jpeg"
                    normalized_image.save(saved_file_name, format='JPEG', quality=95)

                    # Append the data
                    image_data.append({
                        "Paths": saved_file_name,
                        "Labels": label,
                    })
            

            except Exception as e:
                print(f"Error processing image {image_name}: {e}")
                continue  # Skip any files that cannot be opened as images

    # Create a DataFrame
    df = pd.DataFrame(image_data)

    # Write to CSV
    df.to_csv(csv_file, index=False)
    return

# Define your folder paths
positive_train_folder = 'ChestXRay2017/train/PNEUMONIA'
negative_train_folder = 'ChestXRay2017/train/NORMAL'
positive_test_folder = 'ChestXRay2017/test/PNEUMONIA'
negative_test_folder = 'ChestXRay2017/test/NORMAL'

folder = 'RGBImages'
csv = 'RGBLabels.csv'

process_images(positive_train_folder, negative_train_folder, positive_test_folder, negative_test_folder,csv,folder)