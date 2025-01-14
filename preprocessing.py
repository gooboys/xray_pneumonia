import numpy as np
import pandas as pd
import os
from PIL import Image


def normalize_brightness(image, target_brightness):
    """
    Normalize the brightness of an image to a target brightness level.
    
    Args:
    - image (PIL.Image.Image): Input image.
    - target_brightness (int): The target brightness level.
    
    Returns:
    - PIL.Image.Image: Image with normalized brightness.
    """
    # Convert image to numpy array for processing
    image_array = np.array(image).astype(np.float32)

    # Calculate current brightness and adjust
    current_brightness = np.mean(image_array)
    brightness_offset = target_brightness - current_brightness
    image_normalized = np.clip(image_array + brightness_offset, 0, 255).astype(np.uint8)

    # Convert back to PIL image and return
    return Image.fromarray(image_normalized)

def normalize_contrast(image, target_contrast):
    """
    Normalize the contrast of an image to a target contrast level.
    
    Args:
    - image (PIL.Image.Image): Input image.
    - target_contrast (int): The target contrast level.
    
    Returns:
    - PIL.Image.Image: Image with normalized contrast.
    """
    # Convert image to numpy array for processing
    image_array = np.array(image).astype(np.float32)

    # Calculate current contrast and adjust
    current_contrast = np.std(image_array)
    contrast_scale = target_contrast / current_contrast if current_contrast != 0 else 1
    image_normalized = np.clip((image_array - np.mean(image_array)) * contrast_scale + np.mean(image_array), 0, 255).astype(np.uint8)

    # Convert back to PIL image and return
    return Image.fromarray(image_normalized)

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

def image_path_labeller(positive_train_folder, negative_train_folder, positive_test_folder, negative_test_folder, csv_file):
    # Initialize a list to store data
    image_data = []

    # Mean data for data normalization
    brightness = []
    contrast = []

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
                    # Convert to grayscale
                    img_gray = img.convert('L')
                    img_array = np.array(img_gray)

                    # Calculate statistics
                    mean_pixel_value = np.mean(img_array)
                    stdev_pixel_value = np.std(img_array)
                    brightness.append(mean_pixel_value)
                    contrast.append(stdev_pixel_value)
                    height, width = img_array.shape

                    # Get label based on the image filename
                    label = get_label(image_name)

                    # Append the data
                    image_data.append({
                        "Path": image_path,
                        "Label": label,
                        "Mean": mean_pixel_value,
                        "Stdev": stdev_pixel_value,
                        "Height": height,
                        "Width": width
                    })

            except Exception as e:
                print(f"Error processing image {image_name}: {e}")
                continue  # Skip any files that cannot be opened as images

    # Create a DataFrame
    df = pd.DataFrame(image_data)

    # Write to CSV
    df.to_csv(csv_file, index=False)

    return np.mean(brightness), np.mean(contrast)

def normalize_all(csv_file, target_contrast, target_brightness, normalized_folder, output_csv):
    """
    Normalize images by adjusting contrast and brightness, save them to a folder,
    and create a CSV with the image paths and a second column from the input CSV.
    
    Args:
    - csv_file (str): Path to the input CSV file.
    - target_contrast (float): Target contrast value for normalization.
    - target_brightness (float): Target brightness value for normalization.
    - normalized_folder (str): Folder where normalized images will be saved.
    - output_csv (str): Path where the new CSV file will be saved.
    """
    # Load the input CSV file and extract both columns needed
    df_input = pd.read_csv(csv_file, usecols=[0, 1])
    image_paths = []
    
    # Makes directory if it does not exist
    os.makedirs(normalized_folder, exist_ok=True)

    # Normalize each image
    for count, (img_path, second_column_value) in enumerate(zip(df_input.iloc[:, 0], df_input.iloc[:, 1])):
        with Image.open(img_path) as img:
            # Normalize contrast and brightness
            post_contrast = normalize_contrast(img, target_contrast)
            post_brightness = normalize_brightness(post_contrast, target_brightness)
            # Resize and apply padding
            normalized_image = apply_lanczos_with_padding(post_brightness, size=(256, 256))
            
            # Save the normalized image with a new filename
            saved_file_name = f"{normalized_folder}/image_{count}.jpeg"
            normalized_image.save(saved_file_name, format='JPEG', quality=95)
            
            # Append the path to the list
            image_paths.append(saved_file_name)
    
    # Create and save the new DataFrame with image paths and the second column
    df_output = pd.DataFrame({
        'Image Paths': image_paths,
        'Second Column': df_input.iloc[:, 1]  # Add second column from input
    })
    df_output.to_csv(output_csv, index=False)
    print(f"Normalization complete. CSV file created at {output_csv}")
    return


# Define your folder paths
positive_train_folder = 'ChestXRay2017/train/PNEUMONIA'
negative_train_folder = 'ChestXRay2017/train/NORMAL'
positive_test_folder = 'ChestXRay2017/test/PNEUMONIA'
negative_test_folder = 'ChestXRay2017/test/NORMAL'
csv_file = 'Labeled_Image_Paths.csv'
normalized_folder = 'NormalizedXRays'
normalized_csv = 'Normalized_Image_Paths.csv'

no_change_folder = 'NormalData'
no_change_csv = 'Regular_Image_paths.csv'

brightness_mean, contrast_mean = image_path_labeller(positive_train_folder, negative_train_folder, positive_test_folder, negative_test_folder, csv_file)
normalize_all(csv_file, contrast_mean, brightness_mean, normalized_folder,normalized_csv)
