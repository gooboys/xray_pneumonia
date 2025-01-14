import os
from PIL import Image, ImageOps
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd


# Average brightness, Standard deviation (both individually per file and for all), Contrast/Dynamic range, Median brightness

# Function to calculate brightness
def calculate_brightness(image):
    grayscale_image = image.convert("L")  # Convert to grayscale
    np_image = np.array(grayscale_image)
    brightness = np.mean(np_image)  # Average pixel value represents brightness
    return brightness

# Calculate the standard deviation of the contrast
def calculate_contrast(image):
    grayscale_image = image.convert("L") # Convert to grayscale
    contrast = np.std(grayscale_image)
    return contrast

# Iterates through all the files in a folder to run different functions for stats
def run_for_stats(folder_path):
    brightness_values = []
    contrast_values = []
    x = []
    y = []
    for root, dirs, files in os.walk(folder_path):
        for filename in files:
            if filename.endswith(".jpeg"):
                #print("ran")
                file_path = os.path.join(root, filename)
                with Image.open(file_path) as img:
                    brightness = calculate_brightness(img)
                    brightness_values.append(brightness)
                    contrast = calculate_contrast(img)
                    contrast_values.append(contrast)
                    width, height = img.size
                    x.append(width)
                    y.append(height)
    return brightness_values, contrast_values, x, y                

def box_whiskers(data, title, y_axis):
    """
    Plot a box and whisker plot without any scatter plot of data points.
    
    Args:
    - data (array-like): A list or array of numerical data.
    """
    # Create a figure and axis
    plt.figure(figsize=(8, 6))
    
    # Create the boxplot
    plt.boxplot(data, positions=[1], widths=0.5, patch_artist=True, 
                boxprops=dict(facecolor='skyblue', color='black'),
                whiskerprops=dict(color='black'),
                flierprops=dict(markerfacecolor='red', marker='o', markersize=6))
    
    # Adding title and labels
    plt.title(title)
    plt.xlabel("Data Set")
    plt.ylabel(y_axis)
    
    # Show the plot
    plt.show()

# Makes a scatterplot (for the size of the images)
def scatter(x_data, y_data, title, y_name, x_name):
    """
    Plots a scatter plot using x_data for the x-axis and y_data for the y-axis.
    
    Args:
    - x_data (array-like): Data for the x-axis.
    - y_data (array-like): Data for the y-axis.
    """
    plt.figure(figsize=(8, 6))
    plt.scatter(x_data, y_data, color='blue', alpha=0.6, marker='o')
    plt.title(title)
    plt.xlabel(x_name)
    plt.ylabel(y_name)
    plt.grid(True)
    plt.show()
    return

def run_all(path):
    brightness_values, contrast_values, x, y = run_for_stats(path)
    box_whiskers(brightness_values, "Box and Whisker Plot of Mean Brightness", "Brightness")
    box_whiskers(contrast_values, "Box and Whisker Plot of Contrast Standard Deviation", "Contrast Values")
    scatter(x,y,"Scatter plot of image sizes", "height", "width")
    return

def countClasses(folder_path):
    normal = 0
    viral = 0
    bacterial = 0
    for root, dirs, files in os.walk(folder_path):
        for filename in files:
            if filename.endswith(".jpeg"):
                if 'virus' in filename:
                    viral += 1
                elif 'bacteria' in filename:
                    bacterial += 1
                else:
                    normal += 1
    classes = ['Normal', 'Viral Pneumonia', 'Bacterial Pneumonia']
    counts = [normal, viral, bacterial]

    # Create the bar graph
    plt.figure(figsize=(8, 6))
    plt.bar(classes, counts, color=['blue', 'orange', 'green'])
    plt.title('Distribution of X-ray Classes')
    plt.xlabel('Class')
    plt.ylabel('Number of Images')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()
    return

path = 'ChestXRay2017'

countClasses(path)
run_all(path)