import matplotlib.pyplot as plt
import shutil
import os
import cv2
import numpy as np
from pathlib import Path
import imghdr

corrupt_images = []
image_extensions = [".png", ".jpg", ".jpeg", "bmp"]  # add there all your images file extensions

def is_valid_image(file_path):
    img_type_accepted_by_tf = ["bmp", "gif", "jpeg", "png"]
    for file_name in os.listdir(file_path):
        if file_name.lower().endswith(tuple(image_extensions)):
            img_path = os.path.join(file_path, file_name)
            img_type = imghdr.what(img_path)
            if img_type is None:
                print(f"{file_name} is not an image")
                corrupt_images.append(file_name)
            elif img_type not in img_type_accepted_by_tf:
                print(f"{file_name} is a {img_type}, not accepted by TensorFlow")
                corrupt_images.append(file_name)

# Function to calculate basic image statistics
def clean_corrupt_images(dataset_path):

    classes = os.listdir(dataset_path)
    #
    for class_name in classes:
        class_path = os.path.join(dataset_path, class_name)
        is_valid_image(class_path)
    print(corrupt_images)
    check_corrupt_images(corrupt_images,dataset_path)

def check_corrupt_images(corrupt_images,dataset_path):
    # Create the 'corrupt_images' folder if it doesn't exist
    corrupt_folder_path = os.path.join(dataset_path, 'corrupt_images')
    os.makedirs(corrupt_folder_path, exist_ok=True)

    # Iterate through each category folder
    for category_folder in os.listdir(dataset_path):
        category_folder_path = os.path.join(dataset_path, category_folder)

        # Check if the current folder is a directory
        if os.path.isdir(category_folder_path):
            for image_filename in os.listdir(category_folder_path):
                if image_filename in corrupt_images:
                    # Move the corrupt image to the 'corrupt_images' folder
                    source_path = os.path.join(category_folder_path, image_filename)
                    destination_path = os.path.join(corrupt_folder_path, image_filename)
                    shutil.move(source_path, destination_path)
                    print(f"{image_filename} is moved to corrupt_images folder")

    #print("Corrupt images have been moved to the 'corrupt_images' folder.")

