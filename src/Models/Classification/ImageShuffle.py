import os
import random
import shutil

# Specify the source folder containing your original images (new_normalized)
source_folder = "../../DATA298-FinalProject/Folder_A"

# Specify the destination folder to store shuffled copies
shuffled_folder = "../../DATA298-FinalProject/shuffled_imagesFolder_A"

# Create the destination folder if it doesn't exist
os.makedirs(shuffled_folder, exist_ok=True)

# List all the image files in the source folder
image_files = [f for f in os.listdir(source_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]

# Shuffle the list of image files
random.shuffle(image_files)

# Copy the shuffled images to the destination folder and rename them
for i, image_file in enumerate(image_files):
    source_path = os.path.join(source_folder, image_file)
    new_image_name = f"shuffled_image_{i+1}{os.path.splitext(image_file)[-1]}"
    destination_path = os.path.join(shuffled_folder, new_image_name)
    shutil.copy(source_path, destination_path)

print("Shuffled copies of images have been created in the 'shuffled_images' folder.")
