## This file samples 950 images and labels and put them in data/train/images and labels
## sample 100 images and labels for validation and 50 for test

import os
import random
import shutil

# Set the paths
images_folder = r'C:\Users\bhati\DATA298-FinalProject\YOLO\images'
obj_train_data_folder = r'C:\Users\bhati\DATA298-FinalProject\YOLO\obj_train_data'
data_folder = r'C:\Users\bhati\DATA298-FinalProject\YOLO\data'

# Create train, val, and test directories
for folder in [  'test']: #'train','val',
    os.makedirs(os.path.join(data_folder, folder, 'images'), exist_ok=True)
    os.makedirs(os.path.join(data_folder, folder, 'labels'), exist_ok=True)

# List all image files in the images folder
image_files = os.listdir(images_folder)

# Randomly sample 950 images for train
# train_images = random.sample(image_files, 950)
# for image in train_images:
#     # Move image to train/images
#     shutil.move(os.path.join(images_folder, image), os.path.join(data_folder, 'train', 'images', image))
#
#     # Move corresponding text file to train/labels
#     txt_file = os.path.splitext(image)[0] + '.txt'
#     shutil.move(os.path.join(obj_train_data_folder, txt_file), os.path.join(data_folder, 'train', 'labels', txt_file))

# Randomly sample 100 images for val
# val_images = random.sample(image_files, 100)
# for image in val_images:
#     # Move image to val/images
#     shutil.move(os.path.join(images_folder, image), os.path.join(data_folder, 'val', 'images', image))
#
#     # Move corresponding text file to val/labels
#     txt_file = os.path.splitext(image)[0] + '.txt'
#     shutil.move(os.path.join(obj_train_data_folder, txt_file), os.path.join(data_folder, 'val', 'labels', txt_file))

# Move the remaining 50 images to test
#test_images = list(set(image_files) - set(train_images) - set(val_images))
test_images = random.sample(image_files, 50)

for image in test_images:
    # Move image to test/images
    shutil.move(os.path.join(images_folder, image), os.path.join(data_folder, 'test', 'images', image))

    # Move corresponding text file to test/labels
    txt_file = os.path.splitext(image)[0] + '.txt'
    shutil.move(os.path.join(obj_train_data_folder, txt_file), os.path.join(data_folder, 'test', 'labels', txt_file))

