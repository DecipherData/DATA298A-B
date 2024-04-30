import os
import random
import shutil

def train_val_split(source_dir, train_dir, validation_dir, test_dir):
    # Define your source directory (with 11 class folders)
    #source_dir = "../../DATA298-FinalProject/FinalD"
    #source_dir = "../../DATA298-FinalProject/new_sub200/new_subsampled200"


    #train_dir = "../../DATA298-FinalProject/new_sub200/new_train_normalizedsubmodel3"
    #validation_dir = "../../DATA298-FinalProject/val_normalized"
    #test_dir = "../../DATA298-FinalProject/new_sub200/new_test_normalizedsubmodel3"

    # Create train, validation, and test directories if they don't exist
    #for directory in [train_dir, validation_dir, test_dir]:
    for directory in [train_dir, validation_dir, test_dir]:
        os.makedirs(directory, exist_ok=True)

    # Set the proportions for the train, validation, and test sets
    train_proportion = 0.8
    validation_proportion = 0.10
    test_proportion = 0.10

    # Get a list of class folders
    class_folders = os.listdir(source_dir)

    # Iterate through each class folder
    for class_folder in class_folders:
        class_path = os.path.join(source_dir, class_folder)
        images = os.listdir(class_path)
        num_images = len(images)
        random.shuffle(images)

        # Create class folders within train, validation, and test directories
        #for directory in [train_dir, validation_dir, test_dir]:
        for directory in [train_dir, validation_dir, test_dir]:
            class_directory = os.path.join(directory, class_folder)
            os.makedirs(class_directory, exist_ok=True)

        train_split = int(train_proportion * num_images)
        validation_split = int((train_proportion + validation_proportion) * num_images)

        for i, image in enumerate(images):
            source_path = os.path.join(class_path, image)
            if i < train_split:
                destination_path = os.path.join(train_dir, class_folder, image)
            elif i < validation_split:
                destination_path = os.path.join(validation_dir, class_folder, image)
            else:
                destination_path = os.path.join(test_dir, class_folder, image)

            # Copy the image to the appropriate directory
            shutil.copy(source_path, destination_path)

#train_val_split((source_dir, train_dir, validation_dir, test_dir))
