import os
import random
import cv2

def data_augmentation(original_dataset_path, augmented_data_path):

    # Set a random seed for reproducibility
    random.seed(1)

    # Get the class names from the folder names in the original dataset path
    class_names = [folder_name for folder_name in os.listdir(original_dataset_path) if os.path.isdir(os.path.join(original_dataset_path, folder_name))]

    # Number of images you want in each class after augmentation
    target_images = 2500

    # Define the apply_augmentation function
    def apply_augmentation(image, class_name):
        if class_name == "cardboard":
            # Apply augmentation specific to the "cardboard" class
            augmented_image = cv2.flip(image, 1)  # Example: Flip horizontally
        elif class_name == "e-waste":
            # Apply augmentation specific to the "e-waste" class
            augmented_image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)  # Example: Rotate 90 degrees
        elif class_name == "furniture":
            # Apply augmentation specific to the "furniture" class
            augmented_image = cv2.blur(image, (5, 5))  # Example: Apply Gaussian blur
        elif class_name == "glass":
            # Apply augmentation specific to the "glass" class
            zoom_factor = random.uniform(1.1, 1.5)  # Zoom between 110% to 150%
            augmented_image = cv2.resize(image, None, fx=zoom_factor, fy=zoom_factor)
        # Add more elif blocks for other classes and their respective augmentation techniques
        else:
            # Default augmentation for unknown classes
            augmented_image = cv2.flip(image, 1)  # Example: Flip horizontally

        return augmented_image

    # Loop through all class names
    for class_name in class_names:
        # Path to the original class folder
        class_path = os.path.join(original_dataset_path, class_name)
        image_files = os.listdir(class_path)
        num_original_images = len(image_files)

        # Number of augmented images to create for this class
        augmented_images = target_images - num_original_images

        # Create a folder for augmented images for this class
        augmented_dataset_path = os.path.join(augmented_data_path, class_name)
        os.makedirs(augmented_dataset_path, exist_ok=True)

        # Shuffle the list of image files for randomness
        random.shuffle(image_files)

        for i in range(augmented_images):
            original_image_file = image_files[i % num_original_images]  # Cycle through the available images
            original_image_path = os.path.join(class_path, original_image_file)

            # Load the original image
            original_image = cv2.imread(original_image_path)

            # Apply the augmentation
            augmented_image = apply_augmentation(original_image, class_name)

            # Create a new file name
            augmented_image_file = f"{i + 1}_{original_image_file}"

            # Save the augmented image in the augmentedData folder
            augmented_image_path = os.path.join(augmented_dataset_path, augmented_image_file)
            cv2.imwrite(augmented_image_path, augmented_image)

    print("Augmented images moved to the 'new_augmentedData' folder.")



