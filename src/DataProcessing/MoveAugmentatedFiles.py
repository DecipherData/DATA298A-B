import shutil
import os

def move_augmented_files_to_train(augmented_data_path,original_dataset_path):
    # Path to the augmented dataset
    # augmented_data_path = "../../DATA298-FinalProject/new_augmentedData"
    # original_dataset_path = "../../DATA298-FinalProject/new_train_normalized"

    # Get the class names from the augmented dataset
    class_names = os.listdir(augmented_data_path)

    # Loop through all class names
    for class_name in class_names:
        print("augmented class name : ", class_name)
        augmented_class_path = os.path.join(augmented_data_path, class_name)
        image_files = os.listdir(augmented_class_path)

        # Path to the corresponding final dataset class
        final_class_path = os.path.join(original_dataset_path, class_name)
        print(" final class path : ", final_class_path)

        # Move augmented images to the respective final dataset class
        for image_file in image_files:
            augmented_image_path = os.path.join(augmented_class_path, image_file)
            final_image_path = os.path.join(final_class_path, image_file)

            # Move the augmented image to the final dataset class
            shutil.copy(augmented_image_path, final_image_path)

    print("Augmented images moved to the respective 'train/class' folders.")


#move_augmented_files_to_train()
