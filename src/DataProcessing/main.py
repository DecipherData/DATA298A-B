from DisplayImages import show_sample_images
import matplotlib.pyplot as plt
from ClassStatistics import calculate_image_statistics
from ClassDistribution import check_class_distribution
from Sampling_Bio_Clothes import sample_bio_clothes
from Data_cleaning import clean_corrupt_images
from DeleteFolders import delete_folders, delete_old_sampledfolders_rename_new, rename_folders
from Find_Text_In_Images import remove_text_images_from_classes
from Delete_Duplicate_Images import find_duplicates
from Normalize_the_Dataset import normalize
from ImageVisualizations import visualize2d
from Train_Val_Test_Split import train_val_split

from DataAugmentation import data_augmentation
from MoveAugmentatedFiles import move_augmented_files_to_train




# Set the path to your garbage image dataset directory
dataset_path = "../../DATA298-FinalProject/FinalData"
#train_path = "../../DATA298-FinalProject/train"


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Welcome to {name}')  # Press Ctrl+F8 to toggle the breakpoint.

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('Waste Management and Recommendation System')
    show_sample_images(dataset_path)
    check_class_distribution(dataset_path)

    clean_corrupt_images(dataset_path) # Clean the dataset
    check_class_distribution(dataset_path)

    #Delete specified folders
    delete_folders(["corrupt_images"],dataset_path)

    sample_bio_clothes(dataset_path)
    delete_old_sampledfolders_rename_new(dataset_path) # delete original folders and rename the new subsampled folders
    check_class_distribution(dataset_path)

    #Remove Text Labels from Biowaste

    original_dir = "../../DATA298-FinalProject/FinalData"
    text_images_dir = "../../DATA298-FinalProject/FinalData_images_with_text"
    clean_dir = "../../DATA298-FinalProject/FinalData_class_clean"

    #remove_text_images_from_classes(original_dir,text_images_dir,clean_dir)
    text_cleaned_data = '../../DATA298-FinalProject/FinalData_class_clean'
    check_class_distribution(text_cleaned_data)

    text_data = '../../DATA298-FinalProject/FinalData_images_with_text'
    check_class_distribution(text_data)

    folders_to_delete = ["biowaste"]
    delete_folders(folders_to_delete,dataset_path)
    folders_to_rename = {"biowaste_s_clean": "biowaste"}
    rename_folders(folders_to_rename,dataset_path)
    check_class_distribution(dataset_path)
    calculate_image_statistics(dataset_path)
    check_class_distribution(dataset_path)

    #Remove Duplicate Files
    dataset_path = r"../../DATA298-FinalProject/FinalData"
    cleaned_path = r"../../DATA298-FinalProject/Removed_Duplicates"
    duplicates_path = r"../../DATA298-FinalProject/Duplicates"

    find_duplicates(dataset_path, cleaned_path, duplicates_path)
    check_class_distribution(cleaned_path)

    # Set your source directory with class folders
    source_dir = "../../DATA298-FinalProject/Removed_Duplicates"
    #Create a destination directory for the normalized and resized images
    normalized_dir = "../../DATA298-FinalProject/Removed_Duplicates_normalized"
    visualize2d(source_dir)
    normalize(source_dir,normalized_dir)
    check_class_distribution(normalized_dir)
    calculate_image_statistics(normalized_dir)
    visualize2d(normalized_dir)

    train_dir = "../../DATA298-FinalProject/latestnormalized_train"
    validation_dir = "../../DATA298-FinalProject/latestnormalized_val"
    test_dir = "../../DATA298-FinalProject/latestnormalized_test"

    train_val_split(normalized_dir, train_dir, validation_dir, test_dir)

    check_class_distribution(train_dir)
    check_class_distribution(validation_dir)
    check_class_distribution(test_dir)

    # Path to the original dataset
    train_dataset_path = "../../DATA298-FinalProject/Removed_Duplicates/train_val_test/train"
    augmented_data_path = "../../DATA298-FinalProject/Removed_Duplicates/train_val_test/train_augmentedData"
    data_augmentation(train_dataset_path, augmented_data_path)
    check_class_distribution(train_dataset_path)
    move_augmented_files_to_train(augmented_data_path,train_dataset_path)
    check_class_distribution(train_dataset_path)
