import os
import shutil

# Function to delete folders
def delete_folders(folders, dataset_path):
    for folder in folders:
        folder_path = os.path.join(dataset_path, folder)
        if os.path.exists(folder_path):
            shutil.rmtree(folder_path)
            print(f"Deleted folder: {folder}")
        else:
            print(f"Folder does not exist: {folder}")

# Function to rename folders
def rename_folders(folders_to_rename, dataset_path):
    for old_name, new_name in folders_to_rename.items():
        old_path = os.path.join(dataset_path, old_name)
        new_path = os.path.join(dataset_path, new_name)
        if os.path.exists(old_path):
            os.rename(old_path, new_path)
            print(f"Renamed folder: {old_name} to {new_name}")
        else:
            print(f"Folder does not exist: {old_name}")

def delete_old_sampledfolders_rename_new(dataset_path):
    # Define the folders to delete and rename
    folders_to_delete = ["clothes", "biowaste"]
    folders_to_rename = {"sampled_clothes": "clothes", "sampled_biowaste": "biowaste"}
    # Delete specified folders
    delete_folders(folders_to_delete,dataset_path)
    # Rename sampled folders
    rename_folders(folders_to_rename,dataset_path)


#delete_old_sampledfolders_rename_new(dataset_path)

# Function to delete folders
#def delete_folders_corrupt(dataset_path):
#    folders_to_delete_corrupt = ["corrupt_images"]

#delete_old_sampledfolders_rename_new(dataset_path)
