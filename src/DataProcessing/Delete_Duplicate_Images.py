from PIL import Image
import os
import imagehash
import shutil

def find_duplicates(dataset_path, cleaned_path, duplicates_path):
    hash_dict = {}
    duplicate_info = {}

    # Create the cleaned and duplicates directories if they don't exist
    os.makedirs(cleaned_path, exist_ok=True)
    os.makedirs(duplicates_path, exist_ok=True)

    for class_folder in os.listdir(dataset_path):
        class_path = os.path.join(dataset_path, class_folder)
        class_cleaned = os.path.join(cleaned_path, class_folder)
        class_duplicates = os.path.join(duplicates_path, class_folder)

        # Create class folders in cleaned and duplicates directories
        os.makedirs(class_cleaned, exist_ok=True)
        os.makedirs(class_duplicates, exist_ok=True)

        for dirpath, dirnames, filenames in os.walk(class_path):
            for filename in filenames:
                image_path = os.path.join(dirpath, filename)

                try:
                    # Open the image using Pillow
                    with Image.open(image_path) as img:
                        # Compute the perceptual hash of the image
                        phash = str(imagehash.phash(img))

                        if phash in hash_dict:
                            # Duplicate found
                            duplicate_info.setdefault(hash_dict[phash], []).append(image_path)
                        else:
                            # Add the hash to the dictionary and copy the image to the cleaned folder
                            hash_dict[phash] = image_path
                            shutil.copy(image_path, os.path.join(class_cleaned, filename))
                except Exception as e:
                    # Handle exceptions
                    print(f"Error processing {image_path}: {str(e)}")

    # Move duplicate images to the duplicate folder
    for duplicate_set in duplicate_info.values():
        for duplicate_image in duplicate_set:
            shutil.copy(duplicate_image, os.path.join(class_duplicates, os.path.basename(duplicate_image)))

    # Print duplicate image pairs
    for duplicate_set in duplicate_info.values():
        if len(duplicate_set) > 1:
            print(f"Duplicate images: {duplicate_set}")
