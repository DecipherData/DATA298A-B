import os
import random
import shutil


def sample_bio_clothes(dataset_path):
    # Set the number of images to sample
    num_samples = 2500
    # Create directories to store the sampled images
    sampled_clothes_dir = "../../DATA298-FinalProject/FinalData/sampled_clothes"
    sampled_biowaste_dir = "../../DATA298-FinalProject/FinalData/sampled_biowaste"
    # Function to randomly sample images from a category folder

    def sample_images_from_category(category_folder, num_samples):
        images = [f for f in os.listdir(category_folder) if f.endswith(".jpg")]
        sampled_images = random.sample(images, min(num_samples, len(images)))
        return sampled_images

    # Iterate through the specified categories and sample images
    categories_to_sample = {
        "clothes": sampled_clothes_dir,
        "biowaste": sampled_biowaste_dir
    }
    isClothesDone = False

    for category, output_dir in categories_to_sample.items():
        category_folder_path = os.path.join(dataset_path, category)
        image_files = os.listdir(category_folder_path)
        srclength = len(image_files)
        # Check if the source class folder has more than 2500 images
        if srclength > num_samples:
            # Check if the source class folder exists
            if os.path.exists(sampled_clothes_dir) and (isClothesDone is False):
                # Delete the existing source class folder
                shutil.rmtree(sampled_clothes_dir)
            # Check if the source class folder exists
            if os.path.exists(sampled_biowaste_dir):
                # Delete the existing source class folder
                shutil.rmtree(sampled_biowaste_dir)
            if isClothesDone is False:
                os.makedirs(sampled_clothes_dir, exist_ok=True)
            os.makedirs(sampled_biowaste_dir, exist_ok=True)

            # Check if the current folder is a directory
            if os.path.isdir(category_folder_path):
                sampled_images = sample_images_from_category(category_folder_path, num_samples)
                # Copy the sampled images to the new directory
                for image in sampled_images:
                    src_path = os.path.join(category_folder_path, image)
                    dest_path = os.path.join(output_dir, image)
                    shutil.copy(src_path, dest_path)
                isClothesDone = True
                print(f"Sampled and copied {num_samples} images to the specified folders {output_dir}.")
        else:
            print("No need to Subsample")

#dataset_path = "../../DATA298-FinalProject/Original"
#sample_bio_clothes(dataset_path)
