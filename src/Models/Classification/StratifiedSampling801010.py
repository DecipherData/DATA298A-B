import os
import random
import shutil

def stratified_sampling(source_dir, destination_dir, sample_proportion):
    # Get a list of class folders
    class_folders = os.listdir(source_dir)

    # Create class folders within the destination directory
    for class_folder in class_folders:
        class_path = os.path.join(source_dir, class_folder)
        images = os.listdir(class_path)
        num_images = len(images)

        # Calculate the number of samples to take based on sample_proportion
        num_samples = int(sample_proportion * num_images)

        # Randomly sample images and copy to the destination directory
        sampled_images = random.sample(images, num_samples)
        for image in sampled_images:
            source_path = os.path.join(class_path, image)
            destination_path = os.path.join(destination_dir, class_folder, image)
            os.makedirs(os.path.dirname(destination_path), exist_ok=True)
            shutil.copy(source_path, destination_path)

# Define source and destination directories
source_dirs = [
    ("../../DATA298-FinalProject/test_splits/before_annotation/train_augmented", "../../DATA298-FinalProject/test_splits/sampled_for_annotation/train_sample"),
    ("../../DATA298-FinalProject/test_splits/before_annotation/val_normalized", "../../DATA298-FinalProject/test_splits/sampled_for_annotation/val_sample"),
    ("../../DATA298-FinalProject/test_splits/before_annotation/test_normalized", "../../DATA298-FinalProject/test_splits/sampled_for_annotation/test_sample")
]

# Define the sampling proportions
sampling_proportions = [0.20, 0.20, 0.20]  # 20% for train, 10% for val, 10% for test

# Perform stratified sampling for each source and destination directory
for source_dir, destination_dir in source_dirs:
    sampling_proportion = sampling_proportions[source_dirs.index((source_dir, destination_dir))]
    stratified_sampling(source_dir, destination_dir, sampling_proportion)
