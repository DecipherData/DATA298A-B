from PIL import Image
import os
import numpy as np

def normalize(source_dir, normalized_dir):
    os.makedirs(normalized_dir, exist_ok=True)

    # Define the target size for resizing (e.g., 224x224)
    target_size = (416, 416)

    # Iterate through each class folder in the source directory
    class_folders = os.listdir(source_dir)

    for class_folder in class_folders:
        class_path = os.path.join(source_dir, class_folder)
        normalized_class_dir = os.path.join(normalized_dir, class_folder)
        os.makedirs(normalized_class_dir, exist_ok=True)

        # Iterate through the images in the class folder
        images = os.listdir(class_path)

        for image_name in images:
            image_path = os.path.join(class_path, image_name)
            normalized_image_path = os.path.join(normalized_class_dir, image_name)

            # Open and resize the image
            with Image.open(image_path) as img:
                # Ensure all images have the same size
                img = img.resize(target_size, Image.BOX)

                # Convert the image to RGB mode (remove alpha channel)
                img = img.convert("RGB")

                # Normalize pixel values to the range [0, 1]
                img = np.array(img) / 255.0

                # Save the normalized and resized image
                normalized_img = Image.fromarray((img * 255).astype(np.uint8))
                normalized_img.save(normalized_image_path)

    print("Image normalization and resizing completed.")



