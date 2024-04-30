import matplotlib.pyplot as plt
import os
import cv2
import numpy as np

# This function displays 5 samples chosen randomly
def show_sample_images(dataset_path, num_samples=5):

    classes = os.listdir(dataset_path)

    plt.figure(figsize=(12, 6))

    for i, class_name in enumerate(classes):
        class_path = os.path.join(dataset_path, class_name)
        images = os.listdir(class_path)
        print(images)
        if len(images) > 0:
            for j in range(num_samples):
                random_image = np.random.choice(images)
                img_path = os.path.join(class_path, random_image)

                img = cv2.imread(img_path)
                if img is not None:

                    # Image loaded successfully, proceed with processing
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                    plt.subplot(num_samples, len(classes), i * num_samples + j + 1)
                    plt.imshow(img)
                    plt.title(class_name)
                    plt.axis('off')
                else:
                    print(f"Failed to load image from path: {img_path}")
        else:
            print("Empty Class Folder")
    plt.show()

#show_sample_images(dataset_path)

