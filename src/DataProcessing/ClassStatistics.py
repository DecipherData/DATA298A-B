import os
import cv2
import matplotlib.pyplot as plt
import numpy as np


def calculate_image_statistics(dataset_path):
    classes = os.listdir(dataset_path)
    class_stats = {}  # Dictionary to store class statistics

    for class_name in classes:
        class_path = os.path.join(dataset_path, class_name)
        images = os.listdir(class_path)

        mean_pixel_values = []
        std_deviations = []
        mean_width = []
        mean_height = []
        mean_channels = []

        for image_name in images:
            img_path = os.path.join(class_path, image_name)
            img = cv2.imread(img_path)

            if img is not None:
                height, width, channels = img.shape
                mean_pixel_value = np.mean(img)
                std_deviation = np.std(img)

                mean_pixel_values.append(mean_pixel_value)
                std_deviations.append(std_deviation)
                mean_width.append(width)
                mean_height.append(height)
                mean_channels.append(channels)
            else:
                print(f"Corrupt Image: {image_name} in class {class_name}")
                os.remove(img_path)
                print(f"Image : {image_name} in class {class_name} is removed.")

        class_stats[class_name] = {
            "Mean Pixel Values": mean_pixel_values,
            "Standard Deviations": std_deviations,
            "Mean Width": mean_width,
            "Mean Height": mean_height,
            "Mean Channels": mean_channels
        }

    # Plot the statistics
    plot_image_statistics(class_stats)

def plot_image_statistics(class_stats):
    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(12, 12))

    for idx, (stat_name, title) in enumerate(
        [("Mean Pixel Values", "Mean Pixel Value Distribution"),
         ("Standard Deviations", "Standard Deviation Distribution"),
         ("Mean Width", "Mean Width Distribution"),
         ("Mean Height", "Mean Height Distribution"),
         ("Mean Channels", "Mean Channels Distribution")]
    ):
        ax = axes[idx // 2, idx % 2]
        data = [class_stats[class_name][stat_name] for class_name in class_stats]

        ax.boxplot(data)
        ax.set_xticklabels(class_stats.keys(), rotation=40)
        ax.set_title(title)
        ax.set_ylabel(stat_name)

    # Remove any empty subplots
    for i in range(5, 6):
        fig.delaxes(axes.flatten()[i])

    plt.tight_layout()
    plt.show()

