import matplotlib.pyplot as plt
import os

# Function to check the class distribution
def check_class_distribution(dataset_path):
    classes = os.listdir(dataset_path)
    class_counts = {}

    for class_name in classes:
        class_path = os.path.join(dataset_path, class_name)
        images = os.listdir(class_path)
        class_counts[class_name] = len(images)

    print("Class Distribution:")
    for class_name, count in class_counts.items():
        print(f"{class_name}: {count} images")

    # Extract class names and counts
    class_names = list(class_counts.keys())
    class_counts = list(class_counts.values())
    # Create a bar chart
    plt.figure(figsize=(12, 6))
    plt.bar(class_names, class_counts, color='skyblue')
    plt.xlabel('Class Name')
    plt.ylabel('Number of Images')
    plt.title('Class Distribution')
    plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better visibility

    # Show the chart
    plt.tight_layout()
    plt.show()

