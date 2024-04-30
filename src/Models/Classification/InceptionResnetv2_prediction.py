import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from keras.models import load_model
import pandas as pd
import random



# Load the saved model
model = load_model(r'C:\Users\bhati\PycharmProjects\WasteManagement\best_inceptionresnetv2_model.h5')

# Load the test data
test_dir = r'C:\Users\bhati\DATA298-FinalProject\Removed_Duplicates\train_val_test\test'

# Image size and batch size
image_size = (299, 299)
batch_size = 32

# Create a test data generator
test_datagen = ImageDataGenerator(preprocessing_function=keras.applications.inception_resnet_v2.preprocess_input)
test_generator = test_datagen.flow_from_directory(
    test_dir, target_size=image_size, batch_size=batch_size, class_mode='categorical', shuffle=False
)

# Evaluate the model on the test data
# test_loss, test_accuracy = model.evaluate(test_generator)
# print(f"Test Loss: {test_loss}")
# print(f"Test Accuracy: {test_accuracy}")

# Predictions and classification report
test_generator.reset()
predictions = model.predict(test_generator)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = test_generator.classes
class_labels = list(test_generator.class_indices.keys())

# Print the classification report
report = classification_report(true_classes, predicted_classes, target_names=class_labels)
print(report)

# Print the confusion matrix
cm = confusion_matrix(true_classes, predicted_classes)
print(cm)


# Plot the confusion matrix using Seaborn
plt.figure(figsize=(12, 8))
cm_df = pd.DataFrame(cm, index=class_labels, columns=class_labels)
sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()


# Display images from each class with true and predicted labels
#num_classes = len(class_labels)
#num_rows = (num_classes + 1) // 4  # Adjusted number of rows based on the number of classes

num_classes = len(class_labels)
num_rows = 4
num_columns = 3

# Increase figure size to accommodate more subplots
plt.figure(figsize=(15, 15))

for class_name in class_labels:
    # Get a list of all images in the current class folder
    class_images = [filename for filename in test_generator.filenames if class_name in filename]

    # Select a random image from the current class
    random_image = random.choice(class_images)
    img_path = os.path.join(test_dir, random_image)

    # Load and preprocess the image
    img = load_img(img_path, target_size=image_size)
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = keras.applications.inception_resnet_v2.preprocess_input(img_array)

    # Get true and predicted labels
    true_label = class_labels[true_classes[test_generator.filenames.index(random_image)]]
    predicted_label = class_labels[predicted_classes[test_generator.filenames.index(random_image)]]

    # Determine subplot position dynamically
    subplot_position = class_labels.index(class_name) + 1

    # Display the image and labels
    plt.subplot(num_rows, num_columns, subplot_position)
    plt.imshow(img_array[0])
    plt.title(f'True: {true_label}\nPredicted: {predicted_label}')
    plt.axis('off')

plt.tight_layout()  # Adjust layout for better spacing
plt.show()

# # Display 11 images (1 from each class) with true and predicted labels
# classes_to_display = set()
#
# plt.figure(figsize=(8, 8))
#
# for i in range(len(test_generator.filenames)):
#     if len(classes_to_display) == len(class_labels):
#         break  # Stop once you have one image from each class
#
#     img_path = os.path.join(test_dir, test_generator.filenames[i])
#
#     # Check if the class of the current image has already been displayed
#     class_name = os.path.dirname(test_generator.filenames[i])
#     if class_name not in classes_to_display:
#         img = load_img(img_path, target_size=image_size)
#         img_array = img_to_array(img)
#         img_array = np.expand_dims(img_array, axis=0)
#         img_array = keras.applications.inception_resnet_v2.preprocess_input(img_array)
#
#         true_label = class_labels[true_classes[i]]
#         predicted_label = class_labels[predicted_classes[i]]
#
#         plt.subplot(6, 2, len(classes_to_display) * 2 + 1)  # Adjust the subplot accordingly
#         plt.imshow(img_array[0])
#         plt.title('Image')
#         plt.axis('off')
#
#         plt.subplot(6, 2, len(classes_to_display) * 2 + 2)  # Adjust the subplot accordingly
#         plt.text(0.5, 0.5, f'True: {true_label}\nPredicted: {predicted_label}', ha='center', va='center', wrap=True)
#         plt.axis('off')
#
#         classes_to_display.add(class_name)
#
# plt.show()
