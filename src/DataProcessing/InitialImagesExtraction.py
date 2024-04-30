import os
import shutil

# Define the base paths
dest_base = r'C:\Users\bhati\PycharmProjects\WasteToWow\Data'
source_base = r'C:\Users\bhati\PycharmProjects\WasteToWow\Data\ExternalDatasets'
destination_base = os.path.join(dest_base, 'FinalData')

# List of folders and their corresponding sources
classes = {
    'bio-waste': [os.path.join(source_base, 'Waste Classification Dataset\\Waste Classification Dataset\\waste_dataset\\organic')],
    'clothes': [os.path.join(source_base, 'garbage_classification\\clothes')],
    'shoes': [os.path.join(source_base, 'garbage_classification\\shoes')],
    'cardboard': [os.path.join(source_base, 'TrashBox_train_set\\cardboard')],
    'paper': [os.path.join(source_base, 'TrashBox_train_set\\paper')],
    'plastic': [os.path.join(source_base, 'TrashBox_train_set\\plastic')],
    'metal': [os.path.join(source_base, 'TrashBox_train_set\\metal')],
    'e-waste': [os.path.join(source_base, 'TrashBox_train_set\\e-waste')],
    'medical': [os.path.join(source_base, 'TrashBox_train_set\\medical')],
    'glass': [os.path.join(source_base, 'TrashBox_train_set\\glass')],
    'furniture': [
        os.path.join(source_base, 'Furniture\\furniture.v2-release.multiclass\\test'),
        os.path.join(source_base, 'Furniture\\furniture.v2-release.multiclass\\train'),
        ]
}

# Create destination folders
for class_name in classes:
    os.makedirs(os.path.join(destination_base, class_name), exist_ok=True)

# Function to copy images, ignoring non-image files
def copy_images(src, dest):
    for item in os.listdir(src):
        if item.lower().endswith(('.png', '.jpg', '.jpeg')):
            shutil.copy(os.path.join(src, item), dest)

# Perform the copying process
for class_name, paths in classes.items():
    for path in paths:
        copy_images(path, os.path.join(destination_base, class_name))
