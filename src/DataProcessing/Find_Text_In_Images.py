from PIL import Image
import pytesseract
import os
import shutil

# Set the tesseract path if it's not in your system's PATH
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def remove_text_images_from_classes(original_dir,text_images_dir,clean_dir):
    # Function to check if an image contains text
    def contains_text(image_path):
        try:
            # Open an image file
            with Image.open(image_path) as img:
                # Use pytesseract to extract text from the image
                text = pytesseract.image_to_string(img)

                # Check if the extracted text is non-empty
                if text.strip():
                    return True
                else:
                    return False
        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")
            return False

    # # Specify the directory containing the original images
    # original_dir = "../../DATA298-FinalProject/FinalData"
    # text_images_dir = "../../DATA298-FinalProject/FinalData_images_with_text"
    # clean_dir = "../../DATA298-FinalProject/FinalData_class_clean"

    # Create the text_images_dir and clean_dir if they don't exist
    if not os.path.exists(text_images_dir):
        os.makedirs(text_images_dir)
    if not os.path.exists(clean_dir):
        os.makedirs(clean_dir)

    # List of class folders
    class_folders = os.listdir(original_dir)

    for class_folder in class_folders:
        class_dir = os.path.join(original_dir, class_folder)
        text_images_class_dir = os.path.join(text_images_dir, class_folder)
        clean_class_dir = os.path.join(clean_dir, class_folder)

        # Create class-specific text_images and clean folders
        if not os.path.exists(text_images_class_dir):
            os.makedirs(text_images_class_dir)
        if not os.path.exists(clean_class_dir):
            os.makedirs(clean_class_dir)

        # Iterate over the images in the class directory
        for file_name in os.listdir(class_dir):
            file_path = os.path.join(class_dir, file_name)

            # Check if the file contains text
            if contains_text(file_path):
                # If it contains text, move it to the class-specific "images_with_text" directory
                shutil.copy(file_path, os.path.join(text_images_class_dir, file_name))
                print(f"Copied {file_name} to {text_images_class_dir}.")
            else:
                # If it doesn't contain text, copy it to the class-specific "class_clean" directory
                shutil.copy(file_path, os.path.join(clean_class_dir, file_name))
                print(f"Copied {file_name} to {clean_class_dir}.")

        print(f"Processed class: {class_folder}")

    print("Image processing completed for all classes.")

#remove_text_images_from_classes()
