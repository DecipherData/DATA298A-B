import os
import ultralytics
import glob
from IPython.display import Image, display
from ultralytics import YOLO

print(ultralytics.checks())

HOME = os.getcwd()
print(HOME)

# Define the path to the root folder of the dataset
base_dir = r'C:\Users\bhati\DATA298-FinalProject\YOLO_PyCharm\train_val_test'
train_dir = os.path.join(base_dir, 'train')
val_dir = os.path.join(base_dir, 'val')
test_dir = os.path.join(base_dir, 'test')

#!yolo task=classify mode=train data='{DATA_DIR}' model=yolov8m-cls.pt epochs=100 imgsz=224
model = YOLO('yolov8m-cls.pt')  # load a pretrained model (recommended for training)
# Train the model
results = model.train(data='{DATA_DIR}', epochs=100, imgsz=224)

#model = YOLO('path/to/best.pt')  # load a custom model

# Validate the model
#metrics = model.val()  # no arguments needed, dataset and settings remembered
#metrics.top1   # top1 accuracy
#metrics.top5   # top5 accuracy

#Prediction
#model = YOLO('path/to/best.pt')  # load a custom model

# Predict with the model
#results = model('https://ultralytics.com/images/bus.jpg')  # predict on an image

# Load the validation results CSV file
validation_results = pd.read_csv(f'{HOME}/runs/classify/train/results.csv')

# Extract ground truth and predicted labels
true_labels = validation_results['gt'].tolist()
predicted_labels = validation_results['pred'].tolist()

# Calculate F1 score and precision
f1 = f1_score(true_labels, predicted_labels, average='weighted')
precision = precision_score(true_labels, predicted_labels, average='weighted')

# Print the calculated metrics
print(f'Weighted F1 Score: {f1:.4f}')
print(f'Weighted Precision: {precision:.4f}')
