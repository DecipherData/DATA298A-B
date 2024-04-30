import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import InceptionResNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras_tuner import RandomSearch
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

# For reproducibility, set the random seed
np.random.seed(42)
tf.random.set_seed(42)

# Define the path to the root folder of the dataset
base_dir = r'C:\Users\bhati\DATA298-FinalProject\Removed_Duplicates\train_val_test'
train_dir = os.path.join(base_dir, 'train')
val_dir = os.path.join(base_dir, 'val')
test_dir = os.path.join(base_dir, 'test')

# Image size and batch size
image_size = (299,299)
batch_size = 32

# Setup ImageDataGenerators
train_datagen = ImageDataGenerator(preprocessing_function=keras.applications.inception_resnet_v2.preprocess_input)
val_datagen = ImageDataGenerator(preprocessing_function=keras.applications.inception_resnet_v2.preprocess_input)
test_datagen = ImageDataGenerator(preprocessing_function=keras.applications.inception_resnet_v2.preprocess_input)

# Create data generators
train_generator = train_datagen.flow_from_directory(
    train_dir, target_size=image_size, batch_size=batch_size, class_mode='categorical')
val_generator = val_datagen.flow_from_directory(
    val_dir, target_size=image_size, batch_size=batch_size, class_mode='categorical')
test_generator = test_datagen.flow_from_directory(
    test_dir, target_size=image_size, batch_size=batch_size, class_mode='categorical', shuffle=False)

# Define a model-building function for Keras Tuner
def build_model(hp):
    base_model = InceptionResNetV2(weights='imagenet', include_top=False, input_shape=(*image_size, 3))
    base_model.trainable = False

    inputs = keras.Input(shape=(*image_size, 3))
    x = base_model(inputs, training=False)
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.2)(x)
    hp_units = hp.Int('units', min_value=32, max_value=512, step=32)
    x = Dense(hp_units, activation='relu')(x)
    outputs = Dense(11, activation='softmax')(x)  # Assuming 11 classes

    model = Model(inputs, outputs)
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),
                  loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Instantiate the tuner
tuner = RandomSearch(
    build_model,
    objective='val_accuracy',
    max_trials=10,
    executions_per_trial=1,
    directory='my_dir',
    project_name='hparam_tuning'
)

tuner.search_space_summary()

# Perform hyperparameter tuning
tuner.search(train_generator, epochs=10, validation_data=val_generator)

# Get the best hyperparameters
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

# Build the model with the best hyperparameters
model = tuner.hypermodel.build(best_hps)

# Callbacks
callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    ModelCheckpoint(filepath='best_inceptionresnetv2_model.h5', save_best_only=True, monitor='val_loss'),
    ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2)
]

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    validation_data=val_generator,
    validation_steps=val_generator.samples // batch_size,
    epochs=20,
    callbacks=callbacks
)

# Evaluate the model on the test data
test_loss, test_accuracy = model.evaluate(test_generator)
print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_accuracy}")

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

# Plot training & validation accuracy values
epochs_range = range(len(history.history['accuracy']))
plt.plot(epochs_range, history.history['accuracy'], label='Train')
plt.plot(epochs_range, history.history['val_accuracy'], label='Val')
plt.plot(epochs_range, [test_accuracy]*len(epochs_range), label='Test', linestyle='--')
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(epochs_range, history.history['loss'], label='Train')
plt.plot(epochs_range, history.history['val_loss'], label='Val')
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(loc='upper left')
plt.show()

