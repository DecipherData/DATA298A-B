from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Input
from tensorflow.keras.applications import InceptionResNetV2
from tensorflow.keras.utils import plot_model

# Load the pre-trained ResNet50 model
base_model = InceptionResNetV2(weights='imagenet', include_top=False)

# Freeze all layers in the base model except for the last one
for layer in base_model.layers[:-1]:
    layer.trainable = False

# Create new model on top
inputs = Input(shape=(299, 299, 3))
x = base_model(inputs, training=False)
x = GlobalAveragePooling2D()(x)
x = Dropout(0.2)(x)  # 20% dropout rate
x = Dense(32, activation='relu')(x)  # Dense layer with 32 units and ReLU activation
outputs = Dense(11, activation='softmax')(x)  # Output layer for 11 classes with SoftMax activation

# Combine the base model and the added layers into a new model
model = Model(inputs, outputs)

# Visualize the model architecture
plot_model(model, to_file=r'C:\vIDUSHI\SJSU\DATA 298A - PlanB Project\modelArchitecturePlot\model_architecture.png', show_shapes=True)
