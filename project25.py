import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Suppress oneDNN warnings

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt

# -------------------------------
# Paths
# -------------------------------
train_dir = 'data/'      # Make sure cats/ and dogs/ are inside this
model_dir = 'model/'     # Make sure this folder exists

# -------------------------------
# Image data generator
# -------------------------------
train_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(128, 128),
    batch_size=4,
    class_mode='binary'
)

# Print class info
print("Class indices:", train_generator.class_indices)
print("Number of images:", train_generator.samples)

# -------------------------------
# Load pretrained MobileNetV2
# -------------------------------
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(128,128,3))
base_model.trainable = False   # Freeze base layers

# Add custom layers on top
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
predictions = Dense(1, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# Compile model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# -------------------------------
# Train the model
# -------------------------------
history = model.fit(
    train_generator,
    epochs=5
)

# -------------------------------
# Save the trained model
# -------------------------------
os.makedirs(model_dir, exist_ok=True)  # Ensure model folder exists
model.save(os.path.join(model_dir, 'cats_dogs_model.h5'))
print("Model saved to", os.path.join(model_dir, 'cats_dogs_model.h5'))

# -------------------------------
# Plot training accuracy
# -------------------------------
plt.plot(history.history['accuracy'], label='train accuracy')
plt.legend()
plt.show()
