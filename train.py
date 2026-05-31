import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense,
    Dropout,
    BatchNormalization
)
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import image_dataset_from_directory

# =====================================================
# CONFIGURATION
# =====================================================

DATASET_PATH = "PetImages"

IMG_HEIGHT = 128
IMG_WIDTH = 128

BATCH_SIZE = 32
EPOCHS = 10

MODEL_DIR = "model"
MODEL_PATH = os.path.join(
    MODEL_DIR,
    "cat_dog_cnn.keras"
)

# =====================================================
# CREATE MODEL DIRECTORY
# =====================================================

os.makedirs(MODEL_DIR, exist_ok=True)

# =====================================================
# LOAD DATASET
# =====================================================

print("Loading Dataset...")

train_ds = image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.2,
    subset="training",
    seed=42,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE
)

val_ds = image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.2,
    subset="validation",
    seed=42,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE
)

class_names = train_ds.class_names

print("\nClasses Found:")
print(class_names)

# =====================================================
# PERFORMANCE OPTIMIZATION
# =====================================================

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().shuffle(1000).prefetch(
    buffer_size=AUTOTUNE
)

val_ds = val_ds.cache().prefetch(
    buffer_size=AUTOTUNE
)

# =====================================================
# CNN MODEL
# =====================================================

print("\nBuilding CNN Model...")

model = Sequential([

    tf.keras.layers.Rescaling(
        1./255,
        input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)
    ),

    Conv2D(
        32,
        (3,3),
        activation="relu"
    ),
    BatchNormalization(),
    MaxPooling2D(),

    Conv2D(
        64,
        (3,3),
        activation="relu"
    ),
    BatchNormalization(),
    MaxPooling2D(),

    Conv2D(
        128,
        (3,3),
        activation="relu"
    ),
    BatchNormalization(),
    MaxPooling2D(),

    Flatten(),

    Dense(
        256,
        activation="relu"
    ),

    Dropout(0.4),

    Dense(
        1,
        activation="sigmoid"
    )
])

# =====================================================
# COMPILE
# =====================================================

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# =====================================================
# CALLBACKS
# =====================================================

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True
)

# =====================================================
# TRAIN
# =====================================================

print("\nTraining Started...")

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=[early_stop]
)

# =====================================================
# SAVE MODEL
# =====================================================

model.save(MODEL_PATH)

print("\nModel Saved Successfully")
print(MODEL_PATH)

# =====================================================
# EVALUATION
# =====================================================

loss, accuracy = model.evaluate(
    val_ds,
    verbose=0
)

print(f"\nValidation Accuracy: {accuracy:.4f}")
print(f"Validation Loss: {loss:.4f}")

# =====================================================
# ACCURACY GRAPH
# =====================================================

plt.figure(figsize=(8,5))

plt.plot(
    history.history["accuracy"],
    label="Training Accuracy"
)

plt.plot(
    history.history["val_accuracy"],
    label="Validation Accuracy"
)

plt.title("CNN Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.savefig(
    "accuracy_plot.png",
    bbox_inches="tight"
)

plt.close()

# =====================================================
# LOSS GRAPH
# =====================================================

plt.figure(figsize=(8,5))

plt.plot(
    history.history["loss"],
    label="Training Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)

plt.title("CNN Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.savefig(
    "loss_plot.png",
    bbox_inches="tight"
)

plt.close()

print("\nPlots Saved:")
print("accuracy_plot.png")
print("loss_plot.png")

print("\nTraining Completed Successfully")