"""
Plant Disease Detection - CNN Training Pipeline
Author: Kalyani Deshmane
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import json

# ── Config ─────────────────────────────────────────────────────────────────────
IMG_SIZE     = (224, 224)
BATCH_SIZE   = 32
EPOCHS       = 30
NUM_CLASSES  = 38          # PlantVillage default class count
DATA_DIR     = "data/"
MODEL_DIR    = "model/saved_model"
HISTORY_PATH = "model/training_history.json"

CLASS_NAMES_PATH = "model/class_names.json"

# ── Data Augmentation & Generators ─────────────────────────────────────────────
def build_generators(data_dir: str):
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        brightness_range=[0.8, 1.2],
        validation_split=0.2,
    )
    val_datagen = ImageDataGenerator(rescale=1.0 / 255, validation_split=0.2)

    train_gen = train_datagen.flow_from_directory(
        os.path.join(data_dir, "train"),
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        subset="training",
        seed=42,
    )
    val_gen = val_datagen.flow_from_directory(
        os.path.join(data_dir, "train"),
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        subset="validation",
        seed=42,
    )
    return train_gen, val_gen

# ── Model Architecture ──────────────────────────────────────────────────────────
def build_model(num_classes: int) -> tf.keras.Model:
    base_model = tf.keras.applications.EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=(*IMG_SIZE, 3),
    )
    base_model.trainable = False   # freeze initially

    inputs = tf.keras.Input(shape=(*IMG_SIZE, 3))
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model

# ── Training ────────────────────────────────────────────────────────────────────
def train(model, train_gen, val_gen):
    cb_list = [
        callbacks.EarlyStopping(patience=5, restore_best_weights=True, verbose=1),
        callbacks.ReduceLROnPlateau(factor=0.5, patience=3, verbose=1),
        callbacks.ModelCheckpoint(
            filepath=os.path.join(MODEL_DIR, "best_model.keras"),
            save_best_only=True,
            monitor="val_accuracy",
            verbose=1,
        ),
    ]

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
        callbacks=cb_list,
    )

    # Fine-tune: unfreeze top layers
    model.layers[1].trainable = True
    for layer in model.layers[1].layers[:-20]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    history_ft = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=10,
        callbacks=cb_list,
    )
    return history, history_ft

# ── Plot & Save ─────────────────────────────────────────────────────────────────
def plot_history(history):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(history.history["accuracy"],     label="Train Acc")
    axes[0].plot(history.history["val_accuracy"], label="Val Acc")
    axes[0].set_title("Accuracy")
    axes[0].legend()

    axes[1].plot(history.history["loss"],     label="Train Loss")
    axes[1].plot(history.history["val_loss"], label="Val Loss")
    axes[1].set_title("Loss")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig("model/training_curves.png", dpi=150)
    plt.show()
    print("Training curves saved to model/training_curves.png")

# ── Main ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    os.makedirs(MODEL_DIR, exist_ok=True)

    train_gen, val_gen = build_generators(DATA_DIR)

    # Save class names
    with open(CLASS_NAMES_PATH, "w") as f:
        json.dump(list(train_gen.class_indices.keys()), f, indent=2)

    num_classes = len(train_gen.class_indices)
    print(f"Classes detected: {num_classes}")

    model = build_model(num_classes)
    model.summary()

    history, _ = train(model, train_gen, val_gen)

    # Save final model
    model.save(os.path.join(MODEL_DIR, "plant_disease_model.keras"))
    print("Model saved!")

    plot_history(history)

    # Save history
    with open(HISTORY_PATH, "w") as f:
        json.dump({k: [float(v) for v in vals]
                   for k, vals in history.history.items()}, f, indent=2)
