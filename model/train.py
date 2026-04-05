import os
import yaml
import json
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

from architecture import build_model


def load_config():
    with open("configs/config.yaml", "r") as f:
        return yaml.safe_load(f)


def create_data_generators(config):
    img_size = config["training"]["image_size"]
    batch_size = config["training"]["batch_size"]

    aug = config["augmentation"]

    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=aug["rotation_range"],
        width_shift_range=aug["width_shift_range"],
        height_shift_range=aug["height_shift_range"],
        shear_range=aug["shear_range"],
        zoom_range=aug["zoom_range"],
        horizontal_flip=aug["horizontal_flip"]
    )

    val_datagen = ImageDataGenerator(rescale=1.0 / 255)

    train_gen = train_datagen.flow_from_directory(
        config["paths"]["train_dir"],
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode="binary"
    )

    val_gen = val_datagen.flow_from_directory(
        config["paths"]["val_dir"],
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode="binary"
    )

    return train_gen, val_gen


def save_class_indices(generator, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(generator.class_indices, f)


def main():
    config = load_config()

    print("[INFO] Loading data...")
    train_gen, val_gen = create_data_generators(config)

    print("[INFO] Building model...")
    model = build_model(config)

    os.makedirs(os.path.dirname(config["paths"]["model_save_path"]), exist_ok=True)

    checkpoint = ModelCheckpoint(
        config["paths"]["model_save_path"],
        monitor="val_loss",
        save_best_only=True,
        verbose=1
    )

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True
    )

    print("[INFO] Starting training...")
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=config["training"]["epochs"],
        callbacks=[checkpoint, early_stop]
    )

    print("[INFO] Saving class indices...")
    save_class_indices(train_gen, config["paths"]["class_indices_path"])

    print("[INFO] Training complete.")


if __name__ == "__main__":
    main()