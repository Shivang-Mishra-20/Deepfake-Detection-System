import os
import yaml
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, roc_auc_score


def load_config():
    with open("configs/config.yaml", "r") as f:
        return yaml.safe_load(f)


def main():
    config = load_config()

    model_path = config["paths"]["model_save_path"]
    val_dir = config["paths"]["val_dir"]
    img_size = config["training"]["image_size"]
    batch_size = config["training"]["batch_size"]

    if not os.path.exists(model_path):
        raise FileNotFoundError("Model not found. Train first.")

    model = load_model(model_path)

    datagen = ImageDataGenerator(rescale=1.0 / 255)

    generator = datagen.flow_from_directory(
        val_dir,
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode="binary",
        shuffle=False
    )

    preds = model.predict(generator)
    y_pred = (preds > 0.5).astype(int).flatten()
    y_true = generator.classes

    print("\nClassification Report:\n")
    print(classification_report(y_true, y_pred))

    try:
        roc = roc_auc_score(y_true, preds)
        print(f"\nROC-AUC: {roc:.4f}")
    except:
        print("ROC-AUC could not be computed.")


if __name__ == "__main__":
    main()