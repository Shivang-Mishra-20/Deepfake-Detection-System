import os
import json
import tensorflow as tf

from app.utils.image_preprocess import preprocess_image


# Global variables (load once)
model = None
class_indices = None


def load_model_and_classes(config):
    global model, class_indices

    # ✅ Get paths from config (NOT hardcoded)
    model_path = config["paths"]["model_save_path"]
    class_indices_path = config["paths"]["class_indices_path"]

    print(f"[DEBUG] Model path: {model_path}")
    print(f"[DEBUG] Class indices path: {class_indices_path}")

    # ✅ Check existence
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")

    if not os.path.exists(class_indices_path):
        raise FileNotFoundError(f"Class indices not found at {class_indices_path}")

    # ✅ Load model
    print("[INFO] Loading model...")
    model = tf.keras.models.load_model(
        model_path,
        compile=False,
        safe_mode=False
    )

    # ✅ Load class indices
    print("[INFO] Loading class indices...")
    with open(class_indices_path, "r") as f:
        class_indices = json.load(f)

    # ✅ Reverse mapping → {0: "real", 1: "fake"}
    class_indices = {v: k for k, v in class_indices.items()}

    print("[INFO] Model and classes loaded successfully.")


def predict(image, config):
    global model, class_indices

    if model is None or class_indices is None:
        raise RuntimeError("Model not loaded. Call load_model_and_classes first.")

    # ✅ Preprocess image
    processed = preprocess_image(
        image,
        config["training"]["image_size"]
    )

    preds = model.predict(processed)[0][0]

    threshold = config["inference"]["confidence_threshold"]

    if preds >= threshold:
        label = class_indices[1]
        confidence = float(preds)
    else:
        label = class_indices[0]
        confidence = float(1 - preds)

    return {
        "prediction": label,
        "confidence": round(confidence, 4)
    }