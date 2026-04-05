import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0


def build_model(config):
    """
    Builds EfficientNet-based model with controlled fine-tuning
    """

    image_size = config["training"]["image_size"]
    weights = config["model"]["weights"]
    trainable_layers = config["model"]["trainable_layers"]

    input_shape = (image_size, image_size, 3)

    # Load base model
    try:
        base_model = EfficientNetB0(
            weights=weights,
            include_top=False,
            input_shape=input_shape
        )
        print("[INFO] Loaded pretrained weights.")
    except Exception as e:
        print(f"[WARNING] Failed to load pretrained weights: {e}")
        base_model = EfficientNetB0(
            weights=None,
            include_top=False,
            input_shape=input_shape
        )

    # Freeze most layers
    for layer in base_model.layers[:-trainable_layers]:
        layer.trainable = False

    # Unfreeze top layers for fine-tuning
    for layer in base_model.layers[-trainable_layers:]:
        layer.trainable = True

    # Custom classification head
    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    output = layers.Dense(1, activation='sigmoid')(x)

    model = models.Model(inputs=base_model.input, outputs=output)

    # Compile
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=config["training"]["learning_rate"]
        ),
        loss='binary_crossentropy',
        metrics=[
            'accuracy',
            tf.keras.metrics.Precision(),
            tf.keras.metrics.Recall()
        ]
    )

    return model