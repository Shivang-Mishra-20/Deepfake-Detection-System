import pytest
from PIL import Image
import yaml

from app.services.inference import predict, load_model_and_classes


def load_config():
    with open("configs/config.yaml", "r") as f:
        return yaml.safe_load(f)


def test_prediction():
    config = load_config()
    load_model_and_classes(config)

    image = Image.new("RGB", (224, 224), color="white")

    result = predict(image, config)

    assert "prediction" in result
    assert "confidence" in result