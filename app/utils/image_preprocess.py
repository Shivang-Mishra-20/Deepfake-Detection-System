from PIL import Image
import numpy as np


def preprocess_image(image, image_size):
    image = image.resize((image_size, image_size))
    image = np.array(image) / 255.0

    if image.shape[-1] == 4:
        image = image[:, :, :3]

    image = np.expand_dims(image, axis=0)
    return image