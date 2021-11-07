import os
from pathlib import Path

import numpy as np
from tensorflow import keras

from image_processor import ImageProcessor


def decode_captcha(filepath, model_path):
    """
    Processes captcha image, crops into characters and returns decoded captcha.
    """
    try:
        model = keras.models.load_model(model_path)

        img_processor = ImageProcessor()
        chars = img_processor.process(filepath)

        if not chars:
            return False

        labels = ['2', '3', '4', '5', '6', '7', '8', 'a', 'b', 'c', 'd', 'e', 'f',
                  'g', 'h', 'k', 'm', 'n', 'p', 'r', 'w', 'x', 'y']

        captcha = ''
        for char in chars:
            char = keras.preprocessing.image.img_to_array(char)
            char = np.expand_dims(char, axis=0)
            char = labels[np.argmax(model.predict(char), axis=-1)[0]]
            if char:
                captcha += char
            else:
                return False

        return captcha
    except:
        return False


if __name__ == '__main__':
    base_dir = Path(__file__).resolve().parent
    filepath = os.path.join(base_dir, 'img', '2a3cd.png')
    model_path = os.path.join(base_dir, 'model_2021_11_07.h5')
    captcha = decode_captcha(filepath, model_path)
    print('Decoded Captcha:', captcha)
