import os
import glob
import random
import shutil
from pathlib import Path
from uuid import uuid4
from datetime import datetime

import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from image_processor import ImageProcessor

BASE_DIR = Path(__file__).resolve().parent


class ModelCreator:

    def __init__(self):
        self.train_path = os.path.join(BASE_DIR, 'data', 'train')
        self.test_path = os.path.join(BASE_DIR, 'data', 'test')

    def _generate_data(self, raw_path):
        """
        Crops raw catcha images into characters, splits data into train-test sets
        and saves to disk.
        """
        # keep train and test directories with their contents if exist
        if os.path.exists(self.train_path) and os.path.exists(self.test_path):
            for img_path in glob.iglob(
                    os.path.join(self.train_path, '**', '*.png'), recursive=True):
                self.image_size = cv2.imread(img_path).shape[:2]
                self.input_shape = self.image_size + (1,)
                # break at first iteration
                break
            return

        # create image processor object instance to process raw images
        img_processor = ImageProcessor()

        # create train and test directories
        os.makedirs(self.train_path)
        os.makedirs(self.test_path)
        # create temporary chars directory
        chars_path = os.path.join(BASE_DIR, 'data', 'temp_chars')
        if os.path.exists(chars_path):
            shutil.rmtree(chars_path)
        os.makedirs(chars_path)

        for filepath in glob.iglob(
                os.path.join(raw_path, '**', '*.png'), recursive=True):
            # grab the base filename as the text
            captcha_text = os.path.splitext(os.path.basename(filepath))[0]
            # crop captcha image to characters
            characters = img_processor.process(filepath)
            if not characters:
                continue
            for i, char in enumerate(captcha_text):
                # Create directory if not exist.
                char_dir = os.path.join(chars_path, char)
                os.makedirs(char_dir, exist_ok=True)
                # Set file name as a random string.
                charpath = os.path.join(char_dir, f'{uuid4().hex}.png')
                # save cropped char into the directory
                cv2.imwrite(charpath, characters[i])

        # get image shape
        self.image_size = cv2.imread(charpath).shape[:2]
        self.input_shape = self.image_size + (1,)

        # split data into train and test sets
        for _, subdirs, files in os.walk(chars_path):
            for subdir in subdirs:
                src_path = os.path.join(chars_path, subdir)
                src_files = os.listdir(src_path)
                if src_files:
                    # create sub-directories in train and test directories
                    train_sub_path = os.path.join(self.train_path, subdir)
                    os.makedirs(train_sub_path, exist_ok=True)
                    test_sub_path = os.path.join(self.test_path, subdir)
                    os.makedirs(test_sub_path, exist_ok=True)
                    # copy each file to sub-directory in train directory
                    for file_name in src_files:
                        file_path = os.path.join(src_path, file_name)
                        if os.path.isfile(file_path):
                            shutil.copy(file_path, train_sub_path)
                    train_files = os.listdir(train_sub_path)
                    # randomly select 20% of files from train and copy to test
                    test_files = random.sample(
                        train_files, round(len(train_files)*0.2))
                    for file_name in test_files:
                        file_path = os.path.join(train_sub_path, file_name)
                        if os.path.isfile(file_path):
                            shutil.copy(file_path, test_sub_path)
                            # remove file from trian directory
                            os.remove(file_path)
            # break at first iteration
            break
        # remove temporary directory
        shutil.rmtree(chars_path)

    def _define_model(self):
        self.model = keras.Sequential(
            [
                keras.Input(shape=self.input_shape),
                layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
                layers.MaxPooling2D(pool_size=(2, 2)),
                layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
                layers.MaxPooling2D(pool_size=(2, 2)),
                layers.Flatten(),
                layers.Dropout(0.5),
                layers.Dense(self.nr_classes, activation="softmax"),
            ]
        )
        self.model.compile(loss="categorical_crossentropy",
                           optimizer="adam", metrics=["accuracy"])

    def _create_data_generators(self, batch_size):
        """
        Creates train, validation and test data generators.
        """
        train_datagen = ImageDataGenerator(
            rescale=1./255, validation_split=0.2)
        self.train_generator = train_datagen.flow_from_directory(
            self.train_path,
            target_size=self.image_size,
            batch_size=batch_size,
            class_mode='categorical',
            color_mode='grayscale',
            seed=1337,
            subset='training')
        self.validation_generator = train_datagen.flow_from_directory(
            self.train_path,
            target_size=self.image_size,
            batch_size=batch_size,
            class_mode='categorical',
            color_mode='grayscale',
            seed=1337,
            subset='validation')

        test_datagen = ImageDataGenerator(rescale=1./255)
        self.test_generator = test_datagen.flow_from_directory(
            self.test_path,
            target_size=self.image_size,
            batch_size=batch_size,
            class_mode='categorical',
            color_mode='grayscale',
            shuffle=False,
            seed=1337)

        self.nr_classes = len((self.test_generator.class_indices))

    def fit_model(self, batch_size=32, epochs=10,
                  raw_path=os.path.join(BASE_DIR, 'data', 'raw')):
        self._generate_data(raw_path)
        self._create_data_generators(batch_size)
        self._define_model()
        self.model.fit(
            self.train_generator,
            steps_per_epoch=self.train_generator.samples//batch_size,
            epochs=epochs,
            validation_data=self.validation_generator,
            validation_steps=self.validation_generator.samples//batch_size)

    def evaluate_model(self):
        self.model.evaluate(self.test_generator)
        predictions = self.model.predict(self.test_generator)
        y_pred = np.argmax(np.rint(predictions), axis=1)
        y_true = self.test_generator.classes
        labels = (self.test_generator.class_indices)
        print(labels)
        print(tf.math.confusion_matrix(labels=y_true, predictions=y_pred))

    def save_model(self):
        ts = str(datetime.utcnow()).split(' ')[0].replace('-', '_')
        model_path = os.path.join(BASE_DIR, f'model_{ts}.h5')
        self.model.save(model_path)


if __name__ == '__main__':
    model = ModelCreator()
    model.fit_model()
    model.evaluate_model()
    model.save_model()
