import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import scipy
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_dir=r"C:\Users\mural\PycharmProjects\ObjectDetection\Images\train"
valid_dir=r"C:\Users\mural\PycharmProjects\ObjectDetection\Images\validation"

train_datagen=ImageDataGenerator(rescale=1/255.,
                                             rotation_range=20,
                                             shear_range=0.2,
                                             zoom_range=0.2,
                                             width_shift_range=0.2,
                                             height_shift_range=0.2,
                                             horizontal_flip=True)
valid_datagen=ImageDataGenerator(
                rescale=1/255.
)

train_data=train_datagen.flow_from_directory(
                        train_dir,
                        batch_size=32,
                        target_size=(48,48),
                        color_mode='grayscale',
                        class_mode='categorical',
                        seed=42
                        )
valid_data=valid_datagen.flow_from_directory(
                        valid_dir,
                        batch_size=32,
                        target_size=(48,48),
                        color_mode='grayscale',
                        class_mode='categorical',
                        seed=42
                        )

base_model=tf.keras.models.Sequential([
  tf.keras.layers.Conv2D(128,kernel_size=3,activation="relu",input_shape=(48,48,1)),
  tf.keras.layers.Conv2D(128, 3, activation="relu"),
  tf.keras.layers.MaxPool2D(2),
  tf.keras.layers.Conv2D(64, 3, activation="relu"),
  tf.keras.layers.Conv2D(32, 3, activation="relu"),
  tf.keras.layers.MaxPool2D(2),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(7, activation="softmax")
])

base_model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

base_model.fit(train_data,
               steps_per_epoch=len(train_data),
               epochs=5,
               validation_data=valid_data,
               validation_steps=len(valid_data),
)
model_json =base_model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)

base_model.save_weights("model.h5")