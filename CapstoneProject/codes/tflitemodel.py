import numpy as np

import tensorflow as tf
from tensorflow import keras

model = keras.models.load_model('yoga_08_0.862.h5')

converter = tf.lite.TFLiteConverter.from_keras_model(model)

tflite_model = converter.convert()

with open('yoga_08_0.862.tflite', 'wb') as f_out:
    f_out.write(tflite_model)