#!/usr/bin/env python
# coding: utf-8



import tflite_runtime.interpreter as tflite
import numpy as np
from io import BytesIO
from urllib import request
from PIL import Image



target_size = (150,150)
#url = 'https://upload.wikimedia.org/wikipedia/commons/9/9a/Pug_600.jpg'



interpreter = tflite.Interpreter(model_path='cats-dogs-v2.tflite')
interpreter.allocate_tensors()

input_index = interpreter.get_input_details()[0]['index']
output_index = interpreter.get_output_details()[0]['index']


def download_prepare_image(url, target_size):
    with request.urlopen(url) as resp:
        buffer = resp.read()
    stream = BytesIO(buffer)
    img = Image.open(stream)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize(target_size, Image.NEAREST)
    return img


def preprocess_image(url, target_size):
    img = download_prepare_image(url, target_size)
    x = np.array(img, dtype='float32')
    X = np.array([x])
    X = X / 255
    return X
    


def predict(url):
    X = preprocess_image(url, target_size)

    interpreter.set_tensor(input_index, X)
    interpreter.invoke()
    preds = interpreter.get_tensor(output_index)

    float_predictions = preds[0].tolist()

    return float_predictions


def lambda_handler(event, context):
    url = event['url']
    result = predict(url)
    return result



