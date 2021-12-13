
import tensorflow as tf
import numpy as np

from tensorflow import keras
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.applications.xception import preprocess_input

from flask import Flask
from flask import request
from flask import jsonify



model_file = 'yoga_08_0.862.h5'
model = keras.models.load_model(model_file)

classes = [
    'chair', 
    'cobra', 
    'downdog', 
    'goddess', 
    'tree', 
    'warrior']

app = Flask('response')

@app.route('/predict', methods=['POST'])

def predict():
    filepath = request.get_json()
    img = load_img(filepath, target_size=(150, 150))
    x = np.array(img)
    X = np.array([x])
    X = preprocess_input(X)
    pred = model.predict(X).tolist()
    result = dict(zip(classes, pred[0]))
    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=8080)