import pickle

import xgboost as xgb

from flask import Flask
from flask import request
from flask import jsonify



model_file = 'model.bin'

with open(model_file, 'rb') as f_in:
    dv, model = pickle.load(f_in)

app = Flask('response')


@app.route('/predict', methods=['POST'])
def predict():
    customer = request.get_json()

    X = dv.transform([customer])
    dtest = xgb.DMatrix(X, feature_names=dv.get_feature_names_out())
    y_pred = model.predict(dtest)
    satisfaction = y_pred > 0.3


    result = {
        'response_probability': float(y_pred),
        'response': bool(satisfaction)
    }

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host='localhost', port=8889)
