import pickle

from flask import Flask
from flask import request
from flask import jsonify

model_file = 'model1.bin'
dv_file = 'dv.bin'

with open(model_file, 'rb') as mf_in:
    model = pickle.load(mf_in)

with open(dv_file, 'rb') as dvf_in:
    dv = pickle.load(dvf_in)


app = Flask('churn')


@app.route('/predict-q4', methods=['POST'])
def predict():
    customer = request.get_json()

    X = dv.transform([customer])
    y_pred = model.predict_proba(X)[0, 1]
    churn = y_pred >= 0.5

    result = {
        'churn_probability': float(y_pred),
        'churn': bool(churn)
    }

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=8888)

