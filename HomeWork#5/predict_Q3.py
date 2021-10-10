
import pickle

model_file = 'model1.bin'
dv_file = 'dv.bin'

with open(model_file, 'rb') as mf_in:
    model = pickle.load(mf_in)

with open(dv_file, 'rb') as dvf_in:
    dv = pickle.load(dvf_in)


customer = {
    "contract": "two_year",
    "tenure": 12,
    "monthlycharges": 19.7
}


X = dv.transform([customer])
y_pred = model.predict_proba(X)[0, 1]

print('input', customer)
print('churn probability', y_pred)
