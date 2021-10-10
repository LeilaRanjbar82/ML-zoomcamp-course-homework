import requests


url = 'http://0.0.0.0:8888/predict-q6'

customer_id = 'xyz-125'
customer = {
    "contract": "two_year",
    "tenure": 12,
    "monthlycharges": 10
}


response = requests.post(url, json=customer).json()
print(response)

if response['churn'] == True:
    print('sending promo email to %s' % customer_id)
else:
    print('not sending promo email to %s' % customer_id)