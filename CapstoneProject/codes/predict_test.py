import requests

url = 'http://0.0.0.0:8080/predict'

filepath = './yoga-posture-cleaned/test/downdog/downdog101.jpg'

response = requests.post(url, json=filepath).json()
print(response)

