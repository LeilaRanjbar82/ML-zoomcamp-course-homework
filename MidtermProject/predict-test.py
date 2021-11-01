#!/usr/bin/env python
# coding: utf-8

import requests

url = 'http://localhost:8889/predict'

customer = {"gender": "male",
            "age": 21,
            "driving_license": "yes",
            "region_code": 30,
            "previously_insured": "yes",
            "vehicle_age": "less_than_1_year",
            "vehicle_damage": "yes",
            "annual_premium": 21425,
            "policy_sales_channel": 26,
            "vintage": 252
            }

response = requests.post(url, json=customer).json()
print(response)

if response['response']:
    print('Customer is interested in Vehicle Insurance provided by the company')
else:
    print('Customer is NOT interested in Vehicle Insurance provided by the company')
