import requests

url = 'http://localhost:8080/2015-03-31/functions/function/invocations'

data = {'url':'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSIPLzGVtV2MDVS8PSUsr7WpM-L91TEVxjp1Q&usqp=CAU'}


result = requests.post(url, json=data).json()
print(result)
