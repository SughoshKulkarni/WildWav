import requests

url = 'http://localhost:5000/predict_api'
r = requests.post(url,json={'sepal-length':5, 'sepal-width':4, 'petal-length':3, 'petal-width':0.2})

print(r.json())