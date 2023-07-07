"""
#!/usr/bin/env sh
export MLFLOW_TRACKING_URI=http://localhost:5000

mlflow models serve --model-uri runs:/88245c47e2c14d0ea0446b22f146b1d8/model --env-manager virtualenv --host 127.0.0.1:5001

mlflow models serve -m ./mlartifacts/0/88245c47e2c14d0ea0446b22f146b1d8/artifacts/model --port 5002 --no-conda

curl http://127.0.0.1:5002/invocations -H 'Content-Type: application/json' -d '{
  "dataframe_split": {
      "data": [[0.04170844, 0.05068012, 0.01211685, -0.00259226, 0.04560437, -0.0010777]]
  }
}'

"""
# import requests
#
# # a = requests.get("http://127.0.0.1:5001/ping")
# # print(a)
#
# # a = requests.get("http://127.0.0.1:5001/health")
# # print(a)
#
# # # mlflow version
# # a = requests.get("http://127.0.0.1:5001/version")
# # print(a.text)
#

import pandas as pd
from sklearn.datasets import load_diabetes
db = load_diabetes()
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(db.data, db.target)
a = pd.DataFrame(X_test)
print(list(a.columns))
print(type(a.to_dict(orient='split')))

import requests

headers = {
    'Content-Type': 'application/json',
}

json_data = {
    "dataframe_split": a.to_dict(orient='split'),
}

response = requests.post('http://127.0.0.1:5002/invocations', headers=headers, json=json_data)
print(response.text)
