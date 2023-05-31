import mlflow

mlflow.log_metric("accuracy", 0.9)

mlflow.log_param("learning_rate", 0.001)

import mlflow

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.log_artifact("data/a.txt", "b/c")


import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor

db = load_diabetes()
X_train, X_test, y_train, y_test = train_test_split(db.data, db.target)

rf = RandomForestRegressor(n_estimators=100, max_depth=6, max_features=3)
rf.fit(X_train, y_train)
mlflow.sklearn.log_model(rf, "myModel")
