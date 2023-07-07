import mlflow

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.autolog()

db = load_diabetes()

X_train, X_test, y_train, y_test = train_test_split(db.data, db.target)

# rf = mlflow.sklearn.load_model("mlflow-artifacts:/0/d0b27e09077744cd8ad21cec5e4f5aab/artifacts/myModel")

model_name = "test_r_m"
model_version = 2
rf = mlflow.sklearn.load_model(model_uri=f"models:/{model_name}/{model_version}")

print(X_test)

predictions = rf.predict(X_test)

print(predictions)
