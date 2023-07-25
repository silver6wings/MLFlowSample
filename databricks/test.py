"""
https://docs.databricks.com/archive/dev-tools/cli/index.html#set-up-authentication
https://mlflow.org/docs/latest/quickstart.html#quickstart-tracking-server
"""

import requests
import mlflow

# response = requests.get(
#     "https://dss-svc-eng-prod-us-east-1.cloud.databricks.com/version"
# )
# assert response.text == mlflow.__version__  # Checking for a strict version match


def test_databricks():
    from sklearn.model_selection import train_test_split
    from sklearn.datasets import load_diabetes
    from sklearn.ensemble import RandomForestRegressor

    mlflow.set_tracking_uri("databricks")
    mlflow.set_experiment("/Users/junchao.yu@disneystreaming.com/my-experiment")

    mlflow.autolog()
    with mlflow.start_run(run_name="abcdefg") as run:
        print(run.info)

        db = load_diabetes()

        X_train, X_test, y_train, y_test = train_test_split(db.data, db.target)

        # Create and trsain models.
        rf = RandomForestRegressor(n_estimators=100, max_depth=6, max_features=3)
        rf.fit(X_train, y_train)

        run = mlflow.active_run()
        print(run.info)

        # # Use the model to make predictions on the test dataset.
        predictions = rf.predict(X_test)
        print(predictions)


if __name__ == '__main__':
    test_databricks()
