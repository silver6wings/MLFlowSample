export MLFLOW_TRACKING_URI=databricks
export MLFLOW_TRACKING_INSECURE_TLS=true
export DATABRICKS_HOST="https://dss-svc-eng-prod-us-east-1.cloud.databricks.com/"
export DATABRICKS_TOKEN="<databricks-token>"


mlflow experiments create -n /Users/junchao.yu@disneystreaming.com/my-experiment
