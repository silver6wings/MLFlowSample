import mlflow

mlflow.set_tracking_uri("http://127.0.0.1:5000")

experiments = mlflow.search_experiments(
    filter_string='tags.version="v1"',
)

for exp in experiments:
    if exp.name == "test_02":
     with mlflow.start_run(experiment_id=exp.experiment_id, run_name="abc"):
        print(exp.tags)