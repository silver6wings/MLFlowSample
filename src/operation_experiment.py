import mlflow

mlflow.set_tracking_uri("http://127.0.0.1:5000")

mlflow.create_experiment(
    name="test_02",
    artifact_location=None,
    tags={
        "version": "v1",
        "priority": "P1",
    },
)

experiments = mlflow.search_experiments(
    # filter_string='tags.version="v1"',
)

for exp in experiments:
    print(exp.experiment_id)
    print(exp.tags)
    # mlflow.delete_experiment(experiment_id=exp.experiment_id)
