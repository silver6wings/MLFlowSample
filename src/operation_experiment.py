import mlflow

# mlflow.create_experiment(
#     name="test_01",
#     artifact_location=None,
#     tags={
#         "version": "v1",
#         "priority": "P1",
#     },
# )

experiments = mlflow.search_experiments(
    filter_string='tags.version="v1"',
)

for exp in experiments:
    print(exp.tags)
