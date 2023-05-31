from typing import List
import mlflow


mlflow.set_tracking_uri("http://127.0.0.1:5000")


# def predict(model_input: List[str]) -> List[str]:
#     return [i.upper() for i in model_input]
#
#
# mlflow.pyfunc.save_model("./model", python_model=predict, input_example=["a"])

# model = mlflow.pyfunc.load_model("model")
# print(model.predict(["a", "b", "c"]))  # -> ["A", "B", "C"]

with mlflow.start_run(run_name='name'):
    mlflow.log_artifacts("./model", artifact_path="model")