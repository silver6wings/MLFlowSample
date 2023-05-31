from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC



# If you are running Databricks Runtime for Machine Learning, `mlflow` is already installed and you can skip the following line.
import mlflow

# Load the iris dataset from scikit-learn
iris = load_iris()
X = iris.data
y = iris.target

def objective(C):
    # Create a support vector classifier model
    clf = SVC(C=C)

    # Use the cross-validation accuracy to compare the models' performance
    accuracy = cross_val_score(clf, X, y).mean()

    # Hyperopt tries to minimize the objective function. A higher accuracy value means a better model, so you must return the negative accuracy.
    return {'loss': -accuracy, 'status': STATUS_OK}

