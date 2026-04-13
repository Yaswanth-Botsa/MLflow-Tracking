import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load dataset
data = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.2, random_state=42
)

# Set experiment
mlflow.set_experiment("Iris_Experiment")

# Start MLflow run
with mlflow.start_run():

    n_estimators = 50

    model = RandomForestClassifier(n_estimators=n_estimators)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    # Log parameters and metrics
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_metric("accuracy", accuracy)

    # Log model
    mlflow.sklearn.log_model(model, "model")

    print("Accuracy:", accuracy)
    
    
    
    
    
    //pip install mlflow scikit-learn pandas//