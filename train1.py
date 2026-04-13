import mlflow
import mlflow.sklearn

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

# Load dataset
data = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.2, random_state=42
)

# Models dictionary
models = {
    "LogisticRegression": LogisticRegression(max_iter=200),
    "RandomForest": RandomForestClassifier(n_estimators=100),
    "DecisionTree": DecisionTreeClassifier(max_depth=5)
}

mlflow.set_experiment("Multi_Model_Experiment")

# Simulate 2 epochs
for epoch in range(1, 3):   # 2 epochs
    for model_name, model in models.items():

        with mlflow.start_run(run_name=f"{model_name}_epoch_{epoch}"):

            # Train model
            model.fit(X_train, y_train)

            preds = model.predict(X_test)

            # Metrics
            accuracy = accuracy_score(y_test, preds)
            precision = precision_score(y_test, preds, average="macro")
            recall = recall_score(y_test, preds, average="macro")

            # Log parameters
            mlflow.log_param("model", model_name)
            mlflow.log_param("epoch", epoch)

            # Log metrics
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)

            # Log model
            mlflow.sklearn.log_model(model, "model")

            print(f"{model_name} | Epoch {epoch} | Acc: {accuracy}")
            
            
            