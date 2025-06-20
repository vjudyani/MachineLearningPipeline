import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle
import yaml
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from mlflow.models import infer_signature
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from urllib.parse import urlparse
import mlflow

# Set DagsHub MLflow tracking environment variables
os.environ['MLFLOW_TRACKING_URI'] = "https://dagshub.com/vedika.judyani/MachineLearningPipeline.mlflow"
os.environ['MLFLOW_TRACKING_USERNAME'] = "vedika.judyani"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "89f80120a4a66ab848285963495af57f5b3fbb49"

def hyperparameter_tuning(X_train, y_train, param_grid):
    rf = RandomForestClassifier()
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train)
    return grid_search

# Load parameters from params.yaml
params = yaml.safe_load(open("params.yaml"))["train"]

def train(data_path, model_path, random_state, n_estimators, max_depth):
    data = pd.read_csv(data_path)
    X = data.drop(columns=["Outcome"])
    y = data["Outcome"]

    # Set tracking URI explicitly (redundant but explicit)
    mlflow.set_tracking_uri(os.environ['MLFLOW_TRACKING_URI'])

    with mlflow.start_run():
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=random_state)

        # Infer signature for model input/output (optional but recommended)
        signature = infer_signature(X_train, y_train)

        # Hyperparameter grid
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [5, 10, None],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2]
        }

        # Hyperparameter tuning
        grid_search = hyperparameter_tuning(X_train, y_train, param_grid)

        best_model = grid_search.best_estimator_

        # Predict and evaluate
        y_pred = best_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {accuracy}")

        # Log metrics and params
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_param("best_n_estimators", grid_search.best_params_['n_estimators'])
        mlflow.log_param("best_max_depth", grid_search.best_params_['max_depth'])
        mlflow.log_param("best_min_samples_split", grid_search.best_params_['min_samples_split'])
        mlflow.log_param("best_min_samples_leaf", grid_search.best_params_['min_samples_leaf'])

        # Log confusion matrix and classification report as text artifacts
        cm = confusion_matrix(y_test, y_pred)
        cr = classification_report(y_test, y_pred)
        mlflow.log_text(str(cm), "confusion_matrix.txt")
        mlflow.log_text(cr, "classification_report.txt")

        # Save model locally
        local_model_path = "model.pkl"
        with open(local_model_path, "wb") as f:
            pickle.dump(best_model, f)

        # Log the pickle file as an artifact
        mlflow.log_artifact(local_model_path)

        print("Model logged successfully!")

        # Save locally with pickle (optional)
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        with open(model_path, 'wb') as f:
            pickle.dump(best_model, f)

        # Log the saved model pickle file as artifact
        mlflow.log_artifact(model_path)

if __name__ == "__main__":
    train(params['data'], params['model'], params['random_state'], params['n_estimators'], params['max_depth'])
