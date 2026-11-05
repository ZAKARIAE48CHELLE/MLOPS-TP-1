import os
import yaml
import json
import joblib
import mlflow
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score
)

# Suppress MLflow banner hints
os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"

def train():
    # 1. Charger les paramètres depuis params.yaml
    with open("params.yaml", "r") as file:
        params = yaml.safe_load(file)["train"]

    # 2. Charger les données préparées
    df = pd.read_csv("data/raw/iris.csv")
    X = df.drop("target", axis=1)
    y = df["target"]

    # 3. Séparation train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=params["test_size"],
        random_state=params["random_state"],
        stratify=y
    )

    # 4. Configuration de MLflow
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("TP_Iris_Classification")

    model_type = params.get("model_type", "random_forest")
    if model_type == "knn":
        n_neighbors = params.get("n_neighbors", 5)
        run_name = f"KNN_k_{n_neighbors}"
        model = KNeighborsClassifier(n_neighbors=n_neighbors)
    else:
        n_estimators = params.get("n_estimators", 100)
        max_depth = params.get("max_depth", 5)
        run_name = f"RandomForest_n_{n_estimators}_depth_{max_depth}"
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=params["random_state"]
        )

    with mlflow.start_run(run_name=run_name):
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        metrics = {
            "accuracy": float(accuracy_score(y_test, predictions)),
            "precision": float(precision_score(y_test, predictions, average="weighted")),
            "recall": float(recall_score(y_test, predictions, average="weighted")),
            "f1_score": float(f1_score(y_test, predictions, average="weighted"))
        }

        # Enregistrement des paramètres et métriques dans MLflow
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)

        # Enregistrement de l'artefact modèle
        os.makedirs("models", exist_ok=True)
        joblib.dump(model, "models/model.pkl")
        mlflow.log_artifact("models/model.pkl", artifact_path="model")

        # Enregistrement des métriques pour DVC
        os.makedirs("reports", exist_ok=True)
        with open("reports/metrics.json", "w") as f:
            json.dump(metrics, f, indent=4)

        print(f"Modèle ({model_type}) entraîné et enregistré sous run '{run_name}'.")
        print("Métriques :", metrics)

if __name__ == "__main__":
    train()
