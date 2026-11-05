import os
import sys
import joblib
import pandas as pd
from sklearn.metrics import accuracy_score

# Suppress MLflow banner hints
os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"

def load_and_evaluate(model_source="models/model.pkl"):
    """
    Recharge un modèle entraîné (soit depuis le fichier sérialisé models/model.pkl,
    soit depuis un RUN_ID MLflow) et effectue une inférence sur les données Iris.
    """
    data_path = "data/raw/iris.csv"
    if not os.path.exists(data_path):
        from src.prepare_data import prepare
        prepare()

    df = pd.read_csv(data_path)
    X = df.drop("target", axis=1)
    y = df["target"]

    print(f"--- Rechargement du modèle depuis : {model_source} ---")

    if os.path.exists(model_source):
        # 1. Chargement direct de l'artefact généré par DVC (joblib)
        model = joblib.load(model_source)
        print("Modèle chargé avec succès via joblib.")
    else:
        # 2. Chargement via MLflow Tracking
        import mlflow
        mlflow.set_tracking_uri("sqlite:///mlflow.db")
        run_id = model_source
        model_uri = f"runs:/{run_id}/model"
        print(f"Chargement via MLflow URI : {model_uri}")
        model = mlflow.sklearn.load_model(model_uri)

    predictions = model.predict(X)
    acc = accuracy_score(y, predictions)
    print(f"Inférence réussie ! Accuracy sur le dataset complet : {acc:.4f}")
    return acc

if __name__ == "__main__":
    source = sys.argv[1] if len(sys.argv) > 1 else "models/model.pkl"
    load_and_evaluate(source)
