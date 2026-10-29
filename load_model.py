import os
import sys

# Suppress MLflow banner hints
os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"

import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score

# Identifiant réel d'un run issu de l'expérience TP_MLflow_Iris (Run RF_100)
DEFAULT_RUN_ID = "512d868c9ebb4b2490cd3c189a902d9d"

# Possibilité de spécifier un autre RUN_ID en ligne de commande : python load_model.py <RUN_ID>
RUN_ID = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_RUN_ID

model_uri = f"runs:/{RUN_ID}/iris_model"

print(f"Chargement du modle depuis : {model_uri}")
try:
    model = mlflow.sklearn.load_model(model_uri)

    # Rechargement et valuation sur l'ensemble Iris complet
    X, y = load_iris(return_X_y=True)
    predictions = model.predict(X)
    acc = accuracy_score(y, predictions)
    print(f"Modle recharg avec succs (Run ID : {RUN_ID})")
    print(f"Accuracy sur Iris complet : {acc:.4f}")
except Exception as e:
    print(f"Erreur lors du rechargement du modle : {e}")
