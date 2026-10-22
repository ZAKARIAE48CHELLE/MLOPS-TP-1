import os

# Suppress MLflow banner hints
os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"

import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score

# Remplacez RUN_ID par l'identifiant d'un run affiché dans l'interface MLflow
RUN_ID = "REMPLACER_PAR_LE_RUN_ID"

model_uri = f"runs:/{RUN_ID}/iris_model"

if RUN_ID == "REMPLACER_PAR_LE_RUN_ID":
    print("Veuillez remplacer RUN_ID par un identifiant réel depuis l'interface MLflow (ex: dans l'expérience TP_MLflow_Iris).")
else:
    print(f"Chargement du modèle depuis : {model_uri}")
    model = mlflow.sklearn.load_model(model_uri)

    # Évaluation sur l'ensemble Iris
    X, y = load_iris(return_X_y=True)
    predictions = model.predict(X)
    print("Accuracy sur Iris complet :", accuracy_score(y, predictions))
