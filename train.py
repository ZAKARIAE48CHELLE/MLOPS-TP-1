import os
import time
import json
import yaml
import joblib
import matplotlib.pyplot as plt
import mlflow
import mlflow.data
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

# Suppress MLflow banner hints
os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"


def get_dvc_hash(dvc_file_path="data/raw/iris.csv.dvc"):
    """Récupère le hash MD5 généré par DVC pour la version des données."""
    if os.path.exists(dvc_file_path):
        try:
            with open(dvc_file_path, "r") as f:
                dvc_meta = yaml.safe_load(f)
                return dvc_meta.get("outs", [{}])[0].get("md5")
        except Exception:
            pass
    return None


def train(params_path="params.yaml"):
    """
    Entraîne le modèle en intégrant la triade MLOps (Git + DVC + MLflow) :
    - Hyperparamètres depuis params.yaml (DVC params)
    - Données suivies par DVC (data/raw/iris.csv)
    - Suivi complet avec MLflow (sqlite:///mlflow.db) : params, métriques, artefacts
    - Enregistrement du modèle (models/model.pkl) et des métriques (reports/metrics.json)
    """
    # 1. Charger les paramètres depuis params.yaml
    if os.path.exists(params_path):
        with open(params_path, "r") as file:
            params = yaml.safe_load(file).get("train", {})
    else:
        params = {
            "model_type": "random_forest",
            "n_estimators": 100,
            "max_depth": 5,
            "random_state": 42,
            "test_size": 0.2,
            "n_neighbors": 5,
        }

    # 2. Charger les données versionnées par DVC
    data_path = "data/raw/iris.csv"
    if not os.path.exists(data_path):
        # Fallback si le dossier data/raw n'a pas encore été généré
        from src.prepare_data import prepare
        prepare()

    df = pd.read_csv(data_path)
    X = df.drop("target", axis=1)
    y = df["target"]
    target_names = ["setosa", "versicolor", "virginica"]

    # 3. Séparation des données Train / Test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=params.get("test_size", 0.2),
        random_state=params.get("random_state", 42),
        stratify=y,
    )

    # 4. Configuration de MLflow (Tracking Store SQLite)
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
            random_state=params.get("random_state", 42),
        )

    # Récupérer l'empreinte DVC des données
    dvc_md5 = get_dvc_hash("data/raw/iris.csv.dvc")

    with mlflow.start_run(run_name=run_name):
        print(f"\n=======================================================")
        print(f" Démarrage du Run MLflow : {run_name}")
        print(f" Modèle : {model_type} | Empreinte DVC : {dvc_md5}")
        print(f"=======================================================")

        # 5. Enregistrer la traçabilité des données DVC dans MLflow
        if dvc_md5:
            mlflow.set_tag("dvc_data_version", dvc_md5)
            mlflow.log_param("dvc_data_md5", dvc_md5)

        # Journaliser le dataset dans l'onglet Datasets de MLflow
        try:
            dataset_source = mlflow.data.from_pandas(
                df,
                targets="target",
                name="iris.csv",
                digest=dvc_md5 if dvc_md5 else None,
            )
            mlflow.log_input(dataset_source, context="training")
        except Exception as e:
            print(f"Notice log_input: {e}")

        # Enregistrer le fichier CSV physique dans les artefacts du run
        if os.path.exists(data_path):
            mlflow.log_artifact(data_path, artifact_path="data")

        # 6. Entraînement et mesure du temps
        start_time = time.time()
        model.fit(X_train, y_train)
        training_duration = time.time() - start_time
        predictions = model.predict(X_test)

        # 7. Calcul des métriques d'évaluation
        acc = float(accuracy_score(y_test, predictions))
        prec = float(precision_score(y_test, predictions, average="weighted"))
        rec = float(recall_score(y_test, predictions, average="weighted"))
        f1 = float(f1_score(y_test, predictions, average="weighted"))
        cm = confusion_matrix(y_test, predictions)

        metrics = {
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1_score": f1,
            "training_duration_seconds": float(training_duration),
        }

        print(f"  Durée d'entraînement : {training_duration:.4f} s")
        print(f"  Accuracy             : {acc:.4f}")
        print(f"  Precision            : {prec:.4f}")
        print(f"  Recall               : {rec:.4f}")
        print(f"  F1-Score             : {f1:.4f}")
        print("  Matrice de Confusion :")
        print(cm)

        # 8. Log des paramètres et métriques dans MLflow
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)

        # 9. Matrice de Confusion graphique sauvegardée en artefact MLflow
        fig, ax = plt.subplots(figsize=(6, 5))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
        disp.plot(cmap="Blues", ax=ax)
        ax.set_title(f"Confusion Matrix - {run_name} (Acc: {acc:.2%})")
        plt.tight_layout()
        mlflow.log_figure(fig, "confusion_matrix.png")
        plt.close(fig)

        # 10. Sauvegarde du modèle sérialisé pour DVC et MLflow
        os.makedirs("models", exist_ok=True)
        model_path = "models/model.pkl"
        joblib.dump(model, model_path)
        mlflow.log_artifact(model_path, artifact_path="model")

        # 11. Sauvegarde des métriques au format JSON pour DVC
        os.makedirs("reports", exist_ok=True)
        reports_path = "reports/metrics.json"
        with open(reports_path, "w") as f:
            json.dump(metrics, f, indent=4)

        print(f"[OK] Artefacts générés : {model_path}, {reports_path}, confusion_matrix.png")
        print(f"[OK] Run '{run_name}' enregistré dans MLflow Tracking avec succès !")


if __name__ == "__main__":
    train()
