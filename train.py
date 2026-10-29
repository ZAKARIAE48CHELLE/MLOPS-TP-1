import os
import time

# Suppress MLflow banner hints
os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"

import matplotlib.pyplot as plt
import mlflow
import mlflow.data
import mlflow.sklearn
import pandas as pd
import yaml
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
)
from sklearn.model_selection import train_test_split


def get_dvc_data_info(dvc_file_path="Data/iris.csv.dvc"):
    """Récupère l'empreinte numérique (hash MD5) de DVC si disponible."""
    if os.path.exists(dvc_file_path):
        try:
            with open(dvc_file_path, "r") as f:
                dvc_meta = yaml.safe_load(f)
                return dvc_meta.get("outs", [{}])[0].get("md5")
        except Exception:
            pass
    return None


def run_experiment(n_estimators_list=[50, 100, 150], max_depth=None):
    """
    Entraîne plusieurs modèles RandomForest sur le jeu de données Iris,
    enregistre les hyperparamètres, la durée d'entraînement, les métriques,
    la matrice de confusion ET la version des données DVC dans MLflow Tracking.
    """
    # 1. Charger les données Iris (depuis le CSV versionné par DVC ou scikit-learn)
    data_csv = "Data/iris.csv"
    if os.path.exists(data_csv):
        df = pd.read_csv(data_csv)
        feature_cols = [c for c in df.columns if c not in ["target", "species"]]
        X = df[feature_cols].values
        y = df["target"].values
        target_names = ["setosa", "versicolor", "virginica"]
    else:
        iris = load_iris()
        X, y = iris.data, iris.target
        target_names = list(iris.target_names)
        df = pd.DataFrame(X, columns=iris.feature_names)
        df["target"] = y

    # Récupération du hash de version DVC
    dvc_hash = get_dvc_data_info("Data/iris.csv.dvc")

    # 2. Diviser le jeu de données (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 3. Définir l'expérience MLflow
    mlflow.set_experiment("TP_MLflow_Iris")

    print(f"=== Expérience : TP_MLflow_Iris (Modèles à évaluer : {n_estimators_list}) ===")
    if dvc_hash:
        print(f"Version des données DVC détectée : {dvc_hash}")

    for n in n_estimators_list:
        run_name = f"RF_{n}" if max_depth is None else f"RF_{n}_depth_{max_depth}"
        with mlflow.start_run(run_name=run_name):
            print(f"\n--- Entraînement du modèle : {run_name} ---")

            # 4. Traçabilité de la version des données dans MLflow (DVC tracking)
            if dvc_hash:
                mlflow.set_tag("dvc_data_version", dvc_hash)
                mlflow.log_param("dvc_md5", dvc_hash)
                try:
                    mlflow_dataset = mlflow.data.from_pandas(
                        df,
                        targets="target",
                        name="iris_dataset",
                        digest=dvc_hash,
                    )
                    mlflow.log_input(mlflow_dataset, context="training")
                except Exception as e:
                    print(f"Notice: log_input ({e})")

            start_time = time.time()
            model = RandomForestClassifier(
                n_estimators=n,
                max_depth=max_depth,
                random_state=42,
            )
            model.fit(X_train, y_train)
            training_duration = time.time() - start_time

            # Prédictions et évaluation
            predictions = model.predict(X_test)
            acc = accuracy_score(y_test, predictions)
            f1 = f1_score(y_test, predictions, average="macro")
            cm = confusion_matrix(y_test, predictions)

            print(f"  Durée d'entraînement : {training_duration:.4f} s")
            print(f"  Accuracy             : {acc:.4f}")
            print(f"  F1-Macro             : {f1:.4f}")
            print("  Matrice de Confusion :")
            print(cm)

            # Enregistrement des paramètres
            mlflow.log_param("n_estimators", n)
            mlflow.log_param("max_depth", str(max_depth))
            mlflow.log_param("test_size", 0.2)
            mlflow.log_param("random_state", 42)

            # Enregistrement des métriques
            mlflow.log_metric("accuracy", acc)
            mlflow.log_metric("f1_macro", f1)
            mlflow.log_metric("training_duration_seconds", training_duration)

            # Génération et enregistrement du graphique de la Matrice de Confusion
            fig, ax = plt.subplots(figsize=(6, 5))
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
            disp.plot(cmap="Blues", ax=ax)
            ax.set_title(f"Matrice de Confusion - {run_name} (Acc: {acc:.2%})")
            plt.tight_layout()

            # Enregistrement du visuel dans les artefacts MLflow
            mlflow.log_figure(fig, "confusion_matrix.png")
            plt.close(fig)

            # Enregistrement du modèle avec skops_trusted_types
            mlflow.sklearn.log_model(
                sk_model=model,
                name="iris_model",
                skops_trusted_types=["sklearn.tree._tree.Tree"],
            )

            print(f"  [OK] Run '{run_name}' enregistré dans MLflow avec version DVC : {dvc_hash[:8]}...")


if __name__ == "__main__":
    run_experiment(n_estimators_list=[50, 100, 150])
