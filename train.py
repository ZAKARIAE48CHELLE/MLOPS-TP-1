import os
import time

# Suppress MLflow banner hints
os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"

import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
)
from sklearn.model_selection import train_test_split


def run_experiment(n_estimators_list=[50, 100, 150], max_depth=None):
    """
    Entraîne plusieurs modèles RandomForest sur le jeu de données Iris,
    enregistre les hyperparamètres, la durée d'entraînement, les métriques
    et la matrice de confusion dans MLflow Tracking.
    """
    # 1. Charger les données Iris
    iris = load_iris()
    X, y = iris.data, iris.target
    target_names = list(iris.target_names)

    # 2. Diviser le jeu de données (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 3. Définir l'expérience MLflow
    mlflow.set_experiment("TP_MLflow_Iris")

    print(f"=== Expérience : TP_MLflow_Iris (Modèles à évaluer : {n_estimators_list}) ===")

    for n in n_estimators_list:
        run_name = f"RF_{n}" if max_depth is None else f"RF_{n}_depth_{max_depth}"
        with mlflow.start_run(run_name=run_name):
            print(f"\n--- Entraînement du modèle : {run_name} ---")

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

            # Enregistrement des paramètres (Défis complémentaires inclus)
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

            # Enregistrement du modèle avec signature et skops_trusted_types
            mlflow.sklearn.log_model(
                sk_model=model,
                name="iris_model",
                skops_trusted_types=["sklearn.tree._tree.Tree"],
            )

            print(f"  [OK] Run '{run_name}' enregistré dans MLflow Tracking avec artefacts.")


if __name__ == "__main__":
    # Exécution des runs requis par le TP (50, 100) et du défi complémentaire (150)
    run_experiment(n_estimators_list=[50, 100, 150])
