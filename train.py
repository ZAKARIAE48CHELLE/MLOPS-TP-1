import os

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


def main():
    # 1. Charger les données Iris
    iris = load_iris()
    X, y = iris.data, iris.target
    target_names = list(iris.target_names)

    # 2. Diviser le jeu de données
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 3. Définir le nom de l'expérience (selon le TP)
    mlflow.set_experiment("TP_MLflow_Iris")

    # 4. Entraîner et enregistrer les modèles (RF_50 et RF_100)
    for n in [50, 100]:
        run_name = f"RF_{n}"
        with mlflow.start_run(run_name=run_name):
            print(f"\n--- Entraînement du modèle : {run_name} ---")

            model = RandomForestClassifier(n_estimators=n, random_state=42)
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)

            # Calcul des métriques
            acc = accuracy_score(y_test, predictions)
            f1 = f1_score(y_test, predictions, average="macro")
            cm = confusion_matrix(y_test, predictions)

            print(f"Accuracy : {acc:.4f}")
            print(f"F1-Macro : {f1:.4f}")
            print("Matrice de Confusion :")
            print(cm)

            # Enregistrement des paramètres
            mlflow.log_param("n_estimators", n)
            mlflow.log_param("test_size", 0.2)

            # Enregistrement des métriques
            mlflow.log_metric("accuracy", acc)
            mlflow.log_metric("f1_macro", f1)

            # Génération et enregistrement de la matrice de confusion (Confusion Matrix)
            fig, ax = plt.subplots(figsize=(6, 5))
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
            disp.plot(cmap="Blues", ax=ax)
            ax.set_title(f"Matrice de Confusion - {run_name} (Acc: {acc:.2%})")
            plt.tight_layout()

            # Enregistrer la figure dans les artefacts MLflow
            mlflow.log_figure(fig, "confusion_matrix.png")
            plt.close(fig)

            # Enregistrement du modèle avec skops_trusted_types
            mlflow.sklearn.log_model(
                sk_model=model,
                name="iris_model",
                skops_trusted_types=["sklearn.tree._tree.Tree"],
            )

            print(f"Run {run_name} terminé et enregistré dans MLflow avec sa matrice de confusion !")


if __name__ == "__main__":
    main()
