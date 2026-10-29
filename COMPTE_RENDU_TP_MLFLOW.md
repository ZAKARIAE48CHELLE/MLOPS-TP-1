# TRAVAUX PRATIQUES MLOps

### Chapitre 3 — Versioning et suivi des modèles avec MLflow

*Module : Déploiement des applications intelligentes*

---

**Nom et prénom :** Zakariae CHELLE
**Groupe :** 5IASDT G1
**Module :** Déploiement des applications intelligentes
**Date :** Octobre 2026
-------------------------------------------------------------------------------------------------------------------------

## 1. Objectifs du TP

Le présent travail pratique a pour objectif la mise en place d'un pipeline complet de suivi et de gestion du cycle de vie des modèles d'apprentissage automatique avec **MLflow**, articulé autour des axes suivants :

1. Mise en place d'un environnement de développement Python isolé et installation des outils MLOps (`mlflow`, `scikit-learn`, `pandas`, `matplotlib`).
2. Développement d'un script d'entraînement (`train.py`) entraînant plusieurs modèles de classification (`RandomForestClassifier`) sur le jeu de données **Iris**.
3. Traçabilité des hyperparamètres, des métriques d'évaluation et des artefacts (modèle sérialisé et graphiques de matrices de confusion) via **MLflow Tracking**.
4. Exploration, visualisation et comparaison des exécutions (runs) au sein de l'interface graphique de MLflow.
5. Rechargement programmatique d'un modèle sauvegardé via son identifiant unique (`RUN_ID`) dans un script dédié (`load_model.py`) pour effectuer de l'inférence.
6. Résolution des défis complémentaires : variation de la taille de la forêt (`n_estimators = 50, 100, 150`), mesure de la durée d'entraînement, enregistrement de la matrice de confusion et analyse de la reproductibilité.

---

## 2. Environnement et configuration

- **Système d'exploitation :** Windows (PowerShell)
- **Interpréteur :** Python 3.13 (Environnement virtuel local `.venv`)
- **Dépendances clés :**
  - `mlflow` : 3.17.2
  - `scikit-learn` : 1.9.1
  - `pandas` : 3.1.0
  - `matplotlib` : 3.11.2

Commandes exécutées pour initialiser l'environnement :

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install mlflow scikit-learn pandas matplotlib
pip freeze > requirements.txt
```

---

## 3. Scripts développés

### 3.1 Script d'entraînement : `train.py`

Le script charge le jeu de données Iris, effectue une séparation stratifiée en ensembles d'entraînement (80%) et de test (20%), puis entraîne successivement plusieurs configurations du modèle `RandomForestClassifier`. Pour chaque run, il consigne dans MLflow :

- Les paramètres : `n_estimators`, `max_depth`, `test_size`, `random_state`.
- Les métriques : `accuracy`, `f1_macro`, `training_duration_seconds`.
- Les artefacts : la matrice de confusion exportée sous format PNG (`confusion_matrix.png`) et le modèle sérialisé (`iris_model`).

### 3.2 Script de rechargement : `load_model.py`

Le script permet de recharger le modèle sauvegardé dans MLflow à partir de son `RUN_ID` via l'URI `runs:/<RUN_ID>/iris_model`, puis vérifie son fonctionnement en prédisant sur le jeu de données complet.

---

## 4. Résultats expérimentaux et tableau comparatif

L'expérience menée sous le nom **`TP_MLflow_Iris`** a produit les résultats suivants sur l'échantillon de test (30 exemples) :

| Run Name         | Run ID                               | `n_estimators` | `test_size` |     Accuracy     |     F1-Macro     | Durée d'entraînement (s) | Modèle Artefact                          |
| :--------------- | :----------------------------------- | :--------------: | :-----------: | :--------------: | :--------------: | :------------------------: | :---------------------------------------- |
| **RF_50**  | `96e65a5f5d5344cfa48d29d172a35d8f` |        50        |     0.20     | **0.9000** | **0.8997** |     **0.0707 s**     | `iris_model` + `confusion_matrix.png` |
| **RF_100** | `512d868c9ebb4b2490cd3c189a902d9d` |       100       |     0.20     | **0.9000** | **0.8997** |     **0.1315 s**     | `iris_model` + `confusion_matrix.png` |
| **RF_150** | `936181c45b3b4798937d6147a4bce033` |       150       |     0.20     | **0.9000** | **0.8997** |     **0.3875 s**     | `iris_model` + `confusion_matrix.png` |

### Matrice de Confusion obtenue (sur test set de 30 échantillons) :

$$
\begin{pmatrix}
10 & 0 & 0 \\
0 & 9 & 1 \\
0 & 2 & 8
\end{pmatrix}
$$

- **Classe 0 (Setosa) :** 10/10 correctement classifiés (100% de précision).
- **Classe 1 (Versicolor) :** 9/10 correctement classifiés (1 erreur vers Virginica).
- **Classe 2 (Virginica) :** 8/10 correctement classifiés (2 erreurs vers Versicolor).

### Rechargement du modèle (`load_model.py`) :

Exécution avec le `RUN_ID = 512d868c9ebb4b2490cd3c189a902d9d` (Run `RF_100`) :

```text
Chargement du modèle depuis : runs:/512d868c9ebb4b2490cd3c189a902d9d/iris_model
Modèle rechargé avec succès (Run ID : 512d868c9ebb4b2490cd3c189a902d9d)
Accuracy sur Iris complet : 0.9800
```

Le modèle rechargé restitue fidèlement ses capacités d'inférence (98% d'accuracy sur l'ensemble global de 150 échantillons).

---

## 5. Défis complémentaires traités

1. **Ajout de `n_estimators = 150` et comparaison des runs :**L'ajout de 150 arbres stabilise la variance mais ne procure aucun gain d'accuracy sur le dataset Iris ($0.9000$), tout en multipliant le temps d'entraînement par plus de 5 par rapport à 50 arbres ($0.3875\text{ s}$ contre $0.0707\text{ s}$). Le modèle `RF_50` représente donc le meilleur compromis performance/ressources.
2. **Durée d'entraînement enregistrée comme métrique :**Intégration de la métrique `training_duration_seconds` calculée avec `time.time()`, offrant une visibilité directe sur le coût computationnel dans l'interface de comparaison MLflow.
3. **Matrice de confusion enregistrée en tant qu'artefact visuel :**Génération dynamique via `ConfusionMatrixDisplay` et enregistrement automatique via `mlflow.log_figure(fig, "confusion_matrix.png")`.
4. **Pourquoi conserver les paramètres, métriques et artefacts facilite la reproductibilité :**
   Conserver l'ensemble de ces informations garantit l'auditabilité et la traçabilité intégrale du modèle : tout chercheur ou ingénieur peut ré-exécuter le code avec les mêmes hyperparamètres, comparer les performances avec des versions antérieures et déployer exactement le binaire validé, éliminant ainsi les aléas de divergence entre l'environnement d'expérimentation et la production.

---

## 6. Réponses détaillées aux Questions de Validation

### Question 1 : Quelle différence existe entre une expérience et un run dans MLflow ?

> **Réponse :**
>
> - **L'expérience (Experiment) :** Est le niveau d'organisation logique le plus haut dans MLflow. Elle sert d'enveloppe thématique regroupant l'ensemble des essais menés pour résoudre un problème donné (ex. `TP_MLflow_Iris`).
> - **Le run (Exécution) :** Représente une exécution unitaire et datée d'un code d'entraînement au sein d'une expérience. Chaque run possède son propre identifiant unique (`RUN_ID`) et consigne un ensemble spécifique d'hyperparamètres, de métriques d'évaluation et d'artefacts (modèles, graphiques, logs). Une expérience contient donc un historique de plusieurs runs.

### Question 2 : Quels paramètres et quelles métriques le script enregistre-t-il ?

> **Réponse :**
>
> - **Paramètres (`log_param`) :**
>   - `n_estimators` : Nombre d'arbres décisionnels constitutifs de la forêt aléatoire (ex. 50, 100, 150).
>   - `max_depth` : Profondeur maximale des arbres pour contrôler la complexité.
>   - `test_size` : Proportion du jeu de données allouée au test ($0.20$).
>   - `random_state` : Graine aléatoire fixée à $42$ pour garantir la stricte reproductibilité des résultats.
> - **Métriques (`log_metric`) :**
>   - `accuracy` : Taux de prédictions exactes sur le jeu de test.
>   - `f1_macro` : Moyenne non pondérée des scores F1 par classe, évaluant l'équilibre de classification.
>   - `training_duration_seconds` : Durée temporelle requise pour l'entraînement de l'algorithme.

### Question 3 : Où peut-on retrouver le modèle sauvegardé ?

> **Réponse :**
> Le modèle sauvegardé est accessible à deux emplacements :
>
> 1. **Dans le système de fichiers (Stockage local / Backend Store) :** Dans l'arborescence des artefacts générée par MLflow :
>    `mlruns/<experiment_id>/<run_id>/artifacts/iris_model/` (ou `mlartifacts/`). Ce dossier contient le fichier de modèle sérialisé (`model.skops` ou `model.pkl`), les fichiers d'environnement (`conda.yaml`, `requirements.txt`, `python_env.yaml`) et les métadonnées (`MLmodel`).
> 2. **Dans l'interface Web MLflow :** En naviguant dans l'expérience concernée, en cliquant sur le run visé, puis dans la section **Artifacts** sous le dossier `iris_model`.

### Question 4 : Pourquoi faut-il conserver l’identifiant du run (RUN_ID) ?

> **Réponse :**
> Le `RUN_ID` est la clé primaire immuable (UUID de 32 caractères) qui identifie formellement une exécution donnée. Il est indispensable de le conserver car :
>
> - Il permet de **recharger le modèle exact** en mémoire ou sur un serveur de prédiction sans risque de confusion via l'URI `runs:/<RUN_ID>/iris_model`.
> - Il assure la **traçabilité (lineage) et la gouvernance MLOps** : on peut relier n'importe quelle prédiction en production aux données d'entraînement, au commit Git exact et aux scores initiaux validés lors de ce run.

### Question 5 : Quelle est la différence entre MLflow Tracking et un registre de modèles (Model Registry) ?

> **Réponse :**
>
> - **MLflow Tracking :** Est un outil d'observation et de carnet de bord pour la phase de recherche & développement. Il enregistre passivement tous les essais menés (même les modèles non concluants), permettant aux data scientists d'analyser, comparer et itérer.
> - **MLflow Model Registry :** Est un système de gestion centralisé pour les modèles destinés à la mise en production. Il offre des fonctionnalités de gouvernance que Tracking ne propose pas : le versionnement formel des modèles promus (v1, v2...), la gestion des cycles de vie par étapes (Staging, Production, Archived), le suivi des revues/validations humaines et l'intégration continue (CI/CD) vers les environnements cibles.

---

## 7. Livrables et statut

- [X] **`train.py`** : Script d'entraînement complet avec `RandomForestClassifier`, gestion des runs `RF_50`, `RF_100`, `RF_150`, calcul et log des métriques, matrice de confusion et artefacts.
- [X] **`load_model.py`** : Script de rechargement fonctionnel et validé par son `RUN_ID`.
- [X] **Tableau comparatif des métriques** : Complété et analysé.
- [X] **Réponses aux questions de validation** : Complétées avec rigueur technique.
- [X] **Dépôt Git** : Versions tracées et synchronisées.
