# COMPTE-RENDU — ATELIER PRATIQUE MLOps

### Intégration de Git + DVC + MLflow dans un workflow MLOps

*Cas pratique : Classification du dataset Iris avec Python et Scikit-learn*

---

**Nom et prénom :** Zakariae CHELLE
**Groupe :** 5IASDT G1
**Module :** Déploiement des applications intelligentes
**Dépôt GitHub :** [https://github.com/ZAKARIAE48CHELLE/MLOPS-TP-1.git](https://github.com/ZAKARIAE48CHELLE/MLOPS-TP-1.git)
**Date :** Novembre 2026
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

## 1. Objectifs de l'Atelier

Cet atelier pratique met en œuvre la complémentarité de la triade MLOps fondamentale :

1. **Git & GitHub** : Versioning rigoureux du code source, gestion des branches (`main`, `develop`), étiquetage des versions majeures (`v1.0`, `v2.0`) et collaboration CI/CD.
2. **DVC (Data Version Control)** : Versioning des données brutes (`data/raw/iris.csv`), configuration d'un stockage distant (remote storage), et orchestration d'un **pipeline ML reproductible** (`dvc.yaml`, `dvc.lock`).
3. **MLflow Tracking** : Journalisation fine des métriques (`accuracy`, `precision`, `recall`, `f1_score`), des hyperparamètres (`n_estimators`, `max_depth`, `random_state`, `test_size`), et des artefacts de modèles sous forme d'expériences comparables (`RandomForest`, `KNN`).
4. **Reproductibilité & Automatisation** : Démonstration de retour en arrière (checkout), reproduction d'exécution (`dvc repro`) et intégration continue via **GitHub Actions** (`.github/workflows/mlops.yml`).

---

## 2. Architecture et Répartition des Responsabilités

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              WORKFLOW MLOps                                 │
└─────────────────────────────────────────────────────────────────────────────┘
          │                                 │                              │
          ▼                                 ▼                              ▼
  ┌──────────────┐                  ┌──────────────┐               ┌──────────────┐
  │     Git      │                  │     DVC      │               │    MLflow    │
  ├──────────────┤                  ├──────────────┤               ├──────────────┤
  │ Code source  │                  │ Données raw  │               │ Tracking R&D │
  │ Scripts src/ │                  │ Cache local  │               │ Paramètres   │
  │ params.yaml  │                  │ Remote store │               │ Métriques    │
  │ dvc.yaml     │                  │ Modèles lourds│              │ Comparaison  │
  │ Pointers .dvc│                  │ Pipeline DAG │               │ Artefacts    │
  └──────────────┘                  └──────────────┘               └──────────────┘
```

| Outil / Composant      | Responsabilité dans le projet                                | Fichiers associés                                                  |
| :--------------------- | :------------------------------------------------------------ | :------------------------------------------------------------------ |
| **Git + GitHub** | Code source, orchestration, configurations et historique      | `src/`, `params.yaml`, `dvc.yaml`, `dvc.lock`, `.dvc/`    |
| **DVC**          | Stockage et versioning des données & artefacts de sortie     | `data/raw/iris.csv.dvc`, `models/model.pkl`, `../dvc-storage` |
| **MLflow**       | Suivi scientifique des expérimentations, comparaison de runs | `mlflow.db`, métriques, tags, runs `RandomForest` & `KNN`    |

---

## 3. Structure Finale du Projet

```text
MLOPS-1/
├── .dvc/
│   ├── .gitignore
│   └── config                  # Configuration du remote DVC 'storage' (../dvc-storage)
├── .github/
│   └── workflows/
│       └── mlops.yml           # Pipeline CI GitHub Actions (repro & metrics)
├── data/
│   └── raw/
│       ├── .gitignore          # Généré par DVC pour ignorer les CSV volumineux dans Git
│       ├── iris.csv            # Dataset Iris généré par src/prepare_data.py
│       ├── iris.csv.dvc        # Pointeur DVC contenant le hash MD5
│       └── winequality-red.csv.dvc # Deuxième dataset versionné avec DVC (TP 2)
├── models/
│   ├── .gitignore          # Généré par DVC (ignore model.pkl dans Git)
│   └── model.pkl           # Modèle entraîné et sérialisé par le pipeline DVC
├── reports/
│   └── metrics.json        # Métriques calculées et suivies par DVC
├── src/
│   ├── prepare_data.py     # Script d'extraction et de préparation du dataset
│   └── train.py            # Script d'entraînement, lecture params.yaml, tracking MLflow
├── params.yaml             # Hyperparamètres versionnés du pipeline
├── dvc.yaml                # Définition formelle des étapes du pipeline MLOps
├── dvc.lock                # État figé des dépendances et sorties pour reproductibilité
├── requirements.txt        # Dépendances de l'environnement Python
└── .gitignore              # Exclusions Git (.venv, mlruns, *.db)
```

---

## 4. Réalisation des Travaux Pratiques (TP 1 à TP 4)

### TP 1 – Gestion du code avec Git

- **Branches créées et publiées :**
  - `main` : Branche de production / version stable.
  - `develop` : Branche active d'intégration et de développement du pipeline.
- **Commits réalisés :** Plus de 5 commits structurés et traçables.
- **Tags de versions créés :**
  - `v1.0` : Modèle Random Forest avec $100$ arbres (`n_estimators: 100`, `max_depth: 5`).
  - `v2.0` : Expérimentation avec $200$ arbres (`n_estimators: 200`, `max_depth: 8`).

### TP 2 – Versioning des données avec DVC

- **Premier dataset :** `data/raw/iris.csv` suivi via `data/raw/iris.csv.dvc`.
- **Deuxième dataset ajouté :** `data/raw/winequality-red.csv` suivi via `data/raw/winequality-red.csv.dvc`.
- **Remote storage configuré :** Stockage distant configuré vers `../dvc-storage`.
- **Synchronisation :** Données poussées avec succès via `dvc push` (3 fichiers dans le remote storage).

### TP 3 – Tracking avec MLflow

Les deux algorithmes ont été entraînés et journalisés dans l'expérience MLflow **`TP_Iris_Classification`** :

| Modèle                | Hyperparamètres                      |     Accuracy     |    Precision    |      Recall      |     F1-Score     |
| :--------------------- | :------------------------------------ | :--------------: | :--------------: | :--------------: | :--------------: |
| **RandomForest** | `n_estimators=100`, `max_depth=5` | **0.9333** | **0.9333** | **0.9333** | **0.9333** |
| **RandomForest** | `n_estimators=200`, `max_depth=8` | **0.9000** | **0.9024** | **0.9000** | **0.8997** |
| **KNN**          | `n_neighbors=5`                     | **1.0000** | **1.0000** | **1.0000** | **1.0000** |

> **Constat scientifique :** Augmenter le nombre d'arbres de 100 à 200 sur le dataset Iris n'augmente pas la performance (légère baisse due à un sur-apprentissage local sur un échantillon réduit). KNN avec $k=5$ atteint une séparation parfaite sur cet échantillon de test stratifié.

### TP 4 – Pipeline intégré DVC + MLflow

- Définition du stage `train` dans `dvc.yaml` reliant les dépendances (`src/train.py`, `data/raw/iris.csv`), les paramètres (`params.yaml`) et les sorties (`models/model.pkl`, `reports/metrics.json`).
- Exécution réussie via `dvc repro`.
- Inspection du graphe orienté acyclique avec `dvc dag` et affichage des métriques avec `dvc metrics show`.

---

## 5. Réponses Détaillées aux Questions de Réflexion (Section 16)

### 1. Quelle est la différence entre Git et DVC ?

> **Réponse :**
>
> - **Git** est conçu pour suivre des fichiers texte légers (code source, scripts, configurations). Il crée un historique différentiel ligne par ligne (`diff`), ce qui le rend totalement inadapté aux gros volumes ou aux binaires dont la moindre modification gonfle exponentiellement la taille du dépôt (`.git`).
> - **DVC (Data Version Control)** est spécifiquement pensé pour gérer les fichiers volumineux (jeux de données, modèles entraînés, checkpoints). DVC stocke le contenu réel dans un cache externe ou un remote storage (local, S3, GDrive, Azure Blob) et ne confie à Git qu'un fichier pointeur textuel léger (`.dvc`) contenant l'empreinte de hachage (MD5).

### 2. Pourquoi éviter de stocker un dataset volumineux directement dans Git ?

> **Réponse :**
>
> 1. **Explosion de la taille du dépôt :** Chaque commit modifiant un binaire stocke une copie intégrale de ce fichier dans l'historique `.git`.
> 2. **Dégradation des performances :** Les opérations quotidiennes (`clone`, `pull`, `push`, `status`) deviennent excessivement lentes.
> 3. **Limites d'hébergement :** Les plateformes comme GitHub bloquent ou rejettent les fichiers dépassant $100\text{ Mo}$.
> 4. **Inadéquation du format :** Git ne sait pas effectuer de comparaison unitaire (diffing) efficace sur des fichiers binaires ou des gigaoctets de CSV/images.

### 3. Quelle est la différence entre `dvc add` et `dvc repro` ?

> **Réponse :**
>
> - **`dvc add <fichier>` :** Est une commande manuelle de suivi statique. Elle calcule le hash du fichier spécifié, place les données dans le cache DVC et génère un fichier pointeur `.dvc`. Elle est utilisée pour les données sources externes (ex: `data/raw/iris.csv`).
> - **`dvc repro` :** Est une commande d'exécution et d'orchestration dynamique du pipeline décrit dans `dvc.yaml`. Elle analyse l'arbre de dépendances (DAG) et ré-exécute automatiquement **uniquement les étapes dont le code, les données d'entrée ou les paramètres ont changé**. Elle régénère les sorties et met à jour `dvc.lock`.

### 4. Quel est le rôle du fichier `dvc.lock` ?

> **Réponse :**
> Le fichier `dvc.lock` est l'équivalent MLOps d'un fichier de verrouillage de dépendances (comme `package-lock.json` ou `poetry.lock`). Il enregistre pour chaque étape du pipeline :
>
> - Le hash exact des dépendances (`deps`),
> - Les valeurs exactes des paramètres utilisés (`params`),
> - Le hash et la taille des sorties générées (`outs`).
>   Commité dans Git, `dvc.lock` garantit la **reproductibilité déterministe** : DVC sait exactement quel artefact correspond à quel état précis du projet sans ambiguïté.

### 5. Comment MLflow permet-il de comparer deux expérimentations ?

> **Réponse :**
> MLflow Tracking centralise toutes les exécutions sous une même expérience (`TP_Iris_Classification`). Dans l'interface Web (`mlflow ui`) :
>
> 1. On peut cocher plusieurs runs (par exemple `RandomForest_n_100` et `KNN_k_5`).
> 2. Le bouton **Compare** affiche un tableau comparatif côte à côte de tous les hyperparamètres et métriques.
> 3. MLflow génère des graphiques interactifs (diagrammes en barres, courbes de dispersion, coordonnées parallèles, matrices de confusion) permettant d'identifier immédiatement le modèle le plus performant selon les métriques cibles.

### 6. Comment retrouver les données et le modèle associés à un commit Git précis ?

> **Réponse :**
> La traçabilité croisée Git $\leftrightarrow$ DVC $\leftrightarrow$ MLflow s'effectue ainsi :
>
> 1. On se positionne sur le commit Git souhaité : `git checkout <commit_hash_ou_tag>`.
> 2. Ce commit restaure l'état exact des fichiers pointeurs : `data/raw/iris.csv.dvc`, `dvc.lock`, et `params.yaml`.
> 3. En exécutant `dvc checkout` (ou `dvc pull`), DVC télécharge et replace dans le dossier de travail le dataset exact et le modèle `model.pkl` correspondant à ce commit.
> 4. Dans MLflow, le `git_commit` est enregistré comme métadonnée native du run, permettant d'inspecter les métriques associées à ce même commit.

### 7. Pourquoi la reproductibilité est-elle importante dans un projet d’intelligence artificielle ?

> **Réponse :**
>
> 1. **Auditabilité et Conformité :** Pouvoir justifier et reproduire à tout moment les décisions d'un modèle en production.
> 2. **Débogage et Maintenance :** Lorsqu'une régression ou un comportement anormal survient en production, il est indispensable de pouvoir réinstancier l'environnement, les données et les poids exacts qui ont produit ce modèle.
> 3. **Collaboration en équipe :** Éviter l'effet « ça fonctionne sur ma machine », en garantissant que chaque membre de l'équipe obtient rigoureusement les mêmes résultats d'entraînement.
> 4. **Transfert en Production (CI/CD) :** Permettre aux pipelines automatisés d'entraîner, tester et déployer des modèles en toute fiabilité sans intervention manuelle risquée.

### 8. Quelle est la différence entre le versioning des modèles avec DVC et leur suivi avec MLflow ?

> **Réponse :**
>
> - **DVC (Data & Artifact Versioning) :** Versionne le modèle comme un **artefact binaire volumineux au sein du pipeline de données**. Il assure que le fichier physique `model.pkl` est lié de façon déterministe au code et aux données d'entrée via `dvc.lock`.
> - **MLflow (Experiment Tracking & Model Lifecycle) :** Suit le modèle sous un angle **scientifique et opérationnel**. Il enregistre le contexte d'entraînement complet (paramètres, métriques, courbes d'apprentissage, signature des entrées/sorties). Via son **Model Registry**, MLflow gère également les étapes de gouvernance (transitions entre *Staging*, *Production* et *Archived*).
> - **Complémentarité :** DVC garantit la reproductibilité de la fabrication du modèle ; MLflow fournit l'évaluation comparative et la gouvernance de son cycle de vie.

---

## 6. Synthèse des Commandes Essentielles

```powershell
# 1. Pipeline DVC
dvc repro                     # Exécute le pipeline et met à jour dvc.lock
dvc dag                       # Visualise les dépendances du pipeline
dvc metrics show              # Affiche les métriques calculées

# 2. Stockage DVC
dvc push                      # Envoie les données/modèles vers le remote storage
dvc pull                      # Récupère les données/modèles depuis le remote storage
dvc status                    # Vérifie la synchronisation des données

# 3. MLflow Tracking
mlflow ui --backend-store-uri sqlite:///mlflow.db   # Lance l'interface web (http://127.0.0.1:5000)

# 4. Git & GitHub
git checkout develop          # Bascule sur la branche develop
git checkout main             # Bascule sur la branche principale
git push origin main develop --tags   # Synchronise les branches et tags sur GitHub
```
