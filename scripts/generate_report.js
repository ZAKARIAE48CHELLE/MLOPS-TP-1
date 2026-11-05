const fs = require('fs');
const path = require('path');
const {
  Document,
  Packer,
  Paragraph,
  TextRun,
  Table,
  TableRow,
  TableCell,
  WidthType,
  ShadingType,
  BorderStyle,
  AlignmentType,
  Footer,
  PageNumber,
  ImageRun
} = require('docx');

const FONT = 'Calibri';
const MONO = 'Consolas';
const GREY = '595959';
const BLACK = '000000';
const LIGHT_GREY = 'D9D9D9';
const BORDER_GREY = '999999';
const W = 9638; // 11906 - 2 * 1134 = 9638 DXA available width

const run = (text, o = {}) =>
  new TextRun({
    text,
    font: o.font || FONT,
    size: o.size || 22, // 11pt
    color: o.color || BLACK,
    ...o,
  });

const RULE = { style: BorderStyle.SINGLE, size: 6, color: BLACK, space: 6 };

const titleBlock = (module, title, sub) => [
  new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { after: 60 },
    children: [run(module, { size: 20, color: GREY })],
  }),
  new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { after: 60 },
    children: [run(title, { bold: true, size: 32 })],
  }),
  new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { after: 240 },
    children: [run(sub, { size: 24 })],
  }),
  new Paragraph({
    spacing: { after: 40 },
    border: { top: RULE },
    tabStops: [{ type: 'right', position: W }],
    children: [
      run('Nom et prénom : '),
      run('Zakariae CHELLE', { bold: true }),
      run('\tGroupe : '),
      run('5IASDT G1', { bold: true }),
    ],
  }),
  new Paragraph({
    spacing: { after: 120 },
    border: { bottom: RULE },
    children: [
      run('Module : '),
      run(module, { bold: true }),
      run('     |     Dépôt GitHub : '),
      run('https://github.com/ZAKARIAE48CHELLE/MLOPS-TP-1', { italics: true }),
    ],
  }),
];

const rich = (t, base = {}) =>
  t
    .split(/(`[^`]+`)/)
    .filter(Boolean)
    .map((s) =>
      s.startsWith('`')
        ? run(s.slice(1, -1), { ...base, font: MONO, size: 20 })
        : run(s, base)
    );

const intro = (t) =>
  new Paragraph({
    spacing: { before: 120, after: 120 },
    children: [run(t, { italics: true, color: GREY })],
  });

const partHeader = (title) =>
  new Paragraph({
    keepNext: true,
    keepLines: true,
    spacing: { before: 240, after: 120 },
    border: {
      bottom: { style: BorderStyle.SINGLE, size: 8, color: BLACK, space: 4 },
    },
    children: [run(title, { bold: true, size: 28 })],
  });

const sectionHeader = (title) =>
  new Paragraph({
    keepNext: true,
    keepLines: true,
    spacing: { before: 180, after: 80 },
    border: {
      bottom: { style: BorderStyle.SINGLE, size: 4, color: BORDER_GREY, space: 4 },
    },
    children: [run(title, { bold: true, size: 24 })],
  });

const question = (n, t) =>
  new Paragraph({
    keepNext: true,
    spacing: { before: 200, after: 60 },
    indent: { left: 360, hanging: 360 },
    tabStops: [{ type: 'left', position: 360 }],
    children: [run(`${n}.\t`, { bold: true }), run(t, { bold: true })],
  });

const answer = (lines, k = false) =>
  lines.map((l, i) =>
    new Paragraph({
      spacing: { after: 60, line: 288 },
      indent: { left: 360 },
      keepNext: k || i < lines.length - 1,
      children: i === 0 ? [run('Réponse : ', { bold: true, underline: {} }), ...rich(l)] : rich(l),
    })
  );

const tip = (t, label = 'Pour comprendre') =>
  new Paragraph({
    spacing: { before: 80, after: 80 },
    indent: { left: 360 },
    border: {
      left: { style: BorderStyle.SINGLE, size: 12, color: '7F7F7F', space: 8 },
    },
    children: [run(label + ' : ', { bold: true, italics: true }), ...rich(t, { italics: true })],
  });

const code = (lines) => {
  const b = { style: BorderStyle.SINGLE, size: 4, color: BORDER_GREY };
  return new Table({
    width: { size: W - 360, type: WidthType.DXA },
    columnWidths: [W - 360],
    indent: { size: 360, type: WidthType.DXA },
    rows: [
      new TableRow({
        cantSplit: true,
        children: [
          new TableCell({
            width: { size: W - 360, type: WidthType.DXA },
            borders: { top: b, bottom: b, left: b, right: b },
            margins: { top: 80, bottom: 80, left: 160, right: 160 },
            children: lines.map((l) =>
              new Paragraph({
                keepNext: true,
                spacing: { after: 0, line: 240 },
                children: [run(l || ' ', { font: MONO, size: 18 })],
              })
            ),
          }),
        ],
      }),
    ],
  });
};

const createTable = (headers, rows, colWidths) => {
  const borderThin = { style: BorderStyle.SINGLE, size: 4, color: BLACK };
  const borderHeader = { style: BorderStyle.SINGLE, size: 8, color: BLACK };

  const headerRow = new TableRow({
    cantSplit: true,
    children: headers.map((h, i) =>
      new TableCell({
        width: { size: colWidths[i], type: WidthType.DXA },
        shading: { fill: LIGHT_GREY, type: ShadingType.CLEAR },
        borders: {
          top: borderHeader,
          bottom: borderHeader,
          left: borderThin,
          right: borderThin,
        },
        margins: { top: 80, bottom: 80, left: 100, right: 100 },
        children: [
          new Paragraph({
            keepNext: true,
            alignment: AlignmentType.CENTER,
            spacing: { after: 0 },
            children: [run(h, { bold: true, size: 20 })],
          }),
        ],
      })
    ),
  });

  const dataRows = rows.map((r) =>
    new TableRow({
      cantSplit: true,
      children: r.map((c, i) =>
        new TableCell({
          width: { size: colWidths[i], type: WidthType.DXA },
          borders: {
            top: borderThin,
            bottom: borderThin,
            left: borderThin,
            right: borderThin,
          },
          margins: { top: 60, bottom: 60, left: 100, right: 100 },
          children: [
            new Paragraph({
              keepNext: true,
              alignment: i === 0 ? AlignmentType.LEFT : AlignmentType.CENTER,
              spacing: { after: 0 },
              children: rich(c, { size: 20 }),
            }),
          ],
        })
      ),
    })
  );

  return new Table({
    width: { size: W, type: WidthType.DXA },
    columnWidths: colWidths,
    rows: [headerRow, ...dataRows],
  });
};

const imageBlock = (imgBuffer, caption, width = 500, height = 321) => [
  new Paragraph({
    alignment: AlignmentType.CENTER,
    keepNext: true,
    spacing: { before: 140, after: 60 },
    children: [
      new ImageRun({
        data: imgBuffer,
        transformation: { width, height },
      }),
    ],
  }),
  new Paragraph({
    alignment: AlignmentType.CENTER,
    keepNext: true,
    spacing: { after: 140 },
    children: [run(caption, { italics: true, size: 18, color: GREY })],
  }),
];

const footer = (module) =>
  new Footer({
    children: [
      new Paragraph({
        alignment: AlignmentType.CENTER,
        border: {
          top: { style: BorderStyle.SINGLE, size: 4, color: BORDER_GREY, space: 6 },
        },
        children: [
          run(`Zakariae CHELLE · 5IASDT G1 · ${module}          Page `, {
            size: 16,
            color: GREY,
          }),
          new TextRun({
            children: [PageNumber.CURRENT],
            font: FONT,
            size: 16,
            color: GREY,
          }),
          run(' / ', { size: 16, color: GREY }),
          new TextRun({
            children: [PageNumber.TOTAL_PAGES],
            font: FONT,
            size: 16,
            color: GREY,
          }),
        ],
      }),
    ],
  });

async function generate() {
  const moduleName = 'Déploiement des applications intelligentes';
  
  // Lecture des images de captures d'écran
  const cmBuffer = fs.readFileSync(path.resolve('reports/confusion_matrix.png'));
  const screenRunsBuffer = fs.readFileSync(path.resolve('reports/mlflow_runs_list.png'));
  const screenDvcBuffer = fs.readFileSync(path.resolve('reports/mlflow_dvc_tracking.png'));
  const screenArtBuffer = fs.readFileSync(path.resolve('reports/mlflow_artifacts.png'));

  const children = [
    ...titleBlock(
      moduleName,
      'ATELIER PRATIQUE MLOps',
      'Intégration de Git + DVC + MLflow dans un workflow reproductible'
    ),

    intro(
      'Ce compte-rendu documente la mise en place complète d’un cycle de vie MLOps intégrant le versioning du code source (Git), le versioning et l’orchestration des pipelines de données (DVC), et le suivi rigoureux des expérimentations scientifiques (MLflow) sur le dataset Iris.'
    ),

    // PARTIE 1
    partHeader('Partie 1 — Architecture et Rôles des Outils MLOps'),
    new Paragraph({
      spacing: { after: 100 },
      children: [
        run(
          'L’industrialisation des modèles d’apprentissage automatique requiert une stricte séparation des responsabilités entre le code, les données volumineuses et les métadonnées expérimentales :'
        ),
      ],
    }),

    createTable(
      ['Composant', 'Responsabilité principale', 'Fichiers suivis', 'Avantage clé'],
      [
        [
          'Git + GitHub',
          'Versioning du code source, scripts et configurations',
          'src/*.py, params.yaml, dvc.yaml, .dvc/',
          'Traçabilité fine, branches et collaboration',
        ],
        [
          'DVC',
          'Gestion des données, modèles volumineux et DAG',
          'data/raw/iris.csv.dvc, models/model.pkl, cache',
          'Reproductibilité sans encombrer Git',
        ],
        [
          'MLflow Tracking',
          'Journalisation des métriques, paramètres et artefacts',
          'sqlite:///mlflow.db, runs RF et KNN',
          'Comparaison scientifique et aide à la décision',
        ],
      ],
      [2200, 2800, 2400, 2238]
    ),

    tip(
      'Git gère les fichiers texte légers (code, configs) ; DVC gère les fichiers volumineux (datasets, modèles sérialisés) via des pointeurs MD5 ; MLflow consigne l’historique des expérimentations et leurs métriques.',
      'Rappel méthodologique'
    ),

    sectionHeader('1.1  Arborescence normalisée du projet'),
    code([
      'MLOPS-1/',
      '├── .dvc/                  # Configuration DVC et remote de stockage (../dvc-storage)',
      '├── .github/workflows/    # Pipeline CI/CD automatisé (GitHub Actions mlops.yml)',
      '├── data/raw/             # Données sources (iris.csv, iris.csv.dvc)',
      '├── models/               # Modèles entraînés suivis par DVC (model.pkl)',
      '├── reports/              # Métriques (metrics.json) et visualisations',
      '├── src/                  # Code source (prepare_data.py, train.py)',
      '├── params.yaml           # Paramètres configurables de l’entraînement',
      '├── dvc.yaml / dvc.lock   # Définition et état figé du pipeline reproductible',
      '├── load_model.py         # Script d’inférence et de rechargement',
      '└── requirements.txt      # Dépendances logicielles du projet',
    ]),

    // PARTIE 2
    partHeader('Partie 2 — Réalisation des Travaux Pratiques (TP 1 à TP 4)'),

    sectionHeader('2.1  TP 1 — Gestion du code source avec Git'),
    new Paragraph({
      spacing: { after: 60 },
      children: [
        run(
          'Un historique de commits rigoureux a été établi avec une séparation nette des branches de travail :'
        ),
      ],
    }),
    new Paragraph({
      spacing: { after: 40 },
      indent: { left: 360 },
      children: [
        run('• Branche ', { bold: true }),
        run('main', { font: MONO }),
        run(' : branche principale stable recevant les versions validées.'),
      ],
    }),
    new Paragraph({
      spacing: { after: 60 },
      indent: { left: 360 },
      children: [
        run('• Branche ', { bold: true }),
        run('develop', { font: MONO }),
        run(' : branche d’intégration pour le développement et l’orchestration du pipeline.'),
      ],
    }),
    new Paragraph({
      spacing: { after: 80 },
      indent: { left: 360 },
      children: [
        run('• Tags Git : ', { bold: true }),
        run('v1.0', { font: MONO, bold: true }),
        run(' (version Random Forest 100 arbres) et '),
        run('v2.0', { font: MONO, bold: true }),
        run(' (version Random Forest 200 arbres).'),
      ],
    }),

    sectionHeader('2.2  TP 2 — Versioning des données avec DVC'),
    new Paragraph({
      spacing: { after: 60 },
      children: [
        run(
          'Les fichiers de données ont été isolés sous le contrôle de DVC sans polluer l’historique Git :'
        ),
      ],
    }),
    code([
      '# Extraction et génération du dataset',
      'python src/prepare_data.py',
      '',
      '# Ajout sous contrôle DVC',
      'dvc add data/raw/iris.csv',
      'dvc add data/raw/winequality-red.csv   # Deuxième dataset (TP 2)',
      '',
      '# Configuration du stockage distant local',
      'dvc remote add -d storage ../dvc-storage',
      'dvc push',
    ]),
    tip(
      'La commande dvc push transfère les données vers le remote storage, tandis que Git ne conserve que le pointeur data/raw/iris.csv.dvc contenant l’empreinte MD5.',
      'Observation'
    ),

    sectionHeader('2.3  TP 3 — Tracking des expérimentations avec MLflow et intégration DVC'),
    new Paragraph({
      spacing: { after: 80 },
      children: [
        run(
          'Trois configurations de modèles ont été exécutées et tracées dans la base de données SQLite (sqlite:///mlflow.db) sous l’expérience TP_Iris_Classification :'
        ),
      ],
    }),

    createTable(
      ['Nom du Run', 'Modèle', 'Hyperparamètres', 'Accuracy', 'Precision', 'Recall', 'F1-Score'],
      [
        [
          'RandomForest_n_100_depth_5',
          'Random Forest',
          'n_est=100, depth=5',
          '0.9333',
          '0.9333',
          '0.9333',
          '0.9333',
        ],
        [
          'RandomForest_n_200_depth_8',
          'Random Forest',
          'n_est=200, depth=8',
          '0.9000',
          '0.9024',
          '0.9000',
          '0.8997',
        ],
        [
          'KNN_k_5',
          'K-Neighbors',
          'k=5, weights=uniform',
          '1.0000',
          '1.0000',
          '1.0000',
          '1.0000',
        ],
      ],
      [2200, 1600, 1838, 1000, 1000, 1000, 1000]
    ),

    tip(
      'Sur cet échantillon stratifié du dataset Iris, le modèle KNN k=5 obtient un score parfait (1.0000). Le passage de 100 à 200 arbres sur Random Forest n’apporte aucun gain d’accuracy, confirmant qu’augmenter la complexité d’un modèle sur un petit dataset peut induire un léger sur-apprentissage.',
      'Analyse comparative'
    ),

    // CAPTURE 1 : MLFLOW RUNS
    ...imageBlock(
      screenRunsBuffer,
      'Figure 1 — Interface Web MLflow : tableau de bord de l’expérience TP_Iris_Classification présentant les runs et leurs métriques d’évaluation',
      500,
      321
    ),

    // CAPTURE 2 : MLFLOW DVC TRACKING
    ...imageBlock(
      screenDvcBuffer,
      'Figure 2 — Détail d’un run MLflow montrant la traçabilité de l’empreinte de données DVC (dvc_data_md5), le commit Git et les hyperparamètres',
      500,
      321
    ),

    // CAPTURE 3 : MLFLOW ARTIFACTS
    ...imageBlock(
      screenArtBuffer,
      'Figure 3 — Section Artefacts de MLflow affichant le dataset brut (data/iris.csv), le modèle entraîné (model/model.pkl) et la matrice de confusion',
      500,
      321
    ),

    // CAPTURE 4 : MATRICE DE CONFUSION
    ...imageBlock(
      cmBuffer,
      'Figure 4 — Matrice de confusion du modèle Random Forest générée et enregistrée dans les artefacts MLflow',
      320,
      256
    ),

    sectionHeader('2.4  TP 4 — Pipeline MLOps intégré (DVC + MLflow)'),
    new Paragraph({
      spacing: { after: 60 },
      children: [
        run(
          'L’enchaînement des étapes est orchestré par le fichier dvc.yaml qui surveille les dépendances et produit les métriques et artefacts :'
        ),
      ],
    }),
    code([
      '# Exécution déterministe du pipeline',
      'dvc repro',
      '',
      '# Visualisation du graphe orienté acyclique',
      'dvc dag',
      '',
      '# Affichage tabulaire des métriques calculées',
      'dvc metrics show',
    ]),
    tip(
      'Si ni les données sources, ni les scripts, ni params.yaml n’ont changé, DVC réutilise instantanément le cache local (Stage is cached - skipping run).',
      'Bénéfice de reproductibilité'
    ),

    // PARTIE 3
    partHeader('Partie 3 — Réponses aux Questions de Réflexion (Section 16)'),

    question(1, 'Quelle est la différence entre Git et DVC ?'),
    ...answer([
      'Git est un système de contrôle de versions conçu pour suivre des fichiers texte légers (scripts Python, fichiers de configuration). Il calcule des différences ligne par ligne. Il n’est pas adapté au stockage de fichiers volumineux ou binaires.',
      'DVC (Data Version Control) est conçu spécifiquement pour gérer les fichiers volumineux (jeux de données, modèles binaires). Il stocke les données dans un cache externe ou un stockage distant et ne confie à Git qu’un petit fichier pointeur textuel (.dvc) contenant une empreinte MD5.',
    ]),
    tip(
      'Git suit la logique du code ; DVC suit l’état et la provenance des données volumineuses sans alourdir le dépôt.',
      'Pour comprendre'
    ),

    question(2, 'Pourquoi éviter de stocker un dataset volumineux directement dans Git ?'),
    ...answer([
      'Stocker des fichiers volumineux dans Git entraîne une explosion irréversible de la taille du dossier .git, car chaque version modifiée conserve une copie complète de l’objet.',
      'Cela ralentit considérablement les opérations réseau (clone, pull, push), enfreint les limites imposées par les forges logicielles (ex. limite de 100 Mo sur GitHub), et ne permet pas d’effectuer de comparatifs textuels pertinents sur des formats binaires.',
    ]),

    question(3, 'Quelle est la différence entre dvc add et dvc repro ?'),
    ...answer([
      'dvc add est une commande manuelle de suivi statique d’un fichier ou dossier de données. Elle calcule l’empreinte de hachage, place les données dans le cache DVC et génère le fichier .dvc correspondant.',
      'dvc repro est une commande d’orchestration dynamique. Elle lit le fichier de pipeline dvc.yaml, examine les dépendances modifiées et réexécute automatiquement les étapes nécessaires en mettant à jour le fichier dvc.lock.',
    ]),

    question(4, 'Quel est le rôle du fichier dvc.lock ?'),
    ...answer([
      'Le fichier dvc.lock enregistre l’état résolu et figé du pipeline après exécution. Il consigne les empreintes MD5 exactes des dépendances (fichiers sources et données), les valeurs des paramètres utilisés, et les empreintes des sorties produites.',
      'Versionné dans Git, dvc.lock garantit la reproductibilité absolue : tout collaborateur peut réinstancier exactement les mêmes sorties avec dvc checkout sans recalcul inutile.',
    ]),

    question(5, 'Comment MLflow permet-il de comparer deux expérimentations ?'),
    ...answer([
      'MLflow centralise tous les runs d’une même problématique sous une expérience unique (TP_Iris_Classification). Dans son interface Web (mlflow ui), l’utilisateur peut sélectionner plusieurs runs et cliquer sur Compare.',
      'MLflow génère automatiquement un tableau comparatif côte à côte des hyperparamètres et des métriques, complété par des graphiques interactifs (diagrammes en barres, scatter plots, courbes de dispersion) pour identifier le modèle optimal.',
    ]),

    question(6, 'Comment retrouver les données et le modèle associés à un commit Git précis ?'),
    ...answer([
      'On se positionne tout d’abord sur le commit Git cible via la commande git checkout <commit_hash> (ou via un tag tel que v1.0). Cette commande restaure l’état exact des fichiers pointeurs (.dvc, dvc.yaml, dvc.lock).',
      'Il suffit ensuite d’exécuter dvc checkout (ou dvc pull) pour que DVC télécharge et replace dans l’arborescence locale les données d’entraînement et le modèle binaire model.pkl qui correspondent rigoureusement à ce commit.',
    ]),

    question(
      7,
      'Pourquoi la reproductibilité est-elle importante dans un projet d’intelligence artificielle ?'
    ),
    ...answer([
      'La reproductibilité est fondamentale pour l’auditabilité et la conformité réglementaire : elle permet d’expliquer et de justifier les prédictions d’un modèle déployé.',
      'Elle est indispensable pour le débogage (reproduire une anomalie constatée en production), la collaboration saine au sein des équipes Data, et l’intégration continue (CI/CD) permettant d’entraîner et tester les modèles de manière déterministe.',
    ]),

    question(
      8,
      'Quelle est la différence entre le versioning des modèles avec DVC et leur suivi avec MLflow ?'
    ),
    ...answer([
      'DVC versionne le modèle comme un artefact binaire au sein du pipeline de données : il garantit que le fichier physique model.pkl est lié de manière déterministe aux données sources et au code via dvc.lock.',
      'MLflow suit le modèle sous un angle scientifique et décisionnel : il conserve le contexte complet (métriques d’évaluation, paramètres, courbes de perte, signature des entrées/sorties) et fournit un Model Registry pour piloter les étapes de déploiement (Staging, Production).',
    ]),
    tip(
      'DVC garantit la reproductibilité de fabrication du modèle ; MLflow assure son évaluation scientifique et sa gouvernance opérationnelle.',
      'Complémentarité MLOps'
    ),

    // PARTIE 4
    partHeader('Partie 4 — Automatisation et Intégration Continue (GitHub Actions)'),
    new Paragraph({
      spacing: { after: 60 },
      children: [
        run(
          'Un pipeline d’intégration continue a été implémenté dans le fichier .github/workflows/mlops.yml afin de valider automatiquement la reproductibilité à chaque modification poussée sur GitHub :'
        ),
      ],
    }),
    code([
      'name: MLOps Pipeline',
      'on:',
      '  push:',
      '    branches: [main, develop]',
      '  pull_request:',
      '    branches: [main]',
      'jobs:',
      '  train:',
      '    runs-on: ubuntu-latest',
      '    steps:',
      '      - uses: actions/checkout@v4',
      '      - uses: actions/setup-python@v5',
      '        with: { python-version: "3.12" }',
      '      - run: pip install -r requirements.txt',
      '      - run: dvc repro',
      '      - run: dvc metrics show',
    ]),

    sectionHeader('Conclusion'),
    new Paragraph({
      spacing: { after: 120, line: 288 },
      children: [
        run(
          'Cet atelier démontre l’efficacité de la combinaison Git + DVC + MLflow pour répondre aux exigences industrielles du MLOps. Grâce au pipeline automatisé, chaque version de modèle est associée à un commit Git précis, une version de données immuable sous DVC et un suivi expérimental exhaustif dans MLflow, garantissant une reproductibilité et une gouvernance totales.'
        ),
      ],
    }),
  ];

  const doc = new Document({
    sections: [
      {
        properties: {
          page: {
            size: { width: 11906, height: 16838 }, // A4
            margin: { top: 1134, bottom: 1134, left: 1134, right: 1134 }, // 2 cm
          },
        },
        footers: {
          default: footer(moduleName),
        },
        children: children,
      },
    ],
  });

  const buffer = await Packer.toBuffer(doc);
  const outputPath = path.resolve('Compte_Rendu_Atelier_Git_DVC_MLflow_MLOps.docx');
  fs.writeFileSync(outputPath, buffer);
  console.log('Document DOCX généré avec succès avec les captures d’écran MLflow :', outputPath);
}

generate().catch(console.error);
