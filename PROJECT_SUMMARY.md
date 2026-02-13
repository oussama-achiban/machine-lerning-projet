# PROJECT SUMMARY

## Objectif

Construire un pipeline de Machine Learning pour analyser et predire l'acces a l'electricite dans les ecoles, avec:
- preprocessing,
- reduction de dimension,
- clustering,
- classification,
- suivi des experiences avec MLflow,
- reporting scientifique (LaTeX).

## Composants

- Pipeline principal: `main.py`
- Modules ML: `src/`
- Donnees: `data/raw`, `data/processed`
- Notebook: `notebooks/exploration.ipynb`
- Rapport: `reports/report.tex` + `notebooks/images/report_images.tex`
- Tracking: `mlruns/`, `mlflow.db`
- UI optionnelle: Next.js (`app/`, `components/`, `package.json`)

## Workflow technique

1. Chargement des donnees CSV
2. Nettoyage + encodage + normalisation
3. Split train/test
4. PCA, t-SNE, NMF
5. K-Means, Agglomerative, DBSCAN
6. Modeles classiques + reseau de neurones (si PyTorch dispo)
7. Evaluation et journalisation MLflow

## Etat actuel

- Le projet est operationnel pour execution locale.
- La documentation Markdown a ete harmonisee avec la structure reelle du repo.
- Le rapport image (`notebooks/images/report_images.tex`) inclut figures, tables, MLflow et lien GitHub.

## Lancement rapide

```bash
pip install -r requirements.txt
python main.py
mlflow ui --backend-store-uri file:./mlruns
```

Repository:
- https://github.com/oussama-achiban/machine-lerning-projet
