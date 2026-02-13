# Global School Electricity Access - Machine Learning Project

Projet de Machine Learning sur l'acces a l'electricite dans les ecoles, avec pipeline Python complet, suivi MLflow, notebook d'exploration et rapport LaTeX.

Repository GitHub:
- https://github.com/oussama-achiban/machine-lerning-projet

## Contenu du projet

- Pipeline ML Python (`main.py` + `src/`)
- Notebook d'analyse (`notebooks/exploration.ipynb`)
- Figures et rapport (`notebooks/images/`, `reports/report.tex`)
- Tracking d'experiences (`mlruns/`, `mlflow.db`)

## Structure principale

```text
machine-learning-project/
  src/
    data_preprocessing.py
    dimensionality_reduction.py
    clustering.py
    classical_models.py
    neural_network_pytorch.py
    evaluation.py
  data/
    raw/
    processed/
  notebooks/
    exploration.ipynb
    images/
  reports/
    report.tex
  models/
  mlruns/
  main.py
  requirements.txt
```

## Prerequis

- Python 3.10+
- pip

## Installation

```bash
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Execution rapide

### 1) Lancer tout le pipeline ML

```bash
python main.py
```

### 2) Ouvrir le notebook

```bash
cd notebooks
jupyter notebook exploration.ipynb
```

### 3) Ouvrir MLflow UI

```bash
mlflow ui --backend-store-uri file:./mlruns
```

Puis ouvrir `http://127.0.0.1:5000`.

## Rapport

```bash
cd reports
pdflatex report.tex
```

Version etendue avec figures/tables:
- `notebooks/images/report_images.tex`

## Push des changements

```bash
git add .
git commit -m "docs: update markdown files"
git push
```

Si le push est rejete:

```bash
git pull --rebase origin main
git push
```

## Notes

- Les resultats dependent du jeu de donnees et de l'environnement.
- Si `data/raw/electricity_access_data.csv` est absent, `main.py` peut creer un dataset d'exemple.
