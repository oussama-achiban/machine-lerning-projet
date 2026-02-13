# QUICK START

## 1) Setup Python

```bash
python -m venv .venv
# PowerShell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 2) Run pipeline ML

```bash
python main.py
```

## 3) Ouvrir MLflow

```bash
mlflow ui --backend-store-uri file:./mlruns
```

URL: `http://127.0.0.1:5000`

## 4) Ouvrir notebook

```bash
cd notebooks
jupyter notebook exploration.ipynb
```

## 5) Compiler le rapport

```bash
cd reports
pdflatex report.tex
```

## 6) Push des changements

```bash
git add .
git commit -m "update"
git push
```

Si rejet `fetch first`:

```bash
git pull --rebase origin main
git push
```
