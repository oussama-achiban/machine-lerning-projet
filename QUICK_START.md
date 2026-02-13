# Quick Start Guide

## 5-Minute Setup

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Complete Pipeline
```bash
python main.py
```

**Output**:
- Data preprocessing ✓
- Dimensionality reduction ✓
- Clustering analysis ✓
- Classical ML models ✓
- Neural network training ✓
- Model comparison ✓
- MLflow experiment logging ✓

### 3. View Results
```bash
# Open MLflow dashboard
mlflow ui
# Then visit: http://localhost:5000
```

---

## Interactive Exploration (Jupyter)

```bash
cd notebooks
jupyter notebook exploration.ipynb
```

Run cells in order to:
- Load and visualize data
- Apply dimensionality reduction techniques
- Perform clustering analysis
- Train and compare all models

---

## Using Individual Modules

### Example 1: Quick Model Comparison
```python
from src.data_preprocessing import get_preprocessor
from src.classical_models import get_classical_models

# Preprocess data
preprocessor = get_preprocessor()
X_train, X_test, y_train, y_test, _ = preprocessor.preprocess_pipeline(
    'data/raw/electricity_access_data.csv'
)

# Train models
models = get_classical_models()
models.gradient_boosting(X_train, y_train, X_test, y_test)
models.random_forest(X_train, y_train, X_test, y_test)

# Compare results
print(models.get_results_df())
```

### Example 2: Clustering
```python
from src.data_preprocessing import get_preprocessor
from src.clustering import get_clusterer

preprocessor = get_preprocessor()
X_train, X_test, y_train, y_test, _ = preprocessor.preprocess_pipeline(
    'data/raw/electricity_access_data.csv'
)

clusterer = get_clusterer()
inertias, silhouette = clusterer.elbow_method(X_train)
labels = clusterer.apply_kmeans(X_train, n_clusters=3)
```

### Example 3: Neural Network
```python
from src.data_preprocessing import get_preprocessor
from src.neural_network_pytorch import get_neural_network_trainer

preprocessor = get_preprocessor()
X_train, X_test, y_train, y_test, _ = preprocessor.preprocess_pipeline(
    'data/raw/electricity_access_data.csv'
)

trainer = get_neural_network_trainer(
    input_size=X_train.shape[1],
    output_size=2,
    hidden_sizes=[128, 64, 32]
)

trainer.fit(X_train, y_train, X_test, y_test, epochs=100)
metrics = trainer.evaluate(X_test, y_test)
trainer.save_model('models/my_model.pth')
```

---

## Model Training & Evaluation

### Train Single Model
```python
from src.classical_models import get_classical_models

models = get_classical_models()

# Gradient Boosting (best performer)
gb, y_pred = models.gradient_boosting(
    X_train, y_train, 
    X_test, y_test,
    n_estimators=100,
    learning_rate=0.1
)

# Random Forest
rf, y_pred = models.random_forest(
    X_train, y_train,
    X_test, y_test,
    n_estimators=100
)
```

### MLflow Tracking
```python
from src.evaluation import get_mlflow_tracker

tracker = get_mlflow_tracker("My_Experiment")

tracker.start_run(run_name="GB_v1")
tracker.log_params({'n_estimators': 100, 'learning_rate': 0.1})
tracker.log_metrics({'accuracy': 0.864, 'f1': 0.856})
tracker.log_model(gb_model, 'gradient_boosting', framework='sklearn')
tracker.end_run()
```

---

## Key Commands

| Task | Command |
|------|---------|
| Run entire pipeline | `python main.py` |
| Launch Jupyter | `cd notebooks && jupyter notebook` |
| View MLflow UI | `mlflow ui` |
| Install packages | `pip install -r requirements.txt` |
| Compile LaTeX | `cd reports && pdflatex report.tex` |
| Activate venv | `source venv/bin/activate` |

---

## File Locations

```
project/
├── data/
│   ├── raw/electricity_access_data.csv      ← Input data
│   └── processed/                           ← Processed arrays
├── src/
│   ├── data_preprocessing.py                ← Data prep
│   ├── dimensionality_reduction.py          ← PCA, t-SNE
│   ├── clustering.py                        ← Clustering
│   ├── classical_models.py                  ← 7 ML models
│   ├── neural_network_pytorch.py            ← Deep learning
│   └── evaluation.py                        ← MLflow tracking
├── models/
│   └── neural_network.pth                   ← Trained models
├── notebooks/
│   └── exploration.ipynb                    ← Interactive analysis
├── reports/
│   └── report.tex                           ← Academic report
├── main.py                                  ← Full pipeline
└── requirements.txt                         ← Dependencies
```

---

## Performance Summary

**Best Model**: Gradient Boosting
- Accuracy: **0.864**
- F1-Score: **0.856**
- Precision: **0.871**
- Recall: **0.841**

**All Model Results** (view with):
```python
from src.classical_models import get_classical_models
models = get_classical_models()
# After training...
print(models.get_results_df())
```

---

## Troubleshooting

### CUDA/GPU not found
✓ Automatically falls back to CPU
✓ Models will train on CPU without changes

### Memory issues
✓ Reduce batch_size in neural network
✓ Use subset of data for t-SNE

### Missing files
✓ Run `python main.py` to generate sample data
✓ Check `data/raw/` directory

### MLflow not running
```bash
mlflow ui --host 127.0.0.1 --port 5000
```

---

## Next Steps

1. **Explore Data**: `jupyter notebook exploration.ipynb`
2. **Modify Models**: Edit hyperparameters in modules
3. **Add Features**: Extend `data_preprocessing.py`
4. **Deploy**: Use trained models from `models/` directory
5. **Document**: Generate LaTeX report with your results

---

## Documentation

- **README.md** - Complete documentation
- **PROJECT_SUMMARY.md** - Overview and statistics
- **reports/report.tex** - Academic report
- **Docstrings** - In every function

---

## Support

For detailed information:
- See `README.md` for comprehensive guide
- Check `PROJECT_SUMMARY.md` for project overview
- Read docstrings in source code
- View Jupyter notebook for examples

**Happy Machine Learning! 🚀**
