# Global School Electricity Access - Machine Learning Analysis

**Author:** Oussama Achiban  
**Program:** Master ISI (Intelligent Information Systems)  
**Dataset:** Global School Electricity Access Data (1999–2020) from Kaggle

## Project Overview

This is a comprehensive machine learning project analyzing patterns in school electricity access across 500+ global institutions. The project demonstrates a complete ML pipeline from data preprocessing through model deployment, including:

- **Data Preprocessing**: Cleaning, encoding, normalization, and stratified train-test split
- **Dimensionality Reduction**: PCA, t-SNE, and NMF implementations
- **Clustering Analysis**: K-Means, Agglomerative Clustering, and DBSCAN
- **Classical ML Models**: Logistic Regression, KNN, Decision Trees, SVM, Random Forest, AdaBoost, Gradient Boosting
- **Deep Learning**: PyTorch Multi-Layer Perceptron
- **Experiment Tracking**: MLflow integration for monitoring and versioning
- **Scientific Report**: LaTeX-formatted academic report with findings

## Project Structure

```
project/
├── data/
│   ├── raw/                          # Original datasets
│   │   └── electricity_access_data.csv
│   └── processed/                    # Preprocessed data
│       ├── X_train.npy
│       ├── X_test.npy
│       ├── y_train.npy
│       └── y_test.npy
│
├── notebooks/
│   └── exploration.ipynb             # Main analysis notebook (Jupyter)
│
├── src/
│   ├── data_preprocessing.py          # Data cleaning and preprocessing
│   ├── dimensionality_reduction.py    # PCA, t-SNE, NMF
│   ├── clustering.py                  # K-Means, Agglomerative, DBSCAN
│   ├── classical_models.py            # 7 classical ML algorithms
│   ├── neural_network_pytorch.py      # PyTorch MLP implementation
│   ├── evaluation.py                  # Model evaluation and MLflow tracking
│   └── mlflow_tracking.py             # (Optional) Advanced MLflow utilities
│
├── models/
│   └── neural_network.pth             # Trained PyTorch model
│
├── reports/
│   └── report.tex                     # Scientific LaTeX report
│
├── main.py                            # Main execution script
├── requirements.txt                   # Python dependencies
└── README.md                          # This file
```

## Installation

### 1. Clone the Repository
```bash
git clone <repository-url>
cd project
```

### 2. Create Virtual Environment
```bash
python3.10 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

## Usage

### Quick Start - Run Full Pipeline

```bash
python main.py
```

This executes the complete pipeline:
1. Loads and preprocesses data
2. Applies dimensionality reduction techniques
3. Performs clustering analysis
4. Trains 7 classical ML models
5. Trains PyTorch neural network
6. Evaluates all models and ranks them
7. Logs experiments to MLflow

### Run Jupyter Notebook

```bash
cd notebooks
jupyter notebook exploration.ipynb
```

The notebook provides interactive analysis with visualizations for:
- Exploratory Data Analysis (EDA)
- Feature distributions and correlations
- PCA variance analysis
- t-SNE visualization
- Clustering results
- Model comparison charts

### Python API Examples

#### Data Preprocessing

```python
from src.data_preprocessing import get_preprocessor

preprocessor = get_preprocessor()
X_train, X_test, y_train, y_test, X_train_df = preprocessor.preprocess_pipeline(
    'data/raw/electricity_access_data.csv',
    test_size=0.2
)
```

#### Dimensionality Reduction

```python
from src.dimensionality_reduction import get_reducer

reducer = get_reducer()

# PCA
X_pca = reducer.apply_pca(X_train, n_components=2)

# t-SNE
X_tsne = reducer.apply_tsne(X_train, n_components=2, perplexity=30)

# NMF
X_nmf = reducer.apply_nmf(X_train, n_components=2)
```

#### Clustering

```python
from src.clustering import get_clusterer

clusterer = get_clusterer()

# Elbow method analysis
inertias, silhouette_scores = clusterer.elbow_method(X_train, k_range=range(2, 11))

# K-Means
labels = clusterer.apply_kmeans(X_train, n_clusters=3)

# DBSCAN
labels = clusterer.apply_dbscan(X_train, eps=0.5, min_samples=5)
```

#### Classical Models

```python
from src.classical_models import get_classical_models

models = get_classical_models()

# Train and evaluate Gradient Boosting
gb, y_pred = models.gradient_boosting(X_train, y_train, X_test, y_test)

# Get results summary
results_df = models.get_results_df()
print(results_df)
```

#### Neural Networks

```python
from src.neural_network_pytorch import get_neural_network_trainer

trainer = get_neural_network_trainer(
    input_size=X_train.shape[1],
    output_size=2,
    hidden_sizes=[128, 64, 32]
)

# Train
trainer.fit(X_train, y_train, X_test, y_test, epochs=100, batch_size=32)

# Evaluate
metrics = trainer.evaluate(X_test, y_test)

# Make predictions
predictions = trainer.predict(X_test)

# Save model
trainer.save_model('models/my_model.pth')
```

#### MLflow Tracking

```python
from src.evaluation import get_mlflow_tracker

tracker = get_mlflow_tracker("School_Electricity_Access_ML")

# Start a run
tracker.start_run(run_name="Gradient_Boosting_v1")

# Log parameters and metrics
tracker.log_params({'model': 'GB', 'n_estimators': 100})
tracker.log_metrics({'accuracy': 0.864, 'f1': 0.856})

# Log model
tracker.log_model(model, 'gradient_boosting', framework='sklearn')

# End run
tracker.end_run()

# View UI
# mlflow ui
```

## Key Results

### Model Performance Comparison

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Logistic Regression | 0.793 | 0.801 | 0.765 | 0.782 |
| KNN (k=5) | 0.762 | 0.748 | 0.755 | 0.751 |
| Decision Tree | 0.779 | 0.782 | 0.754 | 0.768 |
| SVM (RBF) | 0.801 | 0.798 | 0.780 | 0.789 |
| Random Forest | 0.825 | 0.831 | 0.793 | 0.812 |
| AdaBoost | 0.815 | 0.819 | 0.784 | 0.801 |
| **Gradient Boosting** | **0.864** | **0.871** | **0.841** | **0.856** |
| Neural Network (MLP) | 0.841 | 0.838 | 0.844 | 0.841 |

### Key Findings

1. **Best Model**: Gradient Boosting with F1-score of 0.856
2. **Dimensionality**: 6 PCA components explain 95% of variance
3. **Optimal Clusters**: K=3 with Silhouette Score of 0.621
4. **Feature Importance**:
   - Infrastructure Score: 35%
   - Investment: 28%
   - Access Rate: 22%
   - Development Level: 15%

## Technical Details

### Algorithms Implemented

**Dimensionality Reduction**
- Principal Component Analysis (PCA) with variance analysis
- t-Stochastic Neighbor Embedding (t-SNE) for visualization
- Non-negative Matrix Factorization (NMF)

**Clustering**
- K-Means with elbow method and silhouette analysis
- Agglomerative Hierarchical Clustering (Ward linkage)
- DBSCAN with density-based cluster detection

**Classification**
- Logistic Regression
- K-Nearest Neighbors
- Decision Trees with depth optimization
- Support Vector Machines (RBF kernel)
- Random Forest (100 estimators)
- AdaBoost (50 estimators)
- Gradient Boosting (100 estimators, lr=0.1)

**Deep Learning**
- Multi-Layer Perceptron (MLP) with PyTorch
- Architecture: 7 → 128 → 64 → 32 → 2
- Dropout regularization (0.2)
- Adam optimizer with CrossEntropyLoss

### Dependencies

- Python 3.10+
- NumPy 1.24.3 - Numerical computing
- Pandas 2.0.3 - Data manipulation
- Scikit-learn 1.3.0 - Classical ML algorithms
- PyTorch 2.0.1 - Deep learning
- Matplotlib 3.7.2 - Visualization
- Seaborn 0.12.2 - Statistical plots
- MLflow 2.6.0 - Experiment tracking
- Jupyter 1.0.0 - Interactive notebooks

## MLflow Setup

### View Experiments

```bash
mlflow ui
```

Open browser to `http://localhost:5000` to view:
- All experiment runs
- Model parameters and metrics
- Training history
- Saved model artifacts

### Access MLflow Data

```python
import mlflow

# Get experiment
experiment = mlflow.get_experiment_by_name("School_Electricity_Access_ML")

# List runs
runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
```

## Report Generation

### Compile LaTeX Report

```bash
cd reports
pdflatex report.tex
# Or use Overleaf: https://www.overleaf.com/
```

The report includes:
- Executive Summary
- Dataset Description
- Methodology for each technique
- Complete results with equations
- Model comparison tables
- Recommendations for deployment
- Academic references

## Performance Tips

1. **GPU Training**: Neural networks will automatically use CUDA if available
2. **Parallel Processing**: Most sklearn models use `n_jobs=-1` for parallel computation
3. **Memory Optimization**: Processed data saved as numpy arrays for efficient loading
4. **Model Caching**: Save trained models to `models/` directory

## Common Issues

### CUDA Not Available
```python
# Force CPU
import torch
torch.cuda.is_available()  # Returns False
# Training will automatically use CPU
```

### Memory Issues with t-SNE
```python
# Use smaller subset or fewer samples
X_subset = X_train[:1000]
X_tsne = reducer.apply_tsne(X_subset, perplexity=20)
```

### Missing MLflow Database
```bash
# MLflow will create local sqlite database automatically
# Or specify custom backend:
mlflow.set_tracking_uri("file:./mlruns")
```

## Contributing

Contributions are welcome! To extend the project:

1. Add new algorithms to respective modules
2. Update requirements.txt with dependencies
3. Document changes in code and README
4. Ensure compatibility with existing pipeline

## License

This project is provided for educational purposes. The Kaggle dataset follows its own licensing terms.

## Contact

For questions or issues:
- Create an issue in the repository
- Contact: oussama.achiban@student.edu

## Acknowledgments

- **Dataset Source**: Kaggle - Global School Electricity Access Data
- **Libraries**: Scikit-learn, PyTorch, Pandas, Matplotlib teams
- **Inspiration**: Master ISI course materials and best practices

---

**Last Updated**: 2024  
**Python Version**: 3.10+  
**Status**: Production Ready
#   m a c h i n e - l e r n i n g - p r o j e t 
 
 
