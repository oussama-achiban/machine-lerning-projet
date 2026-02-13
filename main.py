"""
Main execution script for Global School Electricity Access ML Project
Orchestrates all components: preprocessing, dimensionality reduction, clustering, 
classical models, neural networks, and MLflow tracking
"""

import sys
import os
import numpy as np
import pandas as pd
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.data_preprocessing import get_preprocessor
from src.dimensionality_reduction import get_reducer
from src.clustering import get_clusterer
from src.classical_models import get_classical_models
from src.neural_network_pytorch import get_neural_network_trainer
from src.evaluation import get_evaluator, get_mlflow_tracker, get_experiment_organizer


def create_sample_data():
    """Create sample dataset for demonstration"""
    print("\nCreating sample dataset...")
    
    # Generate synthetic data
    np.random.seed(42)
    n_samples = 500
    
    data = {
        'school_id': range(n_samples),
        'year': np.random.randint(1999, 2021, n_samples),
        'access_rate': np.random.rand(n_samples) * 100,
        'infrastructure_score': np.random.randint(0, 100, n_samples),
        'region': np.random.choice(['Africa', 'Asia', 'Americas', 'Europe'], n_samples),
        'development_level': np.random.choice(['Low', 'Medium', 'High'], n_samples),
        'investment': np.random.randint(1000, 100000, n_samples),
    }
    
    df = pd.DataFrame(data)
    
    # Create data directories
    Path('data/raw').mkdir(parents=True, exist_ok=True)
    
    # Save sample data
    df.to_csv('data/raw/electricity_access_data.csv', index=False)
    print(f"Sample data created: {df.shape[0]} samples, {df.shape[1]} features")
    
    return df


def main():
    """Main execution pipeline"""
    
    print("="*70)
    print("GLOBAL SCHOOL ELECTRICITY ACCESS - MACHINE LEARNING PROJECT")
    print("Student: Oussama Achiban | Master ISI")
    print("="*70)
    
    # Initialize MLflow
    tracker = get_mlflow_tracker("School_Electricity_Access_ML")
    
    # ============================================================================
    # 1. DATA PREPROCESSING
    # ============================================================================
    
    print("\n" + "="*70)
    print("PHASE 1: DATA PREPROCESSING")
    print("="*70)
    
    # Create sample data if needed
    if not os.path.exists('data/raw/electricity_access_data.csv'):
        create_sample_data()
    
    preprocessor = get_preprocessor()
    X_train, X_test, y_train, y_test, X_train_df = preprocessor.preprocess_pipeline(
        'data/raw/electricity_access_data.csv',
        test_size=0.2
    )
    
    # Save processed data
    Path('data/processed').mkdir(parents=True, exist_ok=True)
    np.save('data/processed/X_train.npy', X_train)
    np.save('data/processed/X_test.npy', X_test)
    np.save('data/processed/y_train.npy', y_train)
    np.save('data/processed/y_test.npy', y_test)
    
    # ============================================================================
    # 2. DIMENSIONALITY REDUCTION
    # ============================================================================
    
    print("\n" + "="*70)
    print("PHASE 2: DIMENSIONALITY REDUCTION")
    print("="*70)
    
    reducer = get_reducer()
    
    # PCA
    X_train_pca = reducer.apply_pca(X_train, n_components=2)
    X_test_pca = reducer.apply_pca(X_test, n_components=2)
    
    # Full PCA analysis
    pca_full, X_train_pca_full = reducer.apply_pca_full(X_train)
    
    # t-SNE
    X_train_tsne = reducer.apply_tsne(X_train, n_components=2, perplexity=30)
    
    # NMF
    X_train_nmf = reducer.apply_nmf(X_train, n_components=2)
    
    # ============================================================================
    # 3. CLUSTERING
    # ============================================================================
    
    print("\n" + "="*70)
    print("PHASE 3: CLUSTERING ANALYSIS")
    print("="*70)
    
    clusterer = get_clusterer()
    
    # Elbow method
    inertias, silhouette_scores = clusterer.elbow_method(X_train, k_range=range(2, 11))
    
    # Apply K-Means
    optimal_k = 3
    kmeans_labels = clusterer.apply_kmeans(X_train, n_clusters=optimal_k)
    
    # Agglomerative Clustering
    agg_labels = clusterer.apply_agglomerative(X_train, n_clusters=optimal_k, linkage='ward')
    
    # DBSCAN
    dbscan_labels = clusterer.apply_dbscan(X_train, eps=0.5, min_samples=5)
    
    # ============================================================================
    # 4. CLASSICAL MACHINE LEARNING MODELS
    # ============================================================================
    
    print("\n" + "="*70)
    print("PHASE 4: CLASSICAL MACHINE LEARNING MODELS")
    print("="*70)
    
    classical = get_classical_models()
    
    # Train all models
    classical.logistic_regression(X_train, y_train, X_test, y_test)
    classical.knn(X_train, y_train, X_test, y_test, n_neighbors=5)
    classical.decision_tree(X_train, y_train, X_test, y_test, max_depth=10)
    classical.svm(X_train, y_train, X_test, y_test, kernel='rbf')
    classical.random_forest(X_train, y_train, X_test, y_test, n_estimators=100)
    classical.adaboost(X_train, y_train, X_test, y_test, n_estimators=50)
    classical.gradient_boosting(X_train, y_train, X_test, y_test, n_estimators=100)
    
    # Print results
    results_df = classical.get_results_df()
    print("\n" + results_df.to_string())
    
    # ============================================================================
    # 5. NEURAL NETWORK (PyTorch)
    # ============================================================================
    
    print("\n" + "="*70)
    print("PHASE 5: NEURAL NETWORK (PyTorch)")
    print("="*70)
    
    input_size = X_train.shape[1]
    output_size = len(np.unique(y_train))
    hidden_sizes = [128, 64, 32]
    
    nn_trainer = get_neural_network_trainer(input_size, output_size, hidden_sizes)
    
    # Train neural network
    nn_trainer.fit(X_train, y_train, X_test, y_test, epochs=100, batch_size=32, verbose=20)
    
    # Evaluate
    nn_metrics = nn_trainer.evaluate(X_test, y_test)
    
    # Save model
    Path('models').mkdir(exist_ok=True)
    nn_trainer.save_model('models/neural_network.pth')
    
    # ============================================================================
    # 6. MODEL COMPARISON AND EVALUATION
    # ============================================================================
    
    print("\n" + "="*70)
    print("PHASE 6: MODEL COMPARISON AND EVALUATION")
    print("="*70)
    
    evaluator = get_evaluator()
    organizer = get_experiment_organizer()
    
    # Combine results
    all_results = {**classical.results, 'Neural Network': nn_metrics}
    
    # Compare and rank
    organizer.compare_models(all_results)
    organizer.rank_models(all_results, metric='f1')
    
    # ============================================================================
    # 7. MLflow TRACKING
    # ============================================================================
    
    print("\n" + "="*70)
    print("PHASE 7: MLflow EXPERIMENT TRACKING")
    print("="*70)
    
    for model_name, metrics in all_results.items():
        tracker.start_run(run_name=model_name)
        
        # Log parameters
        params = {
            'model': model_name,
            'test_size': 0.2,
            'random_state': 42
        }
        tracker.log_params(params)
        
        # Log metrics
        tracker.log_metrics(metrics)
        
        # Log model if available
        if model_name != 'Neural Network' and model_name in classical.models:
            tracker.log_model(classical.models[model_name], model_name, framework='sklearn')
        
        tracker.end_run()
    
    print("\n" + "="*70)
    print("PROJECT COMPLETE!")
    print("="*70)
    print("\nGenerated artifacts:")
    print("- data/raw/electricity_access_data.csv (sample data)")
    print("- data/processed/ (preprocessed data)")
    print("- models/neural_network.pth (trained model)")
    print("- MLflow experiments tracking all model results")
    
    return X_train, X_test, y_train, y_test, classical, nn_trainer


if __name__ == "__main__":
    X_train, X_test, y_train, y_test, classical, nn = main()
    print("\nAll data and models available in the namespace for further analysis.")
