"""
Evaluation module with MLflow integration
Comprehensive model evaluation and experiment tracking
"""

import numpy as np
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                            confusion_matrix, classification_report, roc_auc_score, roc_curve)
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Tuple
import mlflow
import mlflow.sklearn
try:
    import mlflow.pytorch
    _MLFLOW_PYTORCH_AVAILABLE = True
except ModuleNotFoundError:
    _MLFLOW_PYTORCH_AVAILABLE = False


class ModelEvaluator:
    """Comprehensive model evaluation"""
    
    @staticmethod
    def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate comprehensive metrics"""
        
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0)
        }
        
        # ROC-AUC (for binary classification)
        if len(np.unique(y_true)) == 2:
            try:
                metrics['roc_auc'] = roc_auc_score(y_true, y_pred)
            except:
                metrics['roc_auc'] = 0.0
        
        return metrics
    
    @staticmethod
    def print_metrics(model_name: str, metrics: Dict[str, float]):
        """Print metrics nicely"""
        print(f"\n{model_name} Metrics:")
        print("-" * 40)
        for metric, value in metrics.items():
            print(f"{metric:15s}: {value:.4f}")
        print("-" * 40)
    
    @staticmethod
    def plot_roc_curve(y_true: np.ndarray, y_pred_proba: np.ndarray,
                      model_name: str = "Model", figsize: Tuple = (8, 6)):
        """Plot ROC curve"""
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        auc = roc_auc_score(y_true, y_pred_proba)
        
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(fpr, tpr, label=f'ROC curve (AUC = {auc:.3f})')
        ax.plot([0, 1], [0, 1], 'k--', label='Random classifier')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(f'ROC Curve - {model_name}')
        ax.legend()
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray,
                             model_name: str = "Model", figsize: Tuple = (8, 6)):
        """Plot confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)
        
        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.set_title(f'Confusion Matrix - {model_name}')
        
        plt.tight_layout()
        return fig


class MLflowTracker:
    """MLflow integration for experiment tracking"""
    
    def __init__(self, experiment_name: str):
        """Initialize MLflow tracker"""
        mlflow.set_experiment(experiment_name)
        self.experiment_name = experiment_name
        print(f"MLflow experiment set to: {experiment_name}")
    
    def start_run(self, run_name: str = None):
        """Start a new MLflow run"""
        mlflow.start_run(run_name=run_name)
        print(f"MLflow run started: {run_name}")
    
    def end_run(self):
        """End the current MLflow run"""
        mlflow.end_run()
        print("MLflow run ended")
    
    def log_params(self, params: Dict):
        """Log parameters"""
        for key, value in params.items():
            mlflow.log_param(key, value)
        print(f"Logged {len(params)} parameters")
    
    def log_metrics(self, metrics: Dict, step: int = None):
        """Log metrics"""
        for key, value in metrics.items():
            mlflow.log_metric(key, value, step=step)
        print(f"Logged {len(metrics)} metrics")
    
    def log_model(self, model, model_name: str, framework: str = 'sklearn'):
        """Log model to MLflow"""
        if framework == 'sklearn':
            mlflow.sklearn.log_model(model, model_name)
        elif framework == 'pytorch':
            if not _MLFLOW_PYTORCH_AVAILABLE:
                raise ModuleNotFoundError(
                    "mlflow.pytorch (and torch) is not available. "
                    "Install PyTorch to log PyTorch models."
                )
            mlflow.pytorch.log_model(model, model_name)
        print(f"Model logged: {model_name}")
    
    def log_artifact(self, local_path: str, artifact_path: str = None):
        """Log artifact"""
        mlflow.log_artifact(local_path, artifact_path)
        print(f"Artifact logged: {local_path}")
    
    def log_text(self, text: str, filename: str = "report.txt"):
        """Log text content"""
        with open(filename, 'w') as f:
            f.write(text)
        mlflow.log_artifact(filename)
        print(f"Text logged: {filename}")
    
    def log_figure(self, fig, name: str = "figure.png"):
        """Log matplotlib figure"""
        fig.savefig(name, dpi=100, bbox_inches='tight')
        mlflow.log_artifact(name)
        print(f"Figure logged: {name}")
    
    def get_run_info(self):
        """Get current run information"""
        run = mlflow.active_run()
        if run:
            return {
                'run_id': run.info.run_id,
                'experiment_id': run.info.experiment_id,
                'start_time': run.info.start_time,
            }
        return None


class ExperimentOrganizer:
    """Organize and compare experiments"""
    
    @staticmethod
    def compare_models(models_results: Dict[str, Dict[str, float]]):
        """Compare multiple models"""
        print("\n" + "="*60)
        print("MODEL COMPARISON")
        print("="*60)
        
        for model_name, metrics in models_results.items():
            ModelEvaluator.print_metrics(model_name, metrics)
    
    @staticmethod
    def rank_models(models_results: Dict[str, Dict[str, float]], metric: str = 'f1'):
        """Rank models by a specific metric"""
        ranked = sorted(models_results.items(), 
                       key=lambda x: x[1].get(metric, 0), 
                       reverse=True)
        
        print(f"\nModels ranked by {metric}:")
        print("-" * 40)
        for rank, (model_name, metrics) in enumerate(ranked, 1):
            print(f"{rank}. {model_name}: {metrics[metric]:.4f}")
    
    @staticmethod
    def plot_model_comparison(models_results: Dict[str, Dict[str, float]],
                             metrics: list = None, figsize: Tuple = (12, 6)):
        """Plot comparison of multiple models"""
        if metrics is None:
            metrics = ['accuracy', 'precision', 'recall', 'f1']
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        for idx, metric in enumerate(metrics):
            ax = axes[idx // 2, idx % 2]
            
            model_names = list(models_results.keys())
            values = [models_results[name].get(metric, 0) for name in model_names]
            
            bars = ax.bar(model_names, values, color='skyblue', edgecolor='navy')
            ax.set_ylabel(metric.capitalize())
            ax.set_title(f'{metric.capitalize()} Comparison')
            ax.set_ylim([0, 1])
            ax.tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}', ha='center', va='bottom')
            
            ax.grid(alpha=0.3, axis='y')
        
        plt.tight_layout()
        return fig


def get_evaluator() -> ModelEvaluator:
    """Factory function to create evaluator"""
    return ModelEvaluator()


def get_mlflow_tracker(experiment_name: str = "ML_Project") -> MLflowTracker:
    """Factory function to create MLflow tracker"""
    return MLflowTracker(experiment_name)


def get_experiment_organizer() -> ExperimentOrganizer:
    """Factory function to create experiment organizer"""
    return ExperimentOrganizer()
