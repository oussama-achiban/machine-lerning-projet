"""
Classical Machine Learning Models
Implements: Logistic Regression, KNN, Decision Tree, SVM, Random Forest, AdaBoost, Gradient Boosting
"""

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import numpy as np
from typing import Dict, Tuple
import matplotlib.pyplot as plt
import seaborn as sns


class ClassicalModels:
    """Trains and evaluates multiple classical ML models"""
    
    def __init__(self):
        self.models = {}
        self.results = {}
    
    def logistic_regression(self, X_train: np.ndarray, y_train: np.ndarray,
                           X_test: np.ndarray = None, y_test: np.ndarray = None,
                           **kwargs) -> Tuple:
        """Train Logistic Regression"""
        print("\n" + "="*50)
        print("LOGISTIC REGRESSION")
        print("="*50)
        
        lr = LogisticRegression(max_iter=1000, random_state=42, **kwargs)
        lr.fit(X_train, y_train)
        
        self.models['Logistic Regression'] = lr
        
        y_pred_train = lr.predict(X_train)
        train_acc = accuracy_score(y_train, y_pred_train)
        print(f"Training Accuracy: {train_acc:.4f}")
        
        if X_test is not None and y_test is not None:
            y_pred_test = lr.predict(X_test)
            self._evaluate_model('Logistic Regression', y_test, y_pred_test)
            return lr, y_pred_test
        
        return lr, y_pred_train
    
    def knn(self, X_train: np.ndarray, y_train: np.ndarray,
           X_test: np.ndarray = None, y_test: np.ndarray = None,
           n_neighbors: int = 5, **kwargs) -> Tuple:
        """Train K-Nearest Neighbors"""
        print("\n" + "="*50)
        print(f"K-NEAREST NEIGHBORS (k={n_neighbors})")
        print("="*50)
        
        knn = KNeighborsClassifier(n_neighbors=n_neighbors, n_jobs=-1, **kwargs)
        knn.fit(X_train, y_train)
        
        self.models['KNN'] = knn
        
        y_pred_train = knn.predict(X_train)
        train_acc = accuracy_score(y_train, y_pred_train)
        print(f"Training Accuracy: {train_acc:.4f}")
        
        if X_test is not None and y_test is not None:
            y_pred_test = knn.predict(X_test)
            self._evaluate_model('KNN', y_test, y_pred_test)
            return knn, y_pred_test
        
        return knn, y_pred_train
    
    def decision_tree(self, X_train: np.ndarray, y_train: np.ndarray,
                     X_test: np.ndarray = None, y_test: np.ndarray = None,
                     max_depth: int = 10, **kwargs) -> Tuple:
        """Train Decision Tree"""
        print("\n" + "="*50)
        print(f"DECISION TREE (max_depth={max_depth})")
        print("="*50)
        
        dt = DecisionTreeClassifier(max_depth=max_depth, random_state=42, **kwargs)
        dt.fit(X_train, y_train)
        
        self.models['Decision Tree'] = dt
        
        y_pred_train = dt.predict(X_train)
        train_acc = accuracy_score(y_train, y_pred_train)
        print(f"Training Accuracy: {train_acc:.4f}")
        
        if X_test is not None and y_test is not None:
            y_pred_test = dt.predict(X_test)
            self._evaluate_model('Decision Tree', y_test, y_pred_test)
            return dt, y_pred_test
        
        return dt, y_pred_train
    
    def svm(self, X_train: np.ndarray, y_train: np.ndarray,
           X_test: np.ndarray = None, y_test: np.ndarray = None,
           kernel: str = 'rbf', **kwargs) -> Tuple:
        """Train Support Vector Machine"""
        print("\n" + "="*50)
        print(f"SUPPORT VECTOR MACHINE (kernel={kernel})")
        print("="*50)
        
        svm = SVC(kernel=kernel, random_state=42, **kwargs)
        svm.fit(X_train, y_train)
        
        self.models['SVM'] = svm
        
        y_pred_train = svm.predict(X_train)
        train_acc = accuracy_score(y_train, y_pred_train)
        print(f"Training Accuracy: {train_acc:.4f}")
        
        if X_test is not None and y_test is not None:
            y_pred_test = svm.predict(X_test)
            self._evaluate_model('SVM', y_test, y_pred_test)
            return svm, y_pred_test
        
        return svm, y_pred_train
    
    def random_forest(self, X_train: np.ndarray, y_train: np.ndarray,
                     X_test: np.ndarray = None, y_test: np.ndarray = None,
                     n_estimators: int = 100, max_depth: int = 10, **kwargs) -> Tuple:
        """Train Random Forest"""
        print("\n" + "="*50)
        print(f"RANDOM FOREST (n_estimators={n_estimators}, max_depth={max_depth})")
        print("="*50)
        
        rf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth,
                                   random_state=42, n_jobs=-1, **kwargs)
        rf.fit(X_train, y_train)
        
        self.models['Random Forest'] = rf
        
        y_pred_train = rf.predict(X_train)
        train_acc = accuracy_score(y_train, y_pred_train)
        print(f"Training Accuracy: {train_acc:.4f}")
        
        if X_test is not None and y_test is not None:
            y_pred_test = rf.predict(X_test)
            self._evaluate_model('Random Forest', y_test, y_pred_test)
            return rf, y_pred_test
        
        return rf, y_pred_train
    
    def adaboost(self, X_train: np.ndarray, y_train: np.ndarray,
                X_test: np.ndarray = None, y_test: np.ndarray = None,
                n_estimators: int = 50, **kwargs) -> Tuple:
        """Train AdaBoost"""
        print("\n" + "="*50)
        print(f"ADABOOST (n_estimators={n_estimators})")
        print("="*50)
        
        ab = AdaBoostClassifier(n_estimators=n_estimators, random_state=42, **kwargs)
        ab.fit(X_train, y_train)
        
        self.models['AdaBoost'] = ab
        
        y_pred_train = ab.predict(X_train)
        train_acc = accuracy_score(y_train, y_pred_train)
        print(f"Training Accuracy: {train_acc:.4f}")
        
        if X_test is not None and y_test is not None:
            y_pred_test = ab.predict(X_test)
            self._evaluate_model('AdaBoost', y_test, y_pred_test)
            return ab, y_pred_test
        
        return ab, y_pred_train
    
    def gradient_boosting(self, X_train: np.ndarray, y_train: np.ndarray,
                         X_test: np.ndarray = None, y_test: np.ndarray = None,
                         n_estimators: int = 100, learning_rate: float = 0.1, **kwargs) -> Tuple:
        """Train Gradient Boosting"""
        print("\n" + "="*50)
        print(f"GRADIENT BOOSTING (n_estimators={n_estimators}, lr={learning_rate})")
        print("="*50)
        
        gb = GradientBoostingClassifier(n_estimators=n_estimators, learning_rate=learning_rate,
                                       random_state=42, **kwargs)
        gb.fit(X_train, y_train)
        
        self.models['Gradient Boosting'] = gb
        
        y_pred_train = gb.predict(X_train)
        train_acc = accuracy_score(y_train, y_pred_train)
        print(f"Training Accuracy: {train_acc:.4f}")
        
        if X_test is not None and y_test is not None:
            y_pred_test = gb.predict(X_test)
            self._evaluate_model('Gradient Boosting', y_test, y_pred_test)
            return gb, y_pred_test
        
        return gb, y_pred_train
    
    def _evaluate_model(self, model_name: str, y_true: np.ndarray, y_pred: np.ndarray):
        """Evaluate model performance"""
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        self.results[model_name] = {
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'f1': f1
        }
        
        print(f"Test Accuracy:  {acc:.4f}")
        print(f"Precision:      {prec:.4f}")
        print(f"Recall:         {rec:.4f}")
        print(f"F1-Score:       {f1:.4f}")
    
    def get_results_df(self):
        """Get results as DataFrame"""
        import pandas as pd
        return pd.DataFrame(self.results).T
    
    def plot_comparison(self, figsize: Tuple = (12, 6)):
        """Plot model comparison"""
        import pandas as pd
        
        df = self.get_results_df()
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        metrics = ['accuracy', 'precision', 'recall', 'f1']
        for idx, metric in enumerate(metrics):
            ax = axes[idx // 2, idx % 2]
            df[metric].sort_values().plot(kind='barh', ax=ax, color='skyblue')
            ax.set_xlabel(metric.capitalize())
            ax.set_title(f'{metric.capitalize()} Comparison')
            ax.set_xlim([0, 1])
            ax.grid(alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray,
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


def get_classical_models() -> ClassicalModels:
    """Factory function to create classical models trainer"""
    return ClassicalModels()
