"""
Dimensionality reduction module
Implements: PCA, t-SNE, NMF
"""

import numpy as np
from sklearn.decomposition import PCA, NMF
from sklearn.manifold import TSNE
from typing import Tuple
import matplotlib.pyplot as plt
import seaborn as sns


class DimensionalityReducer:
    """Handles dimensionality reduction techniques"""
    
    def __init__(self):
        self.pca = None
        self.tsne = None
        self.nmf = None
    
    def apply_pca(self, X: np.ndarray, n_components: int = 2) -> np.ndarray:
        """Apply PCA for dimensionality reduction"""
        print(f"\nApplying PCA with {n_components} components...")
        
        self.pca = PCA(n_components=n_components, random_state=42)
        X_pca = self.pca.fit_transform(X)
        
        # Print variance explained
        total_var = np.sum(self.pca.explained_variance_ratio_)
        print(f"Explained variance ratio: {self.pca.explained_variance_ratio_}")
        print(f"Total variance explained: {total_var:.4f}")
        
        return X_pca
    
    def apply_pca_full(self, X: np.ndarray) -> Tuple[PCA, np.ndarray]:
        """Apply PCA and return full fitted object"""
        pca = PCA(random_state=42)
        X_pca = pca.fit_transform(X)
        
        # Find number of components for 95% variance
        cumsum = np.cumsum(pca.explained_variance_ratio_)
        n_components = np.argmax(cumsum >= 0.95) + 1
        
        print(f"\nFull PCA Analysis:")
        print(f"Components needed for 95% variance: {n_components}")
        print(f"Original shape: {X.shape}")
        print(f"Reduced shape: {X_pca.shape}")
        
        return pca, X_pca
    
    def apply_tsne(self, X: np.ndarray, n_components: int = 2, 
                   perplexity: int = 30, n_iter: int = 1000) -> np.ndarray:
        """Apply t-SNE for visualization"""
        print(f"\nApplying t-SNE with {n_components} components...")
        print(f"Perplexity: {perplexity}, Iterations: {n_iter}")
        
        self.tsne = TSNE(
            n_components=n_components,
            perplexity=perplexity,
            n_iter=n_iter,
            random_state=42,
            n_jobs=-1
        )
        X_tsne = self.tsne.fit_transform(X)
        
        print(f"t-SNE output shape: {X_tsne.shape}")
        return X_tsne
    
    def apply_nmf(self, X: np.ndarray, n_components: int = 2) -> np.ndarray:
        """Apply NMF for non-negative matrix factorization"""
        print(f"\nApplying NMF with {n_components} components...")
        
        # NMF requires non-negative data
        X_non_neg = X - X.min() + 1e-10
        
        self.nmf = NMF(
            n_components=n_components,
            init='random',
            random_state=42,
            max_iter=500
        )
        X_nmf = self.nmf.fit_transform(X_non_neg)
        
        print(f"NMF reconstruction error: {self.nmf.reconstruction_err_:.4f}")
        print(f"NMF output shape: {X_nmf.shape}")
        
        return X_nmf
    
    def plot_pca_variance(self, pca: PCA, figsize: Tuple = (12, 5)):
        """Plot PCA variance explained"""
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # Individual variance
        axes[0].bar(range(1, len(pca.explained_variance_ratio_) + 1),
                   pca.explained_variance_ratio_)
        axes[0].set_xlabel('Principal Component')
        axes[0].set_ylabel('Explained Variance Ratio')
        axes[0].set_title('PCA: Variance Explained per Component')
        axes[0].grid(alpha=0.3)
        
        # Cumulative variance
        cumsum = np.cumsum(pca.explained_variance_ratio_)
        axes[1].plot(range(1, len(cumsum) + 1), cumsum, 'bo-')
        axes[1].axhline(y=0.95, color='r', linestyle='--', label='95% threshold')
        axes[1].set_xlabel('Number of Components')
        axes[1].set_ylabel('Cumulative Explained Variance')
        axes[1].set_title('PCA: Cumulative Variance Explained')
        axes[1].legend()
        axes[1].grid(alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_2d_reduction(self, X_reduced: np.ndarray, y: np.ndarray = None,
                         title: str = "2D Dimensionality Reduction", figsize: Tuple = (10, 8)):
        """Plot 2D dimensionality reduction results"""
        fig, ax = plt.subplots(figsize=figsize)
        
        if y is not None:
            scatter = ax.scatter(X_reduced[:, 0], X_reduced[:, 1],
                               c=y, cmap='viridis', alpha=0.6, s=50)
            plt.colorbar(scatter, ax=ax, label='Class')
        else:
            ax.scatter(X_reduced[:, 0], X_reduced[:, 1], alpha=0.6, s=50)
        
        ax.set_xlabel('Component 1')
        ax.set_ylabel('Component 2')
        ax.set_title(title)
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        return fig


def get_reducer() -> DimensionalityReducer:
    """Factory function to create reducer"""
    return DimensionalityReducer()
