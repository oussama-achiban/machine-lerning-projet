"""
Clustering module
Implements: K-Means, Agglomerative Clustering, DBSCAN
Includes: elbow method, silhouette analysis, visualization
"""

import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score, silhouette_samples, davies_bouldin_score
import matplotlib.pyplot as plt
from typing import Tuple
import seaborn as sns


class Clusterer:
    """Handles clustering operations"""
    
    def __init__(self):
        self.kmeans = None
        self.agglomerative = None
        self.dbscan = None
        self.inertias = []
        self.silhouette_scores = []
    
    def apply_kmeans(self, X: np.ndarray, n_clusters: int = 3, n_init: int = 10) -> np.ndarray:
        """Apply K-Means clustering"""
        print(f"\nApplying K-Means with {n_clusters} clusters...")
        
        self.kmeans = KMeans(n_clusters=n_clusters, n_init=n_init, 
                            random_state=42)
        labels = self.kmeans.fit_predict(X)
        
        print(f"Inertia: {self.kmeans.inertia_:.4f}")
        print(f"Silhouette Score: {silhouette_score(X, labels):.4f}")
        print(f"Davies-Bouldin Index: {davies_bouldin_score(X, labels):.4f}")
        
        return labels
    
    def elbow_method(self, X: np.ndarray, k_range: range = range(1, 11)) \
            -> Tuple[np.ndarray, np.ndarray]:
        """Find optimal number of clusters using elbow method"""
        print("\nPerforming Elbow Method analysis...")
        
        inertias = []
        silhouette_scores = []
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
            labels = kmeans.fit_predict(X)
            inertias.append(kmeans.inertia_)
            
            if k > 1:  # Silhouette score requires at least 2 clusters
                silhouette_scores.append(silhouette_score(X, labels))
            else:
                silhouette_scores.append(0)
        
        self.inertias = inertias
        self.silhouette_scores = silhouette_scores
        
        print(f"Inertias: {inertias}")
        print(f"Silhouette Scores: {silhouette_scores}")
        
        return np.array(inertias), np.array(silhouette_scores)
    
    def apply_agglomerative(self, X: np.ndarray, n_clusters: int = 3,
                           linkage: str = 'ward') -> np.ndarray:
        """Apply Agglomerative Clustering"""
        print(f"\nApplying Agglomerative Clustering ({linkage} linkage)...")
        print(f"Number of clusters: {n_clusters}")
        
        self.agglomerative = AgglomerativeClustering(
            n_clusters=n_clusters,
            linkage=linkage
        )
        labels = self.agglomerative.fit_predict(X)
        
        print(f"Silhouette Score: {silhouette_score(X, labels):.4f}")
        print(f"Davies-Bouldin Index: {davies_bouldin_score(X, labels):.4f}")
        
        return labels
    
    def apply_dbscan(self, X: np.ndarray, eps: float = 0.5, min_samples: int = 5) -> np.ndarray:
        """Apply DBSCAN clustering"""
        print(f"\nApplying DBSCAN (eps={eps}, min_samples={min_samples})...")
        
        self.dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels = self.dbscan.fit_predict(X)
        
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)
        
        print(f"Number of clusters: {n_clusters}")
        print(f"Number of noise points: {n_noise}")
        
        if n_clusters > 1:
            score = silhouette_score(X, labels)
            print(f"Silhouette Score: {score:.4f}")
        
        return labels
    
    def plot_elbow(self, k_range: range = range(1, 11), figsize: Tuple = (12, 5)):
        """Plot elbow method results"""
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        k_list = list(k_range)
        
        # Inertia
        axes[0].plot(k_list, self.inertias, 'bo-', linewidth=2, markersize=8)
        axes[0].set_xlabel('Number of Clusters (k)')
        axes[0].set_ylabel('Inertia')
        axes[0].set_title('Elbow Method - Inertia')
        axes[0].grid(alpha=0.3)
        
        # Silhouette scores
        axes[1].plot(k_list[1:], self.silhouette_scores[1:], 'ro-', linewidth=2, markersize=8)
        axes[1].set_xlabel('Number of Clusters (k)')
        axes[1].set_ylabel('Silhouette Score')
        axes[1].set_title('Silhouette Score vs Number of Clusters')
        axes[1].grid(alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_clusters(self, X: np.ndarray, labels: np.ndarray, title: str = "Clustering Results",
                     figsize: Tuple = (10, 8)):
        """Plot clustering results (assumes 2D data or already reduced)"""
        fig, ax = plt.subplots(figsize=figsize)
        
        if X.shape[1] == 2:
            scatter = ax.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis',
                               alpha=0.6, s=50, edgecolors='k', linewidth=0.5)
            ax.set_xlabel('Feature 1')
            ax.set_ylabel('Feature 2')
        else:
            # For higher dimensions, just show first two principal components
            scatter = ax.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis',
                               alpha=0.6, s=50, edgecolors='k', linewidth=0.5)
            ax.set_xlabel('Component 1')
            ax.set_ylabel('Component 2')
        
        plt.colorbar(scatter, ax=ax, label='Cluster')
        ax.set_title(title)
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_silhouette(self, X: np.ndarray, labels: np.ndarray,
                       title: str = "Silhouette Analysis", figsize: Tuple = (10, 8)):
        """Plot silhouette analysis"""
        fig, ax = plt.subplots(figsize=figsize)
        
        silhouette_vals = silhouette_samples(X, labels)
        
        y_lower = 10
        for i in range(len(np.unique(labels))):
            cluster_silhouette_vals = silhouette_vals[labels == i]
            cluster_silhouette_vals.sort()
            
            size_cluster_i = cluster_silhouette_vals.shape[0]
            y_upper = y_lower + size_cluster_i
            
            ax.fill_betweenx(np.arange(y_lower, y_upper),
                            0, cluster_silhouette_vals,
                            alpha=0.7, label=f'Cluster {i}')
            
            y_lower = y_upper + 10
        
        ax.axvline(x=silhouette_score(X, labels), color="red", linestyle="--",
                  label=f"Average: {silhouette_score(X, labels):.3f}")
        
        ax.set_xlabel('Silhouette Coefficient')
        ax.set_ylabel('Cluster')
        ax.set_title(title)
        ax.legend(loc='best')
        
        plt.tight_layout()
        return fig


def get_clusterer() -> Clusterer:
    """Factory function to create clusterer"""
    return Clusterer()
