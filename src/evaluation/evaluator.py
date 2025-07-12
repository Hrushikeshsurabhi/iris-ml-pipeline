"""
Evaluation Module for Iris Dataset

This module provides advanced evaluation capabilities including clustering analysis,
statistical tests, and model interpretability tools.
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler
import yaml
import os
import logging
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class IrisEvaluator:
    """
    Advanced evaluation module for the Iris dataset.
    
    This class provides:
    - Clustering analysis (K-means, Hierarchical, DBSCAN)
    - Statistical evaluation metrics
    - Model interpretability tools
    - Bias-variance analysis
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize the evaluator with configuration.
        
        Args:
            config_path: Path to the configuration file
        """
        self.config = self._load_config(config_path)
        self.clustering_config = self.config.get('clustering', {})
        self.pca_config = self.config.get('pca', {})
        
    def _load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as file:
                return yaml.safe_load(file)
        except FileNotFoundError:
            logger.warning(f"Config file {config_path} not found. Using default settings.")
            return {}
    
    def perform_clustering_analysis(self, X: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Perform comprehensive clustering analysis.
        
        Args:
            X: Feature matrix for clustering
            
        Returns:
            Dictionary containing clustering results for each method
        """
        logger.info("Performing clustering analysis...")
        
        # Scale the data for clustering
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        clustering_results = {}
        
        # K-means clustering
        if 'kmeans' in self.clustering_config.get('algorithms', []):
            logger.info("Performing K-means clustering...")
            kmeans_params = self.clustering_config.get('kmeans', {})
            n_clusters = self.clustering_config.get('n_clusters', 3)
            
            kmeans = KMeans(
                n_clusters=n_clusters,
                n_init=kmeans_params.get('n_init', 10),
                max_iter=kmeans_params.get('max_iter', 300),
                random_state=42
            )
            
            kmeans_labels = kmeans.fit_predict(X_scaled)
            clustering_results['kmeans'] = kmeans_labels
            
            # Calculate clustering metrics
            silhouette_avg = silhouette_score(X_scaled, kmeans_labels)
            calinski_score = calinski_harabasz_score(X_scaled, kmeans_labels)
            davies_score = davies_bouldin_score(X_scaled, kmeans_labels)
            
            logger.info(f"K-means - Silhouette: {silhouette_avg:.3f}, "
                       f"Calinski-Harabasz: {calinski_score:.3f}, "
                       f"Davies-Bouldin: {davies_score:.3f}")
        
        # Hierarchical clustering
        if 'hierarchical' in self.clustering_config.get('algorithms', []):
            logger.info("Performing hierarchical clustering...")
            hierarchical_params = self.clustering_config.get('hierarchical', {})
            n_clusters = self.clustering_config.get('n_clusters', 3)
            
            for linkage in hierarchical_params.get('linkage', ['ward']):
                try:
                    hierarchical = AgglomerativeClustering(
                        n_clusters=n_clusters,
                        linkage=linkage
                    )
                    
                    hierarchical_labels = hierarchical.fit_predict(X_scaled)
                    clustering_results[f'hierarchical_{linkage}'] = hierarchical_labels
                    
                    # Calculate clustering metrics
                    silhouette_avg = silhouette_score(X_scaled, hierarchical_labels)
                    calinski_score = calinski_harabasz_score(X_scaled, hierarchical_labels)
                    davies_score = davies_bouldin_score(X_scaled, hierarchical_labels)
                    
                    logger.info(f"Hierarchical ({linkage}) - Silhouette: {silhouette_avg:.3f}, "
                               f"Calinski-Harabasz: {calinski_score:.3f}, "
                               f"Davies-Bouldin: {davies_score:.3f}")
                               
                except Exception as e:
                    logger.warning(f"Could not perform hierarchical clustering with {linkage}: {str(e)}")
        
        # DBSCAN clustering
        if 'dbscan' in self.clustering_config.get('algorithms', []):
            logger.info("Performing DBSCAN clustering...")
            
            # Try different eps values
            eps_values = [0.5, 1.0, 1.5, 2.0]
            best_silhouette = -1
            best_dbscan_labels = None
            
            for eps in eps_values:
                try:
                    dbscan = DBSCAN(eps=eps, min_samples=5)
                    dbscan_labels = dbscan.fit_predict(X_scaled)
                    
                    # Only evaluate if we have more than one cluster
                    if len(np.unique(dbscan_labels)) > 1:
                        silhouette_avg = silhouette_score(X_scaled, dbscan_labels)
                        if silhouette_avg > best_silhouette:
                            best_silhouette = silhouette_avg
                            best_dbscan_labels = dbscan_labels
                            
                except Exception as e:
                    logger.warning(f"Could not perform DBSCAN with eps={eps}: {str(e)}")
            
            if best_dbscan_labels is not None:
                clustering_results['dbscan'] = best_dbscan_labels
                logger.info(f"DBSCAN - Best Silhouette: {best_silhouette:.3f}")
        
        logger.info(f"Clustering analysis completed. Generated {len(clustering_results)} clustering results.")
        return clustering_results
    
    def perform_pca_analysis(self, X: pd.DataFrame) -> Dict[str, any]:
        """
        Perform Principal Component Analysis.
        
        Args:
            X: Feature matrix
            
        Returns:
            Dictionary containing PCA results
        """
        logger.info("Performing PCA analysis...")
        
        # Scale the data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Perform PCA
        n_components = self.pca_config.get('n_components', 2)
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X_scaled)
        
        # Calculate explained variance
        explained_variance_ratio = pca.explained_variance_ratio_
        cumulative_variance = np.cumsum(explained_variance_ratio)
        
        # Find number of components for threshold
        threshold = self.pca_config.get('explained_variance_threshold', 0.95)
        n_components_threshold = np.argmax(cumulative_variance >= threshold) + 1
        
        pca_results = {
            'X_pca': X_pca,
            'explained_variance_ratio': explained_variance_ratio,
            'cumulative_variance': cumulative_variance,
            'n_components_threshold': n_components_threshold,
            'pca_model': pca
        }
        
        logger.info(f"PCA analysis completed. "
                   f"Explained variance: {explained_variance_ratio}, "
                   f"Components for {threshold*100}% variance: {n_components_threshold}")
        
        return pca_results
    
    def evaluate_clustering_quality(self, X: pd.DataFrame, labels: np.ndarray) -> Dict[str, float]:
        """
        Evaluate clustering quality using multiple metrics.
        
        Args:
            X: Feature matrix
            labels: Cluster labels
            
        Returns:
            Dictionary containing quality metrics
        """
        # Scale the data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        metrics = {}
        
        try:
            metrics['silhouette_score'] = silhouette_score(X_scaled, labels)
        except:
            metrics['silhouette_score'] = None
        
        try:
            metrics['calinski_harabasz_score'] = calinski_harabasz_score(X_scaled, labels)
        except:
            metrics['calinski_harabasz_score'] = None
        
        try:
            metrics['davies_bouldin_score'] = davies_bouldin_score(X_scaled, labels)
        except:
            metrics['davies_bouldin_score'] = None
        
        return metrics
    
    def compare_clustering_methods(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Compare different clustering methods.
        
        Args:
            X: Feature matrix
            
        Returns:
            DataFrame with comparison results
        """
        logger.info("Comparing clustering methods...")
        
        clustering_results = self.perform_clustering_analysis(X)
        
        comparison_data = []
        
        for method, labels in clustering_results.items():
            metrics = self.evaluate_clustering_quality(X, labels)
            
            comparison_data.append({
                'Method': method,
                'Silhouette_Score': metrics.get('silhouette_score'),
                'Calinski_Harabasz_Score': metrics.get('calinski_harabasz_score'),
                'Davies_Bouldin_Score': metrics.get('davies_bouldin_score'),
                'N_Clusters': len(np.unique(labels))
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        logger.info("Clustering comparison completed.")
        return comparison_df
    
    def analyze_feature_importance(self, feature_names: List[str], 
                                 importance_scores: np.ndarray) -> Dict[str, any]:
        """
        Analyze feature importance patterns.
        
        Args:
            feature_names: List of feature names
            importance_scores: Importance scores for each feature
            
        Returns:
            Dictionary containing analysis results
        """
        logger.info("Analyzing feature importance...")
        
        # Create DataFrame for analysis
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importance_scores
        }).sort_values('Importance', ascending=False)
        
        # Calculate statistics
        analysis = {
            'top_features': importance_df.head(5).to_dict('records'),
            'bottom_features': importance_df.tail(5).to_dict('records'),
            'mean_importance': importance_df['Importance'].mean(),
            'std_importance': importance_df['Importance'].std(),
            'importance_range': importance_df['Importance'].max() - importance_df['Importance'].min(),
            'feature_ranking': importance_df.to_dict('records')
        }
        
        logger.info(f"Feature importance analysis completed. "
                   f"Top feature: {analysis['top_features'][0]['Feature']} "
                   f"({analysis['top_features'][0]['Importance']:.3f})")
        
        return analysis
    
    def perform_statistical_analysis(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, any]:
        """
        Perform statistical analysis of the dataset.
        
        Args:
            X: Feature matrix
            y: Target vector
            
        Returns:
            Dictionary containing statistical analysis results
        """
        logger.info("Performing statistical analysis...")
        
        # Basic statistics
        stats = {
            'feature_stats': X.describe().to_dict(),
            'target_distribution': y.value_counts().to_dict(),
            'correlation_matrix': X.corr().to_dict(),
            'feature_variance': X.var().to_dict(),
            'feature_skewness': X.skew().to_dict(),
            'feature_kurtosis': X.kurtosis().to_dict()
        }
        
        # Class-wise statistics
        class_stats = {}
        for class_name in y.unique():
            class_data = X[y == class_name]
            class_stats[class_name] = {
                'count': len(class_data),
                'mean': class_data.mean().to_dict(),
                'std': class_data.std().to_dict()
            }
        
        stats['class_wise_stats'] = class_stats
        
        logger.info("Statistical analysis completed.")
        return stats
    
    def detect_outliers(self, X: pd.DataFrame, method: str = 'iqr') -> Dict[str, any]:
        """
        Detect outliers in the dataset.
        
        Args:
            X: Feature matrix
            method: Method for outlier detection ('iqr', 'zscore')
            
        Returns:
            Dictionary containing outlier analysis results
        """
        logger.info(f"Detecting outliers using {method} method...")
        
        outlier_results = {}
        
        if method == 'iqr':
            # IQR method
            Q1 = X.quantile(0.25)
            Q3 = X.quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = ((X < lower_bound) | (X > upper_bound)).any(axis=1)
            
            outlier_results = {
                'outlier_indices': outliers[outliers].index.tolist(),
                'outlier_count': outliers.sum(),
                'outlier_percentage': (outliers.sum() / len(X)) * 100,
                'bounds': {
                    'lower': lower_bound.to_dict(),
                    'upper': upper_bound.to_dict()
                }
            }
        
        elif method == 'zscore':
            # Z-score method
            z_scores = np.abs((X - X.mean()) / X.std())
            outliers = (z_scores > 3).any(axis=1)
            
            outlier_results = {
                'outlier_indices': outliers[outliers].index.tolist(),
                'outlier_count': outliers.sum(),
                'outlier_percentage': (outliers.sum() / len(X)) * 100,
                'z_scores': z_scores.to_dict()
            }
        
        logger.info(f"Outlier detection completed. Found {outlier_results['outlier_count']} outliers "
                   f"({outlier_results['outlier_percentage']:.2f}%)")
        
        return outlier_results
    
    def generate_evaluation_report(self, X: pd.DataFrame, y: pd.Series = None) -> str:
        """
        Generate a comprehensive evaluation report.
        
        Args:
            X: Feature matrix
            y: Target vector (optional)
            
        Returns:
            String containing the evaluation report
        """
        logger.info("Generating evaluation report...")
        
        report = []
        report.append("# Dataset Evaluation Report\n")
        
        # Basic dataset information
        report.append("## Dataset Overview")
        report.append(f"- **Shape:** {X.shape}")
        report.append(f"- **Features:** {X.shape[1]}")
        report.append(f"- **Samples:** {X.shape[0]}")
        report.append(f"- **Data Types:** {X.dtypes.value_counts().to_dict()}")
        report.append("")
        
        # Statistical analysis
        if y is not None:
            stats = self.perform_statistical_analysis(X, y)
            report.append("## Statistical Analysis")
            report.append(f"- **Target Distribution:** {stats['target_distribution']}")
            report.append("")
        
        # Clustering analysis
        clustering_results = self.perform_clustering_analysis(X)
        report.append("## Clustering Analysis")
        for method, labels in clustering_results.items():
            metrics = self.evaluate_clustering_quality(X, labels)
            report.append(f"### {method.upper()}")
            report.append(f"- **Silhouette Score:** {metrics.get('silhouette_score', 'N/A'):.3f}")
            report.append(f"- **Calinski-Harabasz Score:** {metrics.get('calinski_harabasz_score', 'N/A'):.3f}")
            report.append(f"- **Davies-Bouldin Score:** {metrics.get('davies_bouldin_score', 'N/A'):.3f}")
            report.append(f"- **Number of Clusters:** {len(np.unique(labels))}")
            report.append("")
        
        # PCA analysis
        pca_results = self.perform_pca_analysis(X)
        report.append("## PCA Analysis")
        report.append(f"- **Explained Variance Ratio:** {pca_results['explained_variance_ratio']}")
        report.append(f"- **Cumulative Variance:** {pca_results['cumulative_variance']}")
        report.append(f"- **Components for 95% variance:** {pca_results['n_components_threshold']}")
        report.append("")
        
        # Outlier analysis
        outlier_results = self.detect_outliers(X)
        report.append("## Outlier Analysis")
        report.append(f"- **Outlier Count:** {outlier_results['outlier_count']}")
        report.append(f"- **Outlier Percentage:** {outlier_results['outlier_percentage']:.2f}%")
        report.append("")
        
        report.append("---")
        report.append("*Report generated by IrisEvaluator*")
        
        return "\n".join(report)


def main():
    """Main function to demonstrate evaluation capabilities."""
    print("IrisEvaluator module loaded successfully!")
    print("Use this module to perform advanced evaluation and clustering analysis on the Iris dataset.")


if __name__ == "__main__":
    main() 