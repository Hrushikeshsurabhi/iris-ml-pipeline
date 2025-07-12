"""
Visualization Module for Iris Dataset

This module provides comprehensive visualization capabilities for the Iris dataset
including EDA plots, model performance visualizations, and interactive charts.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import yaml
import os
import logging
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class IrisVisualizer:
    """
    A comprehensive visualizer for the Iris dataset.
    
    This class provides various visualization methods for:
    - Exploratory Data Analysis (EDA)
    - Feature distributions and relationships
    - Model performance evaluation
    - Clustering results
    - Interactive visualizations
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize the visualizer with configuration.
        
        Args:
            config_path: Path to the configuration file
        """
        self.config = self._load_config(config_path)
        self.figsize = tuple(self.config.get('visualization', {}).get('figsize', [12, 8]))
        self.dpi = self.config.get('visualization', {}).get('dpi', 300)
        self.save_format = self.config.get('visualization', {}).get('save_format', 'png')
        self.plot_path = self.config.get('evaluation', {}).get('plot_path', 'reports/figures/')
        
        # Create plot directory
        os.makedirs(self.plot_path, exist_ok=True)
        
        # Setup plotting style
        self.setup_plotting_style()
    
    def _load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as file:
                return yaml.safe_load(file)
        except FileNotFoundError:
            logger.warning(f"Config file {config_path} not found. Using default settings.")
            return {}
    
    def setup_plotting_style(self):
        """Setup matplotlib and seaborn plotting styles."""
        style = self.config.get('visualization', {}).get('style', 'seaborn-v0_8')
        palette = self.config.get('visualization', {}).get('color_palette', 'Set2')
        
        plt.style.use(style)
        sns.set_palette(palette)
        
        # Set default figure size
        plt.rcParams['figure.figsize'] = self.figsize
        plt.rcParams['figure.dpi'] = self.dpi
        
        logger.info(f"Plotting style set to: {style}")
    
    def plot_data_overview(self, df: pd.DataFrame, save: bool = True) -> None:
        """
        Create a comprehensive overview of the dataset.
        
        Args:
            df: DataFrame to visualize
            save: Whether to save the plot
        """
        logger.info("Creating data overview plots...")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Iris Dataset Overview', fontsize=16, fontweight='bold')
        
        # 1. Target distribution
        target_counts = df['target'].value_counts()
        axes[0, 0].pie(target_counts.values, labels=target_counts.index, autopct='%1.1f%%')
        axes[0, 0].set_title('Target Distribution')
        
        # 2. Feature distributions by target
        features = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
        for i, feature in enumerate(features):
            row, col = (i + 1) // 3, (i + 1) % 3
            for target in df['target'].unique():
                data = df[df['target'] == target][feature]
                axes[row, col].hist(data, alpha=0.7, label=target, bins=15)
            axes[row, col].set_title(f'{feature} Distribution')
            axes[row, col].legend()
            axes[row, col].set_xlabel(feature)
            axes[row, col].set_ylabel('Frequency')
        
        plt.tight_layout()
        
        if save:
            plt.savefig(os.path.join(self.plot_path, 'data_overview.png'), 
                       dpi=self.dpi, bbox_inches='tight')
            logger.info("Data overview plot saved")
        
        plt.show()
    
    def plot_correlation_matrix(self, df: pd.DataFrame, save: bool = True) -> None:
        """
        Plot correlation matrix of numeric features.
        
        Args:
            df: DataFrame to visualize
            save: Whether to save the plot
        """
        logger.info("Creating correlation matrix plot...")
        
        # Select numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if 'target' in numeric_cols:
            numeric_cols.remove('target')
        
        correlation_matrix = df[numeric_cols].corr()
        
        plt.figure(figsize=(10, 8))
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='coolwarm', 
                   center=0, square=True, linewidths=0.5)
        plt.title('Feature Correlation Matrix', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save:
            plt.savefig(os.path.join(self.plot_path, 'correlation_matrix.png'), 
                       dpi=self.dpi, bbox_inches='tight')
            logger.info("Correlation matrix plot saved")
        
        plt.show()
    
    def plot_feature_relationships(self, df: pd.DataFrame, save: bool = True) -> None:
        """
        Plot pairwise relationships between features.
        
        Args:
            df: DataFrame to visualize
            save: Whether to save the plot
        """
        logger.info("Creating feature relationship plots...")
        
        # Select numeric features
        features = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
        
        # Create pairplot
        pair_plot = sns.pairplot(df, vars=features, hue='target', diag_kind='hist')
        pair_plot.fig.suptitle('Feature Relationships by Target Class', y=1.02, fontsize=16)
        pair_plot.fig.subplots_adjust(top=0.95)
        
        if save:
            pair_plot.savefig(os.path.join(self.plot_path, 'feature_relationships.png'), 
                             dpi=self.dpi, bbox_inches='tight')
            logger.info("Feature relationships plot saved")
        
        plt.show()
    
    def plot_boxplots(self, df: pd.DataFrame, save: bool = True) -> None:
        """
        Create boxplots for each feature by target class.
        
        Args:
            df: DataFrame to visualize
            save: Whether to save the plot
        """
        logger.info("Creating boxplots...")
        
        features = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Feature Distributions by Target Class (Boxplots)', fontsize=16, fontweight='bold')
        
        for i, feature in enumerate(features):
            row, col = i // 2, i % 2
            sns.boxplot(data=df, x='target', y=feature, ax=axes[row, col])
            axes[row, col].set_title(f'{feature} by Target Class')
            axes[row, col].set_xlabel('Target Class')
            axes[row, col].set_ylabel(feature)
            axes[row, col].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save:
            plt.savefig(os.path.join(self.plot_path, 'boxplots.png'), 
                       dpi=self.dpi, bbox_inches='tight')
            logger.info("Boxplots saved")
        
        plt.show()
    
    def plot_violin_plots(self, df: pd.DataFrame, save: bool = True) -> None:
        """
        Create violin plots for each feature by target class.
        
        Args:
            df: DataFrame to visualize
            save: Whether to save the plot
        """
        logger.info("Creating violin plots...")
        
        features = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Feature Distributions by Target Class (Violin Plots)', fontsize=16, fontweight='bold')
        
        for i, feature in enumerate(features):
            row, col = i // 2, i % 2
            sns.violinplot(data=df, x='target', y=feature, ax=axes[row, col])
            axes[row, col].set_title(f'{feature} by Target Class')
            axes[row, col].set_xlabel('Target Class')
            axes[row, col].set_ylabel(feature)
            axes[row, col].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save:
            plt.savefig(os.path.join(self.plot_path, 'violin_plots.png'), 
                       dpi=self.dpi, bbox_inches='tight')
            logger.info("Violin plots saved")
        
        plt.show()
    
    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray, 
                            class_names: List[str], model_name: str = "Model", 
                            save: bool = True) -> None:
        """
        Plot confusion matrix for classification results.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            class_names: Names of the classes
            model_name: Name of the model for the title
            save: Whether to save the plot
        """
        logger.info(f"Creating confusion matrix for {model_name}...")
        
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names)
        plt.title(f'Confusion Matrix - {model_name}', fontsize=14, fontweight='bold')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.tight_layout()
        
        if save:
            plt.savefig(os.path.join(self.plot_path, f'confusion_matrix_{model_name.lower().replace(" ", "_")}.png'), 
                       dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Confusion matrix for {model_name} saved")
        
        plt.show()
    
    def plot_roc_curves(self, y_true: np.ndarray, y_pred_proba: np.ndarray, 
                       class_names: List[str], model_name: str = "Model", 
                       save: bool = True) -> None:
        """
        Plot ROC curves for multi-class classification.
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            class_names: Names of the classes
            model_name: Name of the model for the title
            save: Whether to save the plot
        """
        logger.info(f"Creating ROC curves for {model_name}...")
        
        n_classes = len(class_names)
        
        # Compute ROC curve and ROC area for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve((y_true == i).astype(int), y_pred_proba[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        
        plt.figure(figsize=(10, 8))
        
        for i in range(n_classes):
            plt.plot(fpr[i], tpr[i], lw=2,
                    label=f'{class_names[i]} (AUC = {roc_auc[i]:.2f})')
        
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curves - {model_name}', fontsize=14, fontweight='bold')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save:
            plt.savefig(os.path.join(self.plot_path, f'roc_curves_{model_name.lower().replace(" ", "_")}.png'), 
                       dpi=self.dpi, bbox_inches='tight')
            logger.info(f"ROC curves for {model_name} saved")
        
        plt.show()
    
    def plot_model_comparison(self, results: Dict[str, Dict[str, float]], 
                            metric: str = 'accuracy', save: bool = True) -> None:
        """
        Plot model comparison based on a specific metric.
        
        Args:
            results: Dictionary containing model results
            metric: Metric to compare (accuracy, precision, recall, f1_score)
            save: Whether to save the plot
        """
        logger.info(f"Creating model comparison plot for {metric}...")
        
        models = list(results.keys())
        scores = [results[model][metric] for model in models]
        
        plt.figure(figsize=(12, 6))
        bars = plt.bar(models, scores, color='skyblue', alpha=0.7)
        
        # Add value labels on bars
        for bar, score in zip(bars, scores):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.title(f'Model Comparison - {metric.title()}', fontsize=14, fontweight='bold')
        plt.xlabel('Models')
        plt.ylabel(metric.title())
        plt.xticks(rotation=45)
        plt.ylim(0, 1.1)
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        
        if save:
            plt.savefig(os.path.join(self.plot_path, f'model_comparison_{metric}.png'), 
                       dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Model comparison plot for {metric} saved")
        
        plt.show()
    
    def plot_pca_visualization(self, X: np.ndarray, y: np.ndarray, 
                             class_names: List[str], save: bool = True) -> None:
        """
        Create PCA visualization of the data.
        
        Args:
            X: Feature matrix
            y: Target labels
            class_names: Names of the classes
            save: Whether to save the plot
        """
        logger.info("Creating PCA visualization...")
        
        # Apply PCA
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)
        
        # Create scatter plot
        plt.figure(figsize=(10, 8))
        
        for i, class_name in enumerate(class_names):
            mask = y == i
            plt.scatter(X_pca[mask, 0], X_pca[mask, 1], label=class_name, alpha=0.7)
        
        plt.xlabel(f'First Principal Component ({pca.explained_variance_ratio_[0]:.1%} variance)')
        plt.ylabel(f'Second Principal Component ({pca.explained_variance_ratio_[1]:.1%} variance)')
        plt.title('PCA Visualization of Iris Dataset', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save:
            plt.savefig(os.path.join(self.plot_path, 'pca_visualization.png'), 
                       dpi=self.dpi, bbox_inches='tight')
            logger.info("PCA visualization saved")
        
        plt.show()
    
    def plot_tsne_visualization(self, X: np.ndarray, y: np.ndarray, 
                              class_names: List[str], save: bool = True) -> None:
        """
        Create t-SNE visualization of the data.
        
        Args:
            X: Feature matrix
            y: Target labels
            class_names: Names of the classes
            save: Whether to save the plot
        """
        logger.info("Creating t-SNE visualization...")
        
        # Apply t-SNE
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        X_tsne = tsne.fit_transform(X)
        
        # Create scatter plot
        plt.figure(figsize=(10, 8))
        
        for i, class_name in enumerate(class_names):
            mask = y == i
            plt.scatter(X_tsne[mask, 0], X_tsne[mask, 1], label=class_name, alpha=0.7)
        
        plt.xlabel('First t-SNE Component')
        plt.ylabel('Second t-SNE Component')
        plt.title('t-SNE Visualization of Iris Dataset', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save:
            plt.savefig(os.path.join(self.plot_path, 'tsne_visualization.png'), 
                       dpi=self.dpi, bbox_inches='tight')
            logger.info("t-SNE visualization saved")
        
        plt.show()
    
    def plot_clustering_results(self, X: np.ndarray, labels: np.ndarray, 
                              title: str = "Clustering Results", save: bool = True) -> None:
        """
        Plot clustering results using PCA for dimensionality reduction.
        
        Args:
            X: Feature matrix
            labels: Cluster labels
            title: Title for the plot
            save: Whether to save the plot
        """
        logger.info("Creating clustering visualization...")
        
        # Apply PCA for visualization
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)
        
        # Create scatter plot
        plt.figure(figsize=(10, 8))
        
        unique_labels = np.unique(labels)
        colors = plt.cm.Set3(np.linspace(0, 1, len(unique_labels)))
        
        for i, label in enumerate(unique_labels):
            mask = labels == label
            plt.scatter(X_pca[mask, 0], X_pca[mask, 1], 
                       c=[colors[i]], label=f'Cluster {label}', alpha=0.7)
        
        plt.xlabel(f'First Principal Component ({pca.explained_variance_ratio_[0]:.1%} variance)')
        plt.ylabel(f'Second Principal Component ({pca.explained_variance_ratio_[1]:.1%} variance)')
        plt.title(title, fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save:
            plt.savefig(os.path.join(self.plot_path, f'clustering_{title.lower().replace(" ", "_")}.png'), 
                       dpi=self.dpi, bbox_inches='tight')
            logger.info("Clustering visualization saved")
        
        plt.show()
    
    def create_interactive_dashboard(self, df: pd.DataFrame) -> None:
        """
        Create an interactive dashboard using Plotly.
        
        Args:
            df: DataFrame to visualize
        """
        logger.info("Creating interactive dashboard...")
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Feature Distributions', 'Correlation Heatmap', 
                          '3D Scatter Plot', 'Feature Relationships'),
            specs=[[{"type": "histogram"}, {"type": "heatmap"}],
                   [{"type": "scatter3d"}, {"type": "scatter"}]]
        )
        
        # 1. Feature distributions
        features = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
        for feature in features:
            for target in df['target'].unique():
                data = df[df['target'] == target][feature]
                fig.add_trace(
                    go.Histogram(x=data, name=f'{target} - {feature}', opacity=0.7),
                    row=1, col=1
                )
        
        # 2. Correlation heatmap
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if 'target' in numeric_cols:
            numeric_cols.remove('target')
        
        corr_matrix = df[numeric_cols].corr()
        fig.add_trace(
            go.Heatmap(z=corr_matrix.values, x=corr_matrix.columns, y=corr_matrix.columns,
                      colorscale='RdBu', zmid=0),
            row=1, col=2
        )
        
        # 3. 3D scatter plot
        for target in df['target'].unique():
            data = df[df['target'] == target]
            fig.add_trace(
                go.Scatter3d(x=data['sepal length (cm)'], 
                           y=data['sepal width (cm)'], 
                           z=data['petal length (cm)'],
                           mode='markers', name=target, opacity=0.7),
                row=2, col=1
            )
        
        # 4. Feature relationship
        fig.add_trace(
            go.Scatter(x=df['petal length (cm)'], y=df['petal width (cm)'],
                      mode='markers', marker=dict(color=df['target'].astype('category').cat.codes),
                      text=df['target'], name='Petal Length vs Width'),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            title_text="Interactive Iris Dataset Dashboard",
            showlegend=True,
            height=800
        )
        
        # Save interactive plot
        fig.write_html(os.path.join(self.plot_path, 'interactive_dashboard.html'))
        logger.info("Interactive dashboard saved")
        
        # Show the plot
        fig.show()
    
    def plot_feature_importance(self, feature_names: List[str], importance_scores: np.ndarray, 
                              model_name: str = "Model", save: bool = True) -> None:
        """
        Plot feature importance scores.
        
        Args:
            feature_names: Names of the features
            importance_scores: Importance scores for each feature
            model_name: Name of the model
            save: Whether to save the plot
        """
        logger.info(f"Creating feature importance plot for {model_name}...")
        
        # Create DataFrame for easier plotting
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importance_scores
        }).sort_values('Importance', ascending=True)
        
        plt.figure(figsize=(10, 8))
        bars = plt.barh(importance_df['Feature'], importance_df['Importance'], 
                       color='lightcoral', alpha=0.7)
        
        # Add value labels
        for bar, importance in zip(bars, importance_df['Importance']):
            plt.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                    f'{importance:.3f}', ha='left', va='center', fontweight='bold')
        
        plt.title(f'Feature Importance - {model_name}', fontsize=14, fontweight='bold')
        plt.xlabel('Importance Score')
        plt.ylabel('Features')
        plt.grid(True, alpha=0.3, axis='x')
        plt.tight_layout()
        
        if save:
            plt.savefig(os.path.join(self.plot_path, f'feature_importance_{model_name.lower().replace(" ", "_")}.png'), 
                       dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Feature importance plot for {model_name} saved")
        
        plt.show()


def main():
    """Main function to demonstrate visualization capabilities."""
    # This would be used when running the module directly
    # For now, it's a placeholder
    print("IrisVisualizer module loaded successfully!")
    print("Use this module to create comprehensive visualizations for the Iris dataset.")


if __name__ == "__main__":
    main() 