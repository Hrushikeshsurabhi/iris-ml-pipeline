"""
Main Execution Script for Iris Dataset ML Project

This script orchestrates the entire machine learning pipeline including:
- Data loading and preprocessing
- Exploratory data analysis
- Model training and evaluation
- Visualization generation
- Report creation
"""

import sys
import os
import logging
import yaml
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Import project modules
from src.data.data_loader import IrisDataLoader
from src.visualization.visualizer import IrisVisualizer
from src.models.model_trainer import IrisModelTrainer
from src.evaluation.evaluator import IrisEvaluator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/iris_ml.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class IrisMLPipeline:
    """
    Complete machine learning pipeline for the Iris dataset.
    
    This class orchestrates all components of the ML project:
    1. Data loading and preprocessing
    2. Exploratory data analysis
    3. Model training and evaluation
    4. Visualization generation
    5. Report creation
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize the ML pipeline.
        
        Args:
            config_path: Path to the configuration file
        """
        self.config = self._load_config(config_path)
        self.start_time = datetime.now()
        
        # Initialize components
        self.data_loader = IrisDataLoader(config_path)
        self.visualizer = IrisVisualizer(config_path)
        self.model_trainer = IrisModelTrainer(config_path)
        self.evaluator = IrisEvaluator(config_path)
        
        # Data storage
        self.df_raw = None
        self.df_processed = None
        self.X_train = None
        self.X_val = None
        self.X_test = None
        self.y_train = None
        self.y_val = None
        self.y_test = None
        
        logger.info("Iris ML Pipeline initialized successfully")
    
    def _load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as file:
                return yaml.safe_load(file)
        except FileNotFoundError:
            logger.error(f"Config file {config_path} not found!")
            raise
    
    def run_data_pipeline(self) -> None:
        """Execute the complete data processing pipeline."""
        logger.info("="*60)
        logger.info("STARTING DATA PROCESSING PIPELINE")
        logger.info("="*60)
        
        try:
            # Step 1: Load raw data
            logger.info("Step 1: Loading raw data...")
            self.df_raw = self.data_loader.load_iris_data()
            
            # Step 2: Preprocess data
            logger.info("Step 2: Preprocessing data...")
            self.df_processed = self.data_loader.preprocess_data(self.df_raw)
            
            # Step 3: Split data
            logger.info("Step 3: Splitting data...")
            self.X_train, self.X_val, self.X_test, self.y_train, self.y_val, self.y_test = \
                self.data_loader.split_data(self.df_processed)
            
            # Step 4: Scale features
            logger.info("Step 4: Scaling features...")
            self.X_train_scaled, self.X_val_scaled, self.X_test_scaled = \
                self.data_loader.scale_features(self.X_train, self.X_val, self.X_test)
            
            # Step 5: Encode targets
            logger.info("Step 5: Encoding targets...")
            self.y_train_encoded, self.y_val_encoded, self.y_test_encoded = \
                self.data_loader.encode_target(self.y_train, self.y_val, self.y_test)
            
            # Step 6: Save processed data
            logger.info("Step 6: Saving processed data...")
            self.data_loader.save_processed_data(
                self.X_train_scaled, self.X_val_scaled, self.X_test_scaled,
                self.y_train_encoded, self.y_val_encoded, self.y_test_encoded
            )
            
            logger.info("Data processing pipeline completed successfully!")
            
        except Exception as e:
            logger.error(f"Error in data pipeline: {str(e)}")
            raise
    
    def run_eda_pipeline(self) -> None:
        """Execute the exploratory data analysis pipeline."""
        logger.info("="*60)
        logger.info("STARTING EXPLORATORY DATA ANALYSIS")
        logger.info("="*60)
        
        try:
            # Generate comprehensive EDA visualizations
            logger.info("Generating EDA visualizations...")
            
            # Data overview
            self.visualizer.plot_data_overview(self.df_processed)
            
            # Correlation analysis
            self.visualizer.plot_correlation_matrix(self.df_processed)
            
            # Feature relationships
            self.visualizer.plot_feature_relationships(self.df_processed)
            
            # Distribution analysis
            self.visualizer.plot_boxplots(self.df_processed)
            self.visualizer.plot_violin_plots(self.df_processed)
            
            # Dimensionality reduction visualizations
            X_numeric = self.df_processed.select_dtypes(include=[np.number])
            if 'target' in X_numeric.columns:
                X_numeric = X_numeric.drop(columns=['target'])
            
            # PCA visualization
            target_encoded = self.data_loader.label_encoder.transform(self.df_processed['target'])
            self.visualizer.plot_pca_visualization(
                X_numeric.values, target_encoded, 
                self.data_loader.label_encoder.classes_
            )
            
            # t-SNE visualization
            self.visualizer.plot_tsne_visualization(
                X_numeric.values, target_encoded,
                self.data_loader.label_encoder.classes_
            )
            
            # Interactive dashboard
            self.visualizer.create_interactive_dashboard(self.df_processed)
            
            logger.info("EDA pipeline completed successfully!")
            
        except Exception as e:
            logger.error(f"Error in EDA pipeline: {str(e)}")
            raise
    
    def run_model_pipeline(self) -> None:
        """Execute the model training and evaluation pipeline."""
        logger.info("="*60)
        logger.info("STARTING MODEL TRAINING PIPELINE")
        logger.info("="*60)
        
        try:
            # Step 1: Train basic models
            logger.info("Step 1: Training basic models...")
            basic_results = self.model_trainer.train_basic_models(
                self.X_train_scaled, self.y_train_encoded,
                self.X_val_scaled, self.y_val_encoded
            )
            
            # Step 2: Hyperparameter tuning
            logger.info("Step 2: Performing hyperparameter tuning...")
            tuned_results = self.model_trainer.hyperparameter_tuning(
                self.X_train_scaled, self.y_train_encoded,
                self.X_val_scaled, self.y_val_encoded
            )
            
            # Step 3: Create ensemble model
            logger.info("Step 3: Creating ensemble model...")
            ensemble_results = self.model_trainer.create_ensemble_model(
                self.X_train_scaled, self.y_train_encoded,
                self.X_val_scaled, self.y_val_encoded
            )
            
            # Step 4: Cross-validation evaluation
            logger.info("Step 4: Performing cross-validation...")
            cv_results = self.model_trainer.cross_validation_evaluation(
                self.X_train_scaled, self.y_train_encoded
            )
            
            # Step 5: Save models
            logger.info("Step 5: Saving trained models...")
            self.model_trainer.save_models()
            
            # Step 6: Print model comparison
            self.model_trainer.print_model_comparison()
            
            logger.info("Model training pipeline completed successfully!")
            
        except Exception as e:
            logger.error(f"Error in model pipeline: {str(e)}")
            raise
    
    def run_evaluation_pipeline(self) -> None:
        """Execute the model evaluation and visualization pipeline."""
        logger.info("="*60)
        logger.info("STARTING MODEL EVALUATION PIPELINE")
        logger.info("="*60)
        
        try:
            # Get feature names
            feature_names = self.X_train_scaled.columns.tolist()
            class_names = self.data_loader.label_encoder.classes_.tolist()
            
            # Evaluate each model
            for model_name, model in self.model_trainer.best_models.items():
                logger.info(f"Evaluating {model_name}...")
                
                # Make predictions
                y_pred = model.predict(self.X_test_scaled)
                y_pred_proba = model.predict_proba(self.X_test_scaled) if hasattr(model, 'predict_proba') else None
                
                # Generate evaluation plots
                self.visualizer.plot_confusion_matrix(
                    self.y_test_encoded.values, y_pred, class_names, model_name
                )
                
                if y_pred_proba is not None:
                    self.visualizer.plot_roc_curves(
                        self.y_test_encoded.values, y_pred_proba, class_names, model_name
                    )
                
                # Plot feature importance if available
                if hasattr(model, 'feature_importances_'):
                    self.visualizer.plot_feature_importance(
                        feature_names, model.feature_importances_, model_name
                    )
                elif hasattr(model, 'coef_'):
                    self.visualizer.plot_feature_importance(
                        feature_names, np.abs(model.coef_[0]), model_name
                    )
            
            # Generate model comparison plots
            self.visualizer.plot_model_comparison(
                self.model_trainer.model_results, 'accuracy'
            )
            self.visualizer.plot_model_comparison(
                self.model_trainer.model_results, 'f1_score'
            )
            
            logger.info("Model evaluation pipeline completed successfully!")
            
        except Exception as e:
            logger.error(f"Error in evaluation pipeline: {str(e)}")
            raise
    
    def run_clustering_pipeline(self) -> None:
        """Execute the clustering analysis pipeline."""
        logger.info("="*60)
        logger.info("STARTING CLUSTERING ANALYSIS PIPELINE")
        logger.info("="*60)
        
        try:
            # Get numeric features for clustering
            X_numeric = self.df_processed.select_dtypes(include=[np.number])
            if 'target' in X_numeric.columns:
                X_numeric = X_numeric.drop(columns=['target'])
            
            # Perform clustering analysis
            clustering_results = self.evaluator.perform_clustering_analysis(X_numeric)
            
            # Visualize clustering results
            for method, labels in clustering_results.items():
                self.visualizer.plot_clustering_results(
                    X_numeric.values, labels, f"{method} Clustering"
                )
            
            logger.info("Clustering analysis pipeline completed successfully!")
            
        except Exception as e:
            logger.error(f"Error in clustering pipeline: {str(e)}")
            raise
    
    def generate_report(self) -> None:
        """Generate a comprehensive analysis report."""
        logger.info("="*60)
        logger.info("GENERATING COMPREHENSIVE REPORT")
        logger.info("="*60)
        
        try:
            # Create report content
            report_content = self._create_report_content()
            
            # Save report
            report_path = "reports/iris_analysis_report.md"
            os.makedirs(os.path.dirname(report_path), exist_ok=True)
            
            with open(report_path, 'w') as f:
                f.write(report_content)
            
            logger.info(f"Report saved to {report_path}")
            
        except Exception as e:
            logger.error(f"Error generating report: {str(e)}")
            raise
    
    def _create_report_content(self) -> str:
        """Create the content for the analysis report."""
        end_time = datetime.now()
        duration = end_time - self.start_time
        
        # Get data summary
        data_summary = self.data_loader.get_data_summary(self.df_processed)
        
        # Get model summary
        model_summary = self.model_trainer.get_model_summary()
        
        report = f"""
# Iris Dataset Machine Learning Analysis Report

**Generated on:** {end_time.strftime('%Y-%m-%d %H:%M:%S')}  
**Analysis Duration:** {duration}

## Executive Summary

This report presents a comprehensive analysis of the Iris dataset using various machine learning techniques. The analysis includes data exploration, model training, evaluation, and clustering analysis.

## Dataset Overview

- **Total Samples:** {data_summary['shape'][0]}
- **Features:** {data_summary['shape'][1]}
- **Target Classes:** {len(self.data_loader.label_encoder.classes_)}
- **Class Distribution:** {dict(self.df_processed['target'].value_counts())}

## Data Quality

- **Missing Values:** {sum(data_summary['missing_values'].values())}
- **Duplicate Rows:** {data_summary['duplicates']}

## Feature Engineering

The following engineered features were created:
- Petal to sepal ratios
- Length to width ratios
- Area approximations
- Polynomial features
- Interaction features

## Model Performance Summary

{model_summary.to_string(index=False) if not model_summary.empty else "No model results available"}

## Key Findings

1. **Best Performing Model:** {model_summary.iloc[0]['Model'] if not model_summary.empty else 'N/A'}
2. **Best Accuracy:** {model_summary.iloc[0]['Accuracy']:.4f if not model_summary.empty else 'N/A'}
3. **Dataset Characteristics:** The Iris dataset shows clear separation between classes, making it suitable for classification tasks.

## Recommendations

1. The {model_summary.iloc[0]['Model'] if not model_summary.empty else 'ensemble'} model performed best and should be used for production.
2. Feature engineering improved model performance significantly.
3. The dataset is well-balanced and suitable for machine learning applications.

## Files Generated

- **Raw Data:** data/raw/iris.csv
- **Processed Data:** data/processed/
- **Trained Models:** models/
- **Visualizations:** reports/figures/
- **Interactive Dashboard:** reports/figures/interactive_dashboard.html

## Next Steps

1. Deploy the best model as a web service
2. Implement real-time prediction capabilities
3. Monitor model performance over time
4. Consider collecting additional features for improved accuracy

---
*Report generated by Iris ML Pipeline*
        """
        
        return report
    
    def run_complete_pipeline(self) -> None:
        """Run the complete ML pipeline from start to finish."""
        logger.info("="*80)
        logger.info("STARTING COMPLETE IRIS ML PIPELINE")
        logger.info("="*80)
        
        try:
            # Run all pipelines
            self.run_data_pipeline()
            self.run_eda_pipeline()
            self.run_model_pipeline()
            self.run_evaluation_pipeline()
            self.run_clustering_pipeline()
            self.generate_report()
            
            # Calculate total duration
            end_time = datetime.now()
            total_duration = end_time - self.start_time
            
            logger.info("="*80)
            logger.info("PIPELINE COMPLETED SUCCESSFULLY!")
            logger.info(f"Total Duration: {total_duration}")
            logger.info("="*80)
            
            print(f"\n🎉 Iris ML Pipeline completed successfully!")
            print(f"⏱️  Total Duration: {total_duration}")
            print(f"📊 Check the reports/ directory for results")
            print(f"📈 Interactive dashboard: reports/figures/interactive_dashboard.html")
            
        except Exception as e:
            logger.error(f"Pipeline failed: {str(e)}")
            raise


def main():
    """Main function to run the Iris ML pipeline."""
    try:
        # Create logs directory
        os.makedirs('logs', exist_ok=True)
        
        # Initialize and run pipeline
        pipeline = IrisMLPipeline()
        pipeline.run_complete_pipeline()
        
    except Exception as e:
        logger.error(f"Pipeline execution failed: {str(e)}")
        print(f"❌ Pipeline failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main() 