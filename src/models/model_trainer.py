"""
Model Trainer Module for Iris Dataset

This module implements various machine learning models for the Iris dataset
classification task, including training, hyperparameter tuning, and evaluation.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import xgboost as xgb
import lightgbm as lgb
import joblib
import yaml
import os
import logging
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class IrisModelTrainer:
    """
    A comprehensive model trainer for the Iris dataset.
    
    This class implements various machine learning algorithms:
    - Supervised Learning: Logistic Regression, SVM, Random Forest, etc.
    - Ensemble Methods: Voting Classifier
    - Advanced Models: XGBoost, LightGBM, Neural Networks
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize the model trainer with configuration.
        
        Args:
            config_path: Path to the configuration file
        """
        self.config = self._load_config(config_path)
        self.models = {}
        self.best_models = {}
        self.model_results = {}
        self.random_state = self.config.get('training', {}).get('random_state', 42)
        self.cv_folds = self.config.get('training', {}).get('cv_folds', 5)
        self.models_path = self.config.get('models', {}).get('save_path', 'models/')
        
        # Create models directory
        os.makedirs(self.models_path, exist_ok=True)
        
        # Initialize models
        self._initialize_models()
    
    def _load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as file:
                return yaml.safe_load(file)
        except FileNotFoundError:
            logger.warning(f"Config file {config_path} not found. Using default settings.")
            return {}
    
    def _initialize_models(self):
        """Initialize all machine learning models."""
        logger.info("Initializing machine learning models...")
        
        # Basic models
        self.models['logistic_regression'] = LogisticRegression(
            random_state=self.random_state,
            max_iter=1000
        )
        
        self.models['svm'] = SVC(
            random_state=self.random_state,
            probability=True
        )
        
        self.models['random_forest'] = RandomForestClassifier(
            random_state=self.random_state,
            n_estimators=100
        )
        
        self.models['decision_tree'] = DecisionTreeClassifier(
            random_state=self.random_state
        )
        
        self.models['knn'] = KNeighborsClassifier()
        
        self.models['naive_bayes'] = GaussianNB()
        
        self.models['neural_network'] = MLPClassifier(
            random_state=self.random_state,
            max_iter=1000,
            hidden_layer_sizes=(100, 50)
        )
        
        # Advanced models
        try:
            self.models['xgboost'] = xgb.XGBClassifier(
                random_state=self.random_state,
                eval_metric='mlogloss'
            )
        except ImportError:
            logger.warning("XGBoost not available. Skipping XGBoost model.")
        
        try:
            self.models['lightgbm'] = lgb.LGBMClassifier(
                random_state=self.random_state,
                verbose=-1
            )
        except ImportError:
            logger.warning("LightGBM not available. Skipping LightGBM model.")
        
        logger.info(f"Initialized {len(self.models)} models")
    
    def train_basic_models(self, X_train: pd.DataFrame, y_train: pd.Series,
                          X_val: pd.DataFrame, y_val: pd.Series) -> Dict[str, Dict[str, float]]:
        """
        Train basic models without hyperparameter tuning.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            
        Returns:
            Dictionary containing model results
        """
        logger.info("Training basic models...")
        
        results = {}
        
        for name, model in self.models.items():
            logger.info(f"Training {name}...")
            
            try:
                # Train model
                model.fit(X_train, y_train)
                
                # Make predictions
                y_pred = model.predict(X_val)
                y_pred_proba = model.predict_proba(X_val) if hasattr(model, 'predict_proba') else None
                
                # Calculate metrics
                accuracy = accuracy_score(y_val, y_pred)
                precision = precision_score(y_val, y_pred, average='weighted')
                recall = recall_score(y_val, y_pred, average='weighted')
                f1 = f1_score(y_val, y_pred, average='weighted')
                
                # Calculate ROC AUC if probabilities are available
                roc_auc = None
                if y_pred_proba is not None:
                    try:
                        roc_auc = roc_auc_score(y_val, y_pred_proba, multi_class='ovr')
                    except:
                        roc_auc = None
                
                # Store results
                results[name] = {
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1,
                    'roc_auc': roc_auc
                }
                
                # Store best model
                self.best_models[name] = model
                
                logger.info(f"{name} - Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
                
            except Exception as e:
                logger.error(f"Error training {name}: {str(e)}")
                results[name] = {
                    'accuracy': 0.0,
                    'precision': 0.0,
                    'recall': 0.0,
                    'f1_score': 0.0,
                    'roc_auc': None
                }
        
        self.model_results = results
        return results
    
    def hyperparameter_tuning(self, X_train: pd.DataFrame, y_train: pd.Series,
                            X_val: pd.DataFrame, y_val: pd.Series) -> Dict[str, Dict[str, float]]:
        """
        Perform hyperparameter tuning for all models.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            
        Returns:
            Dictionary containing tuned model results
        """
        logger.info("Performing hyperparameter tuning...")
        
        param_grids = self.config.get('model_params', {})
        results = {}
        
        for name, model in self.models.items():
            logger.info(f"Tuning hyperparameters for {name}...")
            
            if name in param_grids:
                try:
                    # Perform grid search
                    grid_search = GridSearchCV(
                        model,
                        param_grids[name],
                        cv=self.cv_folds,
                        scoring='accuracy',
                        n_jobs=-1,
                        verbose=0
                    )
                    
                    grid_search.fit(X_train, y_train)
                    
                    # Get best model
                    best_model = grid_search.best_estimator_
                    self.best_models[name] = best_model
                    
                    # Evaluate best model
                    y_pred = best_model.predict(X_val)
                    y_pred_proba = best_model.predict_proba(X_val) if hasattr(best_model, 'predict_proba') else None
                    
                    # Calculate metrics
                    accuracy = accuracy_score(y_val, y_pred)
                    precision = precision_score(y_val, y_pred, average='weighted')
                    recall = recall_score(y_val, y_pred, average='weighted')
                    f1 = f1_score(y_val, y_pred, average='weighted')
                    
                    # Calculate ROC AUC
                    roc_auc = None
                    if y_pred_proba is not None:
                        try:
                            roc_auc = roc_auc_score(y_val, y_pred_proba, multi_class='ovr')
                        except:
                            roc_auc = None
                    
                    results[name] = {
                        'accuracy': accuracy,
                        'precision': precision,
                        'recall': recall,
                        'f1_score': f1,
                        'roc_auc': roc_auc,
                        'best_params': grid_search.best_params_
                    }
                    
                    logger.info(f"{name} - Best Accuracy: {accuracy:.4f}, Best F1: {f1:.4f}")
                    logger.info(f"Best parameters: {grid_search.best_params_}")
                    
                except Exception as e:
                    logger.error(f"Error tuning {name}: {str(e)}")
                    results[name] = {
                        'accuracy': 0.0,
                        'precision': 0.0,
                        'recall': 0.0,
                        'f1_score': 0.0,
                        'roc_auc': None,
                        'best_params': {}
                    }
            else:
                logger.warning(f"No parameter grid found for {name}. Using default parameters.")
                # Train with default parameters
                model.fit(X_train, y_train)
                self.best_models[name] = model
                
                # Evaluate
                y_pred = model.predict(X_val)
                y_pred_proba = model.predict_proba(X_val) if hasattr(model, 'predict_proba') else None
                
                accuracy = accuracy_score(y_val, y_pred)
                precision = precision_score(y_val, y_pred, average='weighted')
                recall = recall_score(y_val, y_pred, average='weighted')
                f1 = f1_score(y_val, y_pred, average='weighted')
                
                roc_auc = None
                if y_pred_proba is not None:
                    try:
                        roc_auc = roc_auc_score(y_val, y_pred_proba, multi_class='ovr')
                    except:
                        roc_auc = None
                
                results[name] = {
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1,
                    'roc_auc': roc_auc,
                    'best_params': {}
                }
        
        self.model_results = results
        return results
    
    def create_ensemble_model(self, X_train: pd.DataFrame, y_train: pd.Series,
                            X_val: pd.DataFrame, y_val: pd.Series) -> Dict[str, float]:
        """
        Create and train an ensemble model using voting classifier.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            
        Returns:
            Dictionary containing ensemble model results
        """
        logger.info("Creating ensemble model...")
        
        # Get best models for ensemble
        ensemble_models = []
        ensemble_names = self.config.get('ensemble', {}).get('include_models', [])
        
        for name in ensemble_names:
            if name in self.best_models:
                ensemble_models.append((name, self.best_models[name]))
        
        if len(ensemble_models) < 2:
            logger.warning("Not enough models for ensemble. Need at least 2 models.")
            return {}
        
        # Create voting classifier
        voting_method = self.config.get('ensemble', {}).get('voting_method', 'soft')
        weights = self.config.get('ensemble', {}).get('weights', 'auto')
        
        ensemble = VotingClassifier(
            estimators=ensemble_models,
            voting=voting_method,
            weights=weights if weights != 'auto' else None
        )
        
        # Train ensemble
        ensemble.fit(X_train, y_train)
        self.best_models['ensemble'] = ensemble
        
        # Evaluate ensemble
        y_pred = ensemble.predict(X_val)
        y_pred_proba = ensemble.predict_proba(X_val)
        
        accuracy = accuracy_score(y_val, y_pred)
        precision = precision_score(y_val, y_pred, average='weighted')
        recall = recall_score(y_val, y_pred, average='weighted')
        f1 = f1_score(y_val, y_pred, average='weighted')
        
        try:
            roc_auc = roc_auc_score(y_val, y_pred_proba, multi_class='ovr')
        except:
            roc_auc = None
        
        results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc
        }
        
        logger.info(f"Ensemble - Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
        
        return results
    
    def cross_validation_evaluation(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Dict[str, float]]:
        """
        Perform cross-validation evaluation for all models.
        
        Args:
            X: Feature matrix
            y: Target vector
            
        Returns:
            Dictionary containing cross-validation results
        """
        logger.info("Performing cross-validation evaluation...")
        
        cv_results = {}
        scoring_metrics = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted']
        
        for name, model in self.best_models.items():
            logger.info(f"Cross-validating {name}...")
            
            cv_scores = {}
            
            for metric in scoring_metrics:
                try:
                    scores = cross_val_score(model, X, y, cv=self.cv_folds, scoring=metric, n_jobs=-1)
                    cv_scores[metric] = {
                        'mean': scores.mean(),
                        'std': scores.std(),
                        'scores': scores.tolist()
                    }
                except Exception as e:
                    logger.warning(f"Could not compute {metric} for {name}: {str(e)}")
                    cv_scores[metric] = {
                        'mean': 0.0,
                        'std': 0.0,
                        'scores': []
                    }
            
            cv_results[name] = cv_scores
        
        return cv_results
    
    def get_feature_importance(self, feature_names: List[str]) -> Dict[str, np.ndarray]:
        """
        Get feature importance for models that support it.
        
        Args:
            feature_names: List of feature names
            
        Returns:
            Dictionary containing feature importance for each model
        """
        logger.info("Extracting feature importance...")
        
        importance_dict = {}
        
        for name, model in self.best_models.items():
            try:
                if hasattr(model, 'feature_importances_'):
                    importance_dict[name] = model.feature_importances_
                elif hasattr(model, 'coef_'):
                    # For linear models, use absolute coefficients
                    importance_dict[name] = np.abs(model.coef_[0])
                else:
                    logger.warning(f"{name} does not support feature importance")
                    importance_dict[name] = None
            except Exception as e:
                logger.error(f"Error extracting feature importance for {name}: {str(e)}")
                importance_dict[name] = None
        
        return importance_dict
    
    def save_models(self) -> None:
        """Save all trained models to disk."""
        logger.info("Saving trained models...")
        
        for name, model in self.best_models.items():
            try:
                model_path = os.path.join(self.models_path, f'{name}.pkl')
                joblib.dump(model, model_path)
                logger.info(f"Saved {name} model to {model_path}")
            except Exception as e:
                logger.error(f"Error saving {name} model: {str(e)}")
    
    def load_models(self) -> None:
        """Load trained models from disk."""
        logger.info("Loading trained models...")
        
        for name in self.best_models.keys():
            try:
                model_path = os.path.join(self.models_path, f'{name}.pkl')
                if os.path.exists(model_path):
                    self.best_models[name] = joblib.load(model_path)
                    logger.info(f"Loaded {name} model from {model_path}")
            except Exception as e:
                logger.error(f"Error loading {name} model: {str(e)}")
    
    def predict(self, X: pd.DataFrame, model_name: str = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions using a specific model or the best model.
        
        Args:
            X: Feature matrix
            model_name: Name of the model to use (if None, use best model)
            
        Returns:
            Tuple of (predictions, probabilities)
        """
        if model_name is None:
            # Use the model with highest accuracy
            best_model_name = max(self.model_results.keys(), 
                                key=lambda x: self.model_results[x]['accuracy'])
            model = self.best_models[best_model_name]
        else:
            model = self.best_models[model_name]
        
        predictions = model.predict(X)
        probabilities = model.predict_proba(X) if hasattr(model, 'predict_proba') else None
        
        return predictions, probabilities
    
    def get_model_summary(self) -> pd.DataFrame:
        """
        Get a summary of all model performances.
        
        Returns:
            DataFrame containing model performance summary
        """
        if not self.model_results:
            return pd.DataFrame()
        
        summary_data = []
        for name, results in self.model_results.items():
            summary_data.append({
                'Model': name,
                'Accuracy': results.get('accuracy', 0.0),
                'Precision': results.get('precision', 0.0),
                'Recall': results.get('recall', 0.0),
                'F1_Score': results.get('f1_score', 0.0),
                'ROC_AUC': results.get('roc_auc', None)
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df = summary_df.sort_values('Accuracy', ascending=False)
        
        return summary_df
    
    def print_model_comparison(self) -> None:
        """Print a comparison of all model performances."""
        summary_df = self.get_model_summary()
        
        if summary_df.empty:
            logger.warning("No model results available for comparison")
            return
        
        print("\n" + "="*80)
        print("MODEL PERFORMANCE COMPARISON")
        print("="*80)
        print(summary_df.to_string(index=False, float_format='%.4f'))
        print("="*80)
        
        # Print best model
        best_model = summary_df.iloc[0]
        print(f"\nBest Model: {best_model['Model']}")
        print(f"Best Accuracy: {best_model['Accuracy']:.4f}")
        print(f"Best F1 Score: {best_model['F1_Score']:.4f}")


def main():
    """Main function to demonstrate model training capabilities."""
    print("IrisModelTrainer module loaded successfully!")
    print("Use this module to train and evaluate various ML models on the Iris dataset.")


if __name__ == "__main__":
    main() 