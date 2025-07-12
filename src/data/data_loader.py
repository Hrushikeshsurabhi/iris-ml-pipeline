"""
Data Loader Module for Iris Dataset

This module handles loading, preprocessing, and splitting the Iris dataset
for machine learning tasks.
"""

import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import yaml
import os
import logging
from typing import Tuple, Optional
import joblib

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class IrisDataLoader:
    """
    A comprehensive data loader for the Iris dataset.
    
    This class handles:
    - Loading the Iris dataset
    - Data preprocessing and cleaning
    - Feature engineering
    - Train/validation/test splitting
    - Data scaling and encoding
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize the data loader with configuration.
        
        Args:
            config_path: Path to the configuration file
        """
        self.config = self._load_config(config_path)
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_names = None
        self.target_names = None
        
    def _load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as file:
                return yaml.safe_load(file)
        except FileNotFoundError:
            logger.warning(f"Config file {config_path} not found. Using default settings.")
            return {}
    
    def load_iris_data(self, save_raw: bool = True) -> pd.DataFrame:
        """
        Load the Iris dataset from sklearn.
        
        Args:
            save_raw: Whether to save the raw data to CSV
            
        Returns:
            DataFrame containing the Iris dataset
        """
        logger.info("Loading Iris dataset...")
        
        # Load dataset from sklearn
        iris = load_iris()
        
        # Create DataFrame
        df = pd.DataFrame(
            data=np.c_[iris.data, iris.target],
            columns=list(iris.feature_names) + ['target']
        )
        
        # Store feature and target names
        self.feature_names = iris.feature_names
        self.target_names = iris.target_names
        
        # Map target values to class names
        df['target'] = df['target'].map({
            0: 'setosa',
            1: 'versicolor', 
            2: 'virginica'
        })
        
        logger.info(f"Loaded {len(df)} samples with {len(self.feature_names)} features")
        logger.info(f"Feature names: {self.feature_names}")
        logger.info(f"Target classes: {self.target_names}")
        
        # Save raw data if requested
        if save_raw:
            os.makedirs(os.path.dirname(self.config.get('data', {}).get('raw_data_path', 'data/raw/iris.csv')), exist_ok=True)
            df.to_csv(self.config.get('data', {}).get('raw_data_path', 'data/raw/iris.csv'), index=False)
            logger.info("Raw data saved to CSV")
        
        return df
    
    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess the dataset including cleaning and feature engineering.
        
        Args:
            df: Raw DataFrame
            
        Returns:
            Preprocessed DataFrame
        """
        logger.info("Preprocessing data...")
        
        # Create a copy to avoid modifying original data
        df_processed = df.copy()
        
        # Check for missing values
        missing_values = df_processed.isnull().sum()
        if missing_values.sum() > 0:
            logger.warning(f"Found missing values: {missing_values}")
            # For Iris dataset, we can safely drop rows with missing values
            df_processed = df_processed.dropna()
        
        # Check for duplicates
        duplicates = df_processed.duplicated().sum()
        if duplicates > 0:
            logger.warning(f"Found {duplicates} duplicate rows")
            df_processed = df_processed.drop_duplicates()
        
        # Feature engineering
        if self.config.get('feature_engineering', {}).get('create_ratios', True):
            df_processed = self._create_ratio_features(df_processed)
        
        if self.config.get('feature_engineering', {}).get('create_polynomials', True):
            df_processed = self._create_polynomial_features(df_processed)
        
        logger.info(f"Preprocessing complete. Final shape: {df_processed.shape}")
        return df_processed
    
    def _create_ratio_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create ratio features from existing features."""
        logger.info("Creating ratio features...")
        
        # Petal to sepal ratios
        df['petal_to_sepal_length_ratio'] = df['petal length (cm)'] / df['sepal length (cm)']
        df['petal_to_sepal_width_ratio'] = df['petal width (cm)'] / df['sepal width (cm)']
        
        # Length to width ratios
        df['sepal_length_to_width_ratio'] = df['sepal length (cm)'] / df['sepal width (cm)']
        df['petal_length_to_width_ratio'] = df['petal length (cm)'] / df['petal width (cm)']
        
        # Area approximations
        df['sepal_area'] = df['sepal length (cm)'] * df['sepal width (cm)']
        df['petal_area'] = df['petal length (cm)'] * df['petal width (cm)']
        
        logger.info("Ratio features created successfully")
        return df
    
    def _create_polynomial_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create polynomial features."""
        logger.info("Creating polynomial features...")
        
        # Get numeric columns (excluding target)
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if 'target' in numeric_cols:
            numeric_cols.remove('target')
        
        # Create squared features
        for col in numeric_cols:
            df[f'{col}_squared'] = df[col] ** 2
        
        # Create interaction features (pairwise products)
        for i, col1 in enumerate(numeric_cols):
            for col2 in numeric_cols[i+1:]:
                df[f'{col1}_{col2}_interaction'] = df[col1] * df[col2]
        
        logger.info("Polynomial features created successfully")
        return df
    
    def split_data(self, df: pd.DataFrame, target_col: str = 'target') -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
        """
        Split data into train, validation, and test sets.
        
        Args:
            df: Preprocessed DataFrame
            target_col: Name of the target column
            
        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        logger.info("Splitting data into train/validation/test sets...")
        
        # Separate features and target
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        # Get split ratios from config
        test_size = self.config.get('training', {}).get('test_size', 0.2)
        val_size = self.config.get('training', {}).get('validation_size', 0.1)
        random_state = self.config.get('training', {}).get('random_state', 42)
        
        # First split: train+val and test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Second split: train and validation
        val_size_adjusted = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, random_state=random_state, stratify=y_temp
        )
        
        logger.info(f"Data split complete:")
        logger.info(f"  Train set: {X_train.shape[0]} samples")
        logger.info(f"  Validation set: {X_val.shape[0]} samples")
        logger.info(f"  Test set: {X_test.shape[0]} samples")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def scale_features(self, X_train: pd.DataFrame, X_val: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Scale features using StandardScaler.
        
        Args:
            X_train: Training features
            X_val: Validation features
            X_test: Test features
            
        Returns:
            Tuple of scaled features
        """
        logger.info("Scaling features...")
        
        # Fit scaler on training data
        self.scaler.fit(X_train)
        
        # Transform all datasets
        X_train_scaled = pd.DataFrame(
            self.scaler.transform(X_train),
            columns=X_train.columns,
            index=X_train.index
        )
        
        X_val_scaled = pd.DataFrame(
            self.scaler.transform(X_val),
            columns=X_val.columns,
            index=X_val.index
        )
        
        X_test_scaled = pd.DataFrame(
            self.scaler.transform(X_test),
            columns=X_test.columns,
            index=X_test.index
        )
        
        logger.info("Feature scaling complete")
        return X_train_scaled, X_val_scaled, X_test_scaled
    
    def encode_target(self, y_train: pd.Series, y_val: pd.Series, y_test: pd.Series) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Encode target labels to numeric values.
        
        Args:
            y_train: Training targets
            y_val: Validation targets
            y_test: Test targets
            
        Returns:
            Tuple of encoded targets
        """
        logger.info("Encoding target labels...")
        
        # Fit encoder on training data
        self.label_encoder.fit(y_train)
        
        # Transform all targets
        y_train_encoded = pd.Series(
            self.label_encoder.transform(y_train),
            index=y_train.index
        )
        
        y_val_encoded = pd.Series(
            self.label_encoder.transform(y_val),
            index=y_val.index
        )
        
        y_test_encoded = pd.Series(
            self.label_encoder.transform(y_test),
            index=y_test.index
        )
        
        logger.info(f"Target encoding complete. Classes: {self.label_encoder.classes_}")
        return y_train_encoded, y_val_encoded, y_test_encoded
    
    def save_processed_data(self, X_train: pd.DataFrame, X_val: pd.DataFrame, X_test: pd.DataFrame,
                          y_train: pd.Series, y_val: pd.Series, y_test: pd.Series) -> None:
        """
        Save processed data to CSV files.
        
        Args:
            X_train, X_val, X_test: Feature DataFrames
            y_train, y_val, y_test: Target Series
        """
        logger.info("Saving processed data...")
        
        # Create processed data directory
        processed_path = self.config.get('data', {}).get('processed_data_path', 'data/processed/')
        os.makedirs(processed_path, exist_ok=True)
        
        # Save train data
        train_data = X_train.copy()
        train_data['target'] = y_train
        train_data.to_csv(os.path.join(processed_path, 'train.csv'), index=False)
        
        # Save validation data
        val_data = X_val.copy()
        val_data['target'] = y_val
        val_data.to_csv(os.path.join(processed_path, 'validation.csv'), index=False)
        
        # Save test data
        test_data = X_test.copy()
        test_data['target'] = y_test
        test_data.to_csv(os.path.join(processed_path, 'test.csv'), index=False)
        
        # Save scaler and encoder
        models_path = self.config.get('models', {}).get('save_path', 'models/')
        os.makedirs(models_path, exist_ok=True)
        
        joblib.dump(self.scaler, os.path.join(models_path, 'scaler.pkl'))
        joblib.dump(self.label_encoder, os.path.join(models_path, 'label_encoder.pkl'))
        
        logger.info("Processed data saved successfully")
    
    def get_data_summary(self, df: pd.DataFrame) -> dict:
        """
        Generate a comprehensive data summary.
        
        Args:
            df: DataFrame to summarize
            
        Returns:
            Dictionary containing data summary
        """
        summary = {
            'shape': df.shape,
            'columns': df.columns.tolist(),
            'dtypes': df.dtypes.to_dict(),
            'missing_values': df.isnull().sum().to_dict(),
            'duplicates': df.duplicated().sum(),
            'numeric_summary': df.describe().to_dict() if df.select_dtypes(include=[np.number]).shape[1] > 0 else {},
            'categorical_summary': {}
        }
        
        # Add categorical summary
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            summary['categorical_summary'][col] = df[col].value_counts().to_dict()
        
        return summary
    
    def load_processed_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
        """
        Load previously processed data from CSV files.
        
        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        logger.info("Loading processed data...")
        
        processed_path = self.config.get('data', {}).get('processed_data_path', 'data/processed/')
        
        # Load data
        train_data = pd.read_csv(os.path.join(processed_path, 'train.csv'))
        val_data = pd.read_csv(os.path.join(processed_path, 'validation.csv'))
        test_data = pd.read_csv(os.path.join(processed_path, 'test.csv'))
        
        # Separate features and target
        X_train = train_data.drop(columns=['target'])
        y_train = train_data['target']
        
        X_val = val_data.drop(columns=['target'])
        y_val = val_data['target']
        
        X_test = test_data.drop(columns=['target'])
        y_test = test_data['target']
        
        logger.info("Processed data loaded successfully")
        return X_train, X_val, X_test, y_train, y_val, y_test


def main():
    """Main function to demonstrate data loading and processing."""
    # Initialize data loader
    loader = IrisDataLoader()
    
    # Load and process data
    df = loader.load_iris_data()
    df_processed = loader.preprocess_data(df)
    
    # Get data summary
    summary = loader.get_data_summary(df_processed)
    print("Data Summary:")
    print(f"Shape: {summary['shape']}")
    print(f"Columns: {summary['columns']}")
    print(f"Missing values: {summary['missing_values']}")
    print(f"Duplicates: {summary['duplicates']}")
    
    # Split data
    X_train, X_val, X_test, y_train, y_val, y_test = loader.split_data(df_processed)
    
    # Scale features
    X_train_scaled, X_val_scaled, X_test_scaled = loader.scale_features(X_train, X_val, X_test)
    
    # Encode targets
    y_train_encoded, y_val_encoded, y_test_encoded = loader.encode_target(y_train, y_val, y_test)
    
    # Save processed data
    loader.save_processed_data(X_train_scaled, X_val_scaled, X_test_scaled,
                             y_train_encoded, y_val_encoded, y_test_encoded)
    
    print("Data processing pipeline completed successfully!")


if __name__ == "__main__":
    main() 