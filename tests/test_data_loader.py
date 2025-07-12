"""
Unit tests for the data loader module.
"""

import unittest
import pandas as pd
import numpy as np
import tempfile
import os
import sys

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data.data_loader import IrisDataLoader


class TestIrisDataLoader(unittest.TestCase):
    """Test cases for IrisDataLoader class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.loader = IrisDataLoader()
    
    def test_load_iris_data(self):
        """Test loading Iris dataset."""
        df = self.loader.load_iris_data(save_raw=False)
        
        # Check basic properties
        self.assertEqual(df.shape[0], 150)  # 150 samples
        self.assertEqual(df.shape[1], 5)    # 4 features + target
        self.assertIn('target', df.columns)
        
        # Check feature names
        expected_features = ['sepal length (cm)', 'sepal width (cm)', 
                           'petal length (cm)', 'petal width (cm)']
        for feature in expected_features:
            self.assertIn(feature, df.columns)
        
        # Check target values
        expected_targets = ['setosa', 'versicolor', 'virginica']
        for target in expected_targets:
            self.assertIn(target, df['target'].values)
    
    def test_preprocess_data(self):
        """Test data preprocessing."""
        # Load raw data
        df_raw = self.loader.load_iris_data(save_raw=False)
        
        # Preprocess data
        df_processed = self.loader.preprocess_data(df_raw)
        
        # Check that processed data has more features
        self.assertGreater(len(df_processed.columns), len(df_raw.columns))
        
        # Check for engineered features
        engineered_features = ['petal_to_sepal_length_ratio', 'sepal_area', 'petal_area']
        for feature in engineered_features:
            self.assertIn(feature, df_processed.columns)
    
    def test_split_data(self):
        """Test data splitting."""
        # Load and preprocess data
        df_raw = self.loader.load_iris_data(save_raw=False)
        df_processed = self.loader.preprocess_data(df_raw)
        
        # Split data
        X_train, X_val, X_test, y_train, y_val, y_test = self.loader.split_data(df_processed)
        
        # Check shapes
        total_samples = len(df_processed)
        self.assertEqual(len(X_train) + len(X_val) + len(X_test), total_samples)
        self.assertEqual(len(y_train) + len(y_val) + len(y_test), total_samples)
        
        # Check that splits are not empty
        self.assertGreater(len(X_train), 0)
        self.assertGreater(len(X_val), 0)
        self.assertGreater(len(X_test), 0)
    
    def test_scale_features(self):
        """Test feature scaling."""
        # Load and preprocess data
        df_raw = self.loader.load_iris_data(save_raw=False)
        df_processed = self.loader.preprocess_data(df_raw)
        X_train, X_val, X_test, y_train, y_val, y_test = self.loader.split_data(df_processed)
        
        # Scale features
        X_train_scaled, X_val_scaled, X_test_scaled = self.loader.scale_features(X_train, X_val, X_test)
        
        # Check that scaled data has same shape
        self.assertEqual(X_train_scaled.shape, X_train.shape)
        self.assertEqual(X_val_scaled.shape, X_val.shape)
        self.assertEqual(X_test_scaled.shape, X_test.shape)
        
        # Check that scaled data has different values (not all zeros)
        self.assertFalse(np.allclose(X_train_scaled, X_train))
    
    def test_encode_target(self):
        """Test target encoding."""
        # Load data
        df_raw = self.loader.load_iris_data(save_raw=False)
        df_processed = self.loader.preprocess_data(df_raw)
        X_train, X_val, X_test, y_train, y_val, y_test = self.loader.split_data(df_processed)
        
        # Encode targets
        y_train_encoded, y_val_encoded, y_test_encoded = self.loader.encode_target(y_train, y_val, y_test)
        
        # Check that encoded targets are numeric
        self.assertTrue(np.issubdtype(y_train_encoded.dtype, np.number))
        self.assertTrue(np.issubdtype(y_val_encoded.dtype, np.number))
        self.assertTrue(np.issubdtype(y_test_encoded.dtype, np.number))
        
        # Check that all unique values are present
        unique_encoded = set(y_train_encoded.unique()) | set(y_val_encoded.unique()) | set(y_test_encoded.unique())
        self.assertEqual(len(unique_encoded), 3)  # 3 classes
    
    def test_get_data_summary(self):
        """Test data summary generation."""
        # Load data
        df_raw = self.loader.load_iris_data(save_raw=False)
        
        # Get summary
        summary = self.loader.get_data_summary(df_raw)
        
        # Check summary structure
        required_keys = ['shape', 'columns', 'dtypes', 'missing_values', 'duplicates']
        for key in required_keys:
            self.assertIn(key, summary)
        
        # Check summary values
        self.assertEqual(summary['shape'], (150, 5))
        self.assertEqual(summary['duplicates'], 0)
        self.assertEqual(sum(summary['missing_values'].values()), 0)


if __name__ == '__main__':
    unittest.main() 