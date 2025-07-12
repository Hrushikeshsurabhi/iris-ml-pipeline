"""
Flask API for Iris Dataset ML Model

This module provides a REST API for making predictions using the trained Iris models.
"""

import sys
import os
import logging
import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
import yaml
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

# Import project modules
from src.data.data_loader import IrisDataLoader

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Global variables
models = {}
scaler = None
label_encoder = None
config = None


def load_config():
    """Load configuration from YAML file."""
    global config
    try:
        with open('config.yaml', 'r') as file:
            config = yaml.safe_load(file)
        logger.info("Configuration loaded successfully")
    except FileNotFoundError:
        logger.error("Config file not found!")
        config = {}


def load_models():
    """Load trained models and preprocessing objects."""
    global models, scaler, label_encoder
    
    try:
        models_path = config.get('models', {}).get('save_path', 'models/')
        
        # Load scaler
        scaler_path = os.path.join(models_path, 'scaler.pkl')
        if os.path.exists(scaler_path):
            scaler = joblib.load(scaler_path)
            logger.info("Scaler loaded successfully")
        
        # Load label encoder
        encoder_path = os.path.join(models_path, 'label_encoder.pkl')
        if os.path.exists(encoder_path):
            label_encoder = joblib.load(encoder_path)
            logger.info("Label encoder loaded successfully")
        
        # Load trained models
        model_names = config.get('models', {}).get('model_names', [])
        for model_name in model_names:
            model_path = os.path.join(models_path, f'{model_name}.pkl')
            if os.path.exists(model_path):
                models[model_name] = joblib.load(model_path)
                logger.info(f"Model {model_name} loaded successfully")
        
        # Load ensemble model if available
        ensemble_path = os.path.join(models_path, 'ensemble.pkl')
        if os.path.exists(ensemble_path):
            models['ensemble'] = joblib.load(ensemble_path)
            logger.info("Ensemble model loaded successfully")
        
        logger.info(f"Loaded {len(models)} models")
        
    except Exception as e:
        logger.error(f"Error loading models: {str(e)}")
        raise


def preprocess_input(data):
    """
    Preprocess input data for prediction.
    
    Args:
        data: Input data as dictionary or list
        
    Returns:
        Preprocessed DataFrame
    """
    try:
        # Convert input to DataFrame
        if isinstance(data, dict):
            df = pd.DataFrame([data])
        elif isinstance(data, list):
            df = pd.DataFrame(data)
        else:
            raise ValueError("Input must be a dictionary or list of dictionaries")
        
        # Check required features
        required_features = ['sepal length (cm)', 'sepal width (cm)', 
                           'petal length (cm)', 'petal width (cm)']
        
        missing_features = [f for f in required_features if f not in df.columns]
        if missing_features:
            raise ValueError(f"Missing required features: {missing_features}")
        
        # Create engineered features (same as in data_loader)
        df_processed = df.copy()
        
        # Create ratio features
        df_processed['petal_to_sepal_length_ratio'] = df_processed['petal length (cm)'] / df_processed['sepal length (cm)']
        df_processed['petal_to_sepal_width_ratio'] = df_processed['petal width (cm)'] / df_processed['sepal width (cm)']
        df_processed['sepal_length_to_width_ratio'] = df_processed['sepal length (cm)'] / df_processed['sepal width (cm)']
        df_processed['petal_length_to_width_ratio'] = df_processed['petal length (cm)'] / df_processed['petal width (cm)']
        df_processed['sepal_area'] = df_processed['sepal length (cm)'] * df_processed['sepal width (cm)']
        df_processed['petal_area'] = df_processed['petal length (cm)'] * df_processed['petal width (cm)']
        
        # Create polynomial features
        numeric_cols = df_processed.select_dtypes(include=[np.number]).columns.tolist()
        for col in numeric_cols:
            df_processed[f'{col}_squared'] = df_processed[col] ** 2
        
        for i, col1 in enumerate(numeric_cols):
            for col2 in numeric_cols[i+1:]:
                df_processed[f'{col1}_{col2}_interaction'] = df_processed[col1] * df_processed[col2]
        
        # Scale features
        if scaler is not None:
            df_scaled = pd.DataFrame(
                scaler.transform(df_processed),
                columns=df_processed.columns,
                index=df_processed.index
            )
            return df_scaled
        else:
            return df_processed
            
    except Exception as e:
        logger.error(f"Error preprocessing input: {str(e)}")
        raise


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'models_loaded': len(models),
        'available_models': list(models.keys())
    })


@app.route('/predict', methods=['POST'])
def predict():
    """
    Make prediction using the best model.
    
    Expected input format:
    {
        "sepal length (cm)": 5.1,
        "sepal width (cm)": 3.5,
        "petal length (cm)": 1.4,
        "petal width (cm)": 0.2
    }
    """
    try:
        # Get input data
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Preprocess input
        X_processed = preprocess_input(data)
        
        # Find best model (highest accuracy)
        best_model_name = None
        best_accuracy = -1
        
        # For now, use the first available model or ensemble
        if 'ensemble' in models:
            best_model_name = 'ensemble'
        elif models:
            best_model_name = list(models.keys())[0]
        else:
            return jsonify({'error': 'No models available'}), 500
        
        # Make prediction
        model = models[best_model_name]
        prediction = model.predict(X_processed)[0]
        probabilities = model.predict_proba(X_processed)[0] if hasattr(model, 'predict_proba') else None
        
        # Decode prediction
        if label_encoder is not None:
            predicted_class = label_encoder.inverse_transform([prediction])[0]
        else:
            predicted_class = str(prediction)
        
        # Prepare response
        response = {
            'prediction': predicted_class,
            'model_used': best_model_name,
            'confidence': float(max(probabilities)) if probabilities is not None else None,
            'probabilities': {
                label_encoder.inverse_transform([i])[0]: float(prob)
                for i, prob in enumerate(probabilities)
            } if probabilities is not None and label_encoder is not None else None,
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/predict/<model_name>', methods=['POST'])
def predict_with_model(model_name):
    """
    Make prediction using a specific model.
    
    Args:
        model_name: Name of the model to use
    """
    try:
        # Check if model exists
        if model_name not in models:
            return jsonify({'error': f'Model {model_name} not found'}), 404
        
        # Get input data
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Preprocess input
        X_processed = preprocess_input(data)
        
        # Make prediction
        model = models[model_name]
        prediction = model.predict(X_processed)[0]
        probabilities = model.predict_proba(X_processed)[0] if hasattr(model, 'predict_proba') else None
        
        # Decode prediction
        if label_encoder is not None:
            predicted_class = label_encoder.inverse_transform([prediction])[0]
        else:
            predicted_class = str(prediction)
        
        # Prepare response
        response = {
            'prediction': predicted_class,
            'model_used': model_name,
            'confidence': float(max(probabilities)) if probabilities is not None else None,
            'probabilities': {
                label_encoder.inverse_transform([i])[0]: float(prob)
                for i, prob in enumerate(probabilities)
            } if probabilities is not None and label_encoder is not None else None,
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Prediction error with {model_name}: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/models', methods=['GET'])
def list_models():
    """List all available models."""
    return jsonify({
        'available_models': list(models.keys()),
        'total_models': len(models),
        'timestamp': datetime.now().isoformat()
    })


@app.route('/models/<model_name>', methods=['GET'])
def model_info(model_name):
    """
    Get information about a specific model.
    
    Args:
        model_name: Name of the model
    """
    try:
        if model_name not in models:
            return jsonify({'error': f'Model {model_name} not found'}), 404
        
        model = models[model_name]
        
        info = {
            'name': model_name,
            'type': type(model).__name__,
            'parameters': model.get_params() if hasattr(model, 'get_params') else None,
            'feature_importance': None,
            'timestamp': datetime.now().isoformat()
        }
        
        # Add feature importance if available
        if hasattr(model, 'feature_importances_'):
            info['feature_importance'] = model.feature_importances_.tolist()
        elif hasattr(model, 'coef_'):
            info['feature_importance'] = np.abs(model.coef_[0]).tolist()
        
        return jsonify(info)
        
    except Exception as e:
        logger.error(f"Error getting model info for {model_name}: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    """
    Make predictions for multiple samples.
    
    Expected input format:
    [
        {
            "sepal length (cm)": 5.1,
            "sepal width (cm)": 3.5,
            "petal length (cm)": 1.4,
            "petal width (cm)": 0.2
        },
        {
            "sepal length (cm)": 6.3,
            "sepal width (cm)": 3.3,
            "petal length (cm)": 4.7,
            "petal width (cm)": 1.6
        }
    ]
    """
    try:
        # Get input data
        data = request.get_json()
        
        if not data or not isinstance(data, list):
            return jsonify({'error': 'Input must be a list of samples'}), 400
        
        # Preprocess input
        X_processed = preprocess_input(data)
        
        # Find best model
        if 'ensemble' in models:
            best_model_name = 'ensemble'
        elif models:
            best_model_name = list(models.keys())[0]
        else:
            return jsonify({'error': 'No models available'}), 500
        
        # Make predictions
        model = models[best_model_name]
        predictions = model.predict(X_processed)
        probabilities = model.predict_proba(X_processed) if hasattr(model, 'predict_proba') else None
        
        # Decode predictions
        if label_encoder is not None:
            predicted_classes = label_encoder.inverse_transform(predictions)
        else:
            predicted_classes = [str(p) for p in predictions]
        
        # Prepare response
        results = []
        for i, (pred_class, pred_prob) in enumerate(zip(predicted_classes, predictions)):
            result = {
                'sample_id': i,
                'prediction': pred_class,
                'confidence': float(max(probabilities[i])) if probabilities is not None else None,
                'probabilities': {
                    label_encoder.inverse_transform([j])[0]: float(prob)
                    for j, prob in enumerate(probabilities[i])
                } if probabilities is not None and label_encoder is not None else None
            }
            results.append(result)
        
        response = {
            'predictions': results,
            'model_used': best_model_name,
            'total_samples': len(data),
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/features', methods=['GET'])
def get_features():
    """Get information about required features."""
    return jsonify({
        'required_features': [
            'sepal length (cm)',
            'sepal width (cm)',
            'petal length (cm)',
            'petal width (cm)'
        ],
        'feature_descriptions': {
            'sepal length (cm)': 'Length of the sepal in centimeters',
            'sepal width (cm)': 'Width of the sepal in centimeters',
            'petal length (cm)': 'Length of the petal in centimeters',
            'petal width (cm)': 'Width of the petal in centimeters'
        },
        'target_classes': label_encoder.classes_.tolist() if label_encoder is not None else ['setosa', 'versicolor', 'virginica'],
        'timestamp': datetime.now().isoformat()
    })


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    return jsonify({'error': 'Endpoint not found'}), 404


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors."""
    return jsonify({'error': 'Internal server error'}), 500


def main():
    """Main function to run the Flask API."""
    try:
        # Load configuration and models
        load_config()
        load_models()
        
        # Get API configuration
        api_config = config.get('api', {})
        host = api_config.get('host', '0.0.0.0')
        port = api_config.get('port', 8000)
        debug = api_config.get('debug', False)
        
        logger.info(f"Starting Iris ML API on {host}:{port}")
        logger.info(f"Available models: {list(models.keys())}")
        
        # Run the app
        app.run(host=host, port=port, debug=debug)
        
    except Exception as e:
        logger.error(f"Failed to start API: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main() 