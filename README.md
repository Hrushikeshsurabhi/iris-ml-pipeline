# Iris ML Pipeline

A comprehensive machine learning pipeline for the Iris dataset with modular components for data processing, model training, evaluation, and API deployment.

## 📊 Project Overview

This comprehensive machine learning project analyzes the famous Iris dataset using various ML techniques and concepts. The project demonstrates data exploration, preprocessing, model training, evaluation, and deployment through a modular, production-ready architecture.

## 🎯 Project Goals

- Perform comprehensive exploratory data analysis (EDA)
- Implement multiple machine learning algorithms
- Compare model performance using various metrics
- Demonstrate feature engineering and selection
- Showcase model deployment and API creation
- Provide interactive visualizations and dashboards
- Implement clustering analysis
- Create a modular, maintainable codebase

## 📁 Project Structure
```


iris-ml-pipeline/
├── data/                   # Dataset files
│   └── iris.csv           # Iris dataset
├── notebooks/              # Jupyter notebooks for analysis
│   ├── 01_data_exploration.ipynb
│   ├── 02_model_training.ipynb
│   ├── 03_clustering_analysis.ipynb
│   └── 04_api_demo.ipynb
├── src/                    # Source code
│   ├── data/              # Data processing modules
│   │   ├── __init__.py
│   │   ├── data_loader.py
│   │   └── preprocessor.py
│   ├── models/            # ML model implementations
│   │   ├── __init__.py
│   │   ├── base_model.py
│   │   ├── classifier.py
│   │   └── clustering.py
│   ├── evaluation/        # Model evaluation metrics
│   │   ├── __init__.py
│   │   ├── metrics.py
│   │   └── visualizer.py
│   ├── api/               # API implementation
│   │   ├── __init__.py
│   │   ├── app.py
│   │   └── models.py
│   └── utils/             # Utility functions
│       ├── __init__.py
│       └── helpers.py
├── models/                 # Trained model files
├── reports/                # Generated reports and visualizations
├── tests/                  # Unit tests
│   ├── __init__.py
│   ├── test_data_loader.py
│   ├── test_classifier.py
│   ├── test_clustering.py
│   └── test_api.py
├── requirements.txt        # Python dependencies
├── main.py                 # Main pipeline execution
├── config.yaml            # Configuration file
└── README.md              # This file
```

## 🚀 Quick Start

1. **Clone and Setup**
   ```bash
   git clone git@github.com:Hrushikeshsurabhi/iris-ml-pipeline.git
   cd iris-ml-pipeline
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Run Complete Pipeline**
   ```bash
   python main.py
   ```

3. **Start API Server**
   ```bash
   python src/api/app.py
   ```

4. **Open Jupyter Notebooks**
   ```bash
   jupyter notebook notebooks/
   ```

5. **Run Tests**
   ```bash
   pytest tests/
   ```

## 📈 Features

### Data Analysis
- Automated data loading and preprocessing
- Exploratory Data Analysis (EDA)
- Statistical analysis and insights
- Data visualization with multiple chart types
- Correlation analysis
- Feature distribution analysis

### Machine Learning Models
- **Supervised Learning:**
  - Logistic Regression
  - Support Vector Machine (SVM)
  - Random Forest
  - Decision Tree
  - K-Nearest Neighbors (KNN)
  - Naive Bayes
  - Neural Network (MLP)

- **Unsupervised Learning:**
  - K-Means Clustering
  - Hierarchical Clustering
  - Principal Component Analysis (PCA)

### Model Evaluation
- Cross-validation
- Confusion matrix
- Classification report
- ROC curves
- Precision-Recall curves
- Model comparison metrics
- Clustering evaluation metrics

### Advanced Concepts
- Feature engineering
- Hyperparameter tuning
- Model ensemble methods
- Bias-variance analysis
- Overfitting detection and prevention
- Clustering analysis and visualization

### API & Deployment
- RESTful API with FastAPI
- Model prediction endpoints
- Interactive API documentation
- Health check endpoints
- Input validation and error handling

## 🛠️ Technologies Used

- **Python 3.8+**
- **Data Science:** pandas, numpy, scikit-learn
- **Visualization:** matplotlib, seaborn, plotly
- **Deep Learning:** tensorflow, keras
- **API:** fastapi, uvicorn
- **Testing:** pytest
- **Notebooks:** jupyter
- **Configuration:** pyyaml

## 📊 Dataset Information

The Iris dataset contains 150 samples of iris flowers with 4 features:
- Sepal length (cm)
- Sepal width (cm)
- Petal length (cm)
- Petal width (cm)

**Target Classes:**
- Setosa
- Versicolor
- Virginica

## 🎯 Key Insights

- Perfect linear separability for Setosa class
- Versicolor and Virginica show some overlap
- Petal measurements are more discriminative than sepal measurements
- Dataset is well-balanced with 50 samples per class
- K-means clustering effectively identifies 3 distinct groups

## 🔧 Configuration

The project uses a `config.yaml` file for configuration management:
- Model parameters
- Data paths
- API settings
- Visualization preferences

## 🧪 Testing

Comprehensive unit tests are included for:
- Data loading and preprocessing
- Model training and prediction
- Clustering algorithms
- API endpoints
- Utility functions

Run tests with:
```bash
pytest tests/
```

## 📝 License

This project is open source and available under the MIT License.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📞 Contact

For questions or support, please open an issue on GitHub.

---

**Repository:** https://github.com/Hrushikeshsurabhi/iris-ml-pipeline 
