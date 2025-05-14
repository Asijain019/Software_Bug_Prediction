# Software Bug Prediction Project

## Description
This project is a machine learning-based solution for predicting software bugs in Java projects. Using various code metrics extracted from real-world software repositories (Eclipse JDT Core, Eclipse PDE UI, Equinox Framework, Lucene, and Mylyn), the system builds and evaluates multiple classification models to predict whether a software component will contain bugs and their potential severity.

The project implements both binary classification (bug/no bug) and multi-class classification (0 bugs, 1 bug, 2+ bugs) approaches using a variety of machine learning algorithms, feature selection techniques, and performance evaluation metrics.

## Table of Contents
- [Installation](#installation)
- [Dataset](#dataset)
- [Features](#features)
- [Methodology](#methodology)
- [Models](#models)
- [Results](#results)
- [Usage](#usage)
- [Requirements](#requirements)
- [License](#license)

## Installation
```bash
# Clone the repository
git clone https://github.com/yourusername/software-bug-prediction.git
cd software-bug-prediction

# Install required packages
pip install -r requirements.txt
```

## Dataset
The project uses data from five Java projects:
- Eclipse JDT Core
- Eclipse PDE UI
- Equinox Framework
- Lucene
- Mylyn

Each project dataset includes various software metrics and the corresponding bug information.

## Features
The project uses the following code metrics as predictive features:
- **rfc**: Response for a Class
- **cbo**: Coupling Between Objects
- **fanOut**: Number of outgoing connections
- **wmc**: Weighted Methods per Class
- **numberOfLinesOfCode**: Lines of code in the component
- **lcom**: Lack of Cohesion of Methods
- **numberOfAttributes**: Number of attributes in the class
- **numberOfAttributesInherited**: Number of inherited attributes
- And several other object-oriented metrics

Feature selection was performed using Lasso regression, which identified the following key features:
- rfc
- cbo
- wmc
- numberOfLinesOfCode
- lcom
- numberOfAttributes
- numberOfAttributesInherited

## Methodology
1. **Data Preprocessing**:
   - Data cleaning and handling missing values
   - Feature scaling and normalization
   - Label clipping for multi-class classification (0, 1, 2+ bugs)

2. **Exploratory Data Analysis**:
   - Class distribution analysis
   - Feature correlation analysis
   - Feature importance analysis with Lasso regression

3. **Model Development**:
   - Split into training, validation, and test sets
   - Implementation of various classification models
   - Hyperparameter tuning

4. **Evaluation**:
   - Accuracy, F1 score, and AUC metrics
   - Confusion matrices
   - ROC curve analysis
   - Comparative analysis of different models

## Models
The project implements and compares the following machine learning models:

**Binary Classification (bug/no bug)**:
- Neural Networks (MLPClassifier)
- AdaBoost Classifier
- Support Vector Machines (SVM)
- Bagging Classifier
- Dummy Classifier (baseline)

**Multi-class Classification (0, 1, 2+ bugs)**:
- Random Forest Classifier
- K-Nearest Neighbors
- Bagging Classifier
- Dummy Classifier (baseline)
- K-Means Clustering (unsupervised approach)

## Results
Detailed model performance results are available in the code output, including:
- Accuracy scores
- F1 scores
- AUC-ROC scores
- Confusion matrices
- ROC curves

Feature selection improved model performance in many cases, with the best performing models achieving significantly better results than the baseline.

## Usage
```python
# Run the main script
python softwarebug.py

# To use a specific model for prediction
from sklearn.ensemble import BaggingClassifier
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

# Load your data
data = pd.read_csv('your_software_metrics.csv')

# Preprocess the data
X = data[['rfc', 'cbo', 'wmc', 'numberOfLinesOfCode', 'lcom', 'numberOfAttributes', 'numberOfAttributesInherited']]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Load the trained model (assuming it was saved)
# model = joblib.load('bagging_model.pkl')

# Or train a new model
model = BaggingClassifier()
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_scaled)
```

## Requirements
- Python 3.7+
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn

Detailed requirements can be found in the `requirements.txt` file.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
