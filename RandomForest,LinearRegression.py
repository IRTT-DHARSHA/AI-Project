!pip install scikit-learn pandas matplotlib seaborn

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Generate sample data (replace with your actual data loading)
from sklearn.datasets import make_regression
data = pd.DataFrame(make_regression(n_samples=20000, n_features=10, n_informative=5, random_state=42)[0])
data.columns = ['Nitrogen', 'Phosphorous', 'Potassium', 'Temperature', 'Humidity', 'pH', 'Rainfall', 'Sunlight', 'Wind', 'Soil Type']
data['Yield'] = make_regression(n_samples=20000, n_features=10, n_informative=5, random_state=42)[1]

def evaluate_model(predictions, model_name, model, X_used_for_training, y_test_data=None):
    """
    Evaluates a model's performance and optionally displays feature importance.

    Args:
        predictions: The model's predictions.
        model_name: The name of the model (e.g., "Linear Regression", "Random Forest").
        model: The trained model object.
        X_used_for_training: The features used for training the model.
        y_test_data (optional): The true target values for the test data. 
                                If not provided, it defaults to the global `y_test`.
    """
    # If y_test_data is not provided, use the global y_test
    global y_test  # Declare y_test as global to access it within the function
    if y_test_data is None:
        y_test_data = y_test

    mae = mean_absolute_error(y_test_data, predictions)
    mse = mean_squared_error(y_test_data, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test_data, predictions)
    print(f"{model_name} Model Evaluation:")
    print(f"  MAE: {mae:.2f}")
    print(f"  MSE: {mse:.2f}")
    print(f"  RMSE: {rmse:.2f}")
    print(f"  R-squared: {r2:.2f}")
    print("-" * 30)

    # Feature Importance (for models with feature_importances_ attribute)
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        features = X_used_for_training.columns
        # Sort feature importances in descending order
        indices = np.argsort(importances)[::-1]

        # Ensure indices are within the bounds of features
        num_features = len(features)  # Get the number of features in X_used_for_training
        importances = importances[:num_features]  # Limit importances to the number of features
        indices = indices[:num_features]  # Limit indices to the number of features

        # Plot feature importances
        plt.figure(figsize=(10, 6))
        plt.title(f'{model_name} - Feature Importance')
        plt.bar(range(len(importances)), importances, align='center')
        plt.xticks(range(len(indices)), features[indices], rotation=90)
        plt.tight_layout()
        plt.show()

# Data preprocessing and model training
X_original = data.drop('Yield', axis=1)
X_rf = X_original.copy()
y = data['Yield']
X_train, X_test, y_train, y_test = train_test_split(X_original, y, test_size=0.2, random_state=42)

# Initialize and train models
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

rf_model = RandomForestRegressor(random_state=42)
rf_model.fit(X_train, y_train)

# Make predictions
lr_predictions = lr_model.predict(X_test)
rf_predictions = rf_model.predict(X_test)

# Evaluate models
evaluate_model(lr_predictions, "Linear Regression", lr_model, X_train, y_test)
evaluate_model(rf_predictions, "Random Forest", rf_model, X_rf, y_test)