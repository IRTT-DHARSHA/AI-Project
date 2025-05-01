import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score



data = pd.read_csv("data.csv") 
print("Columns in the dataset:")
print(data.columns)
print("\nFirst few rows of the dataset:")
print(data.head()) 
target_column = "Yield" 

if target_column not in data.columns:
    print(f"Error: Column '{target_column}' not found in the dataset.")

   
    data['Yield'] = data['Nitrogen'] * data['Rainfall'] * 0.5  
    print("\nFirst few rows of the dataset after calculating 'Yield':")
    print(data[['Nitrogen', 'Rainfall', 'Yield']].head()) 
    if target_column not in data.columns:
        import sys
        sys.exit(f"Error: Column '{target_column}' still not found in the dataset after calculation or reloading. Exiting.")

X = data.drop([target_column], axis=1)  
y = data[target_column] 
print(f"Target column '{target_column}' found. Proceeding with model training.")


X = pd.get_dummies(X, columns=['Crop', 'Soil_Type', 'Variety'], drop_first=True) 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Linear Regression
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
lr_predictions = lr_model.predict(X_test)

# Random Forest
rf_model = RandomForestRegressor(random_state=42)
rf_model.fit(X_train, y_train)
rf_predictions = rf_model.predict(X_test)

# Evaluate models
def evaluate_model(predictions, model_name):
    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, predictions)
    print(f"{model_name} Model Evaluation:")
    print(f"  MAE: {mae:.2f}")
    print(f"  MSE: {mse:.2f}")
    print(f"  RMSE: {rmse:.2f}")
    print(f"  R-squared: {r2:.2f}")
    print("-" * 30)

evaluate_model(lr_predictions, "Linear Regression")
evaluate_model(rf_predictions, "Random Forest")