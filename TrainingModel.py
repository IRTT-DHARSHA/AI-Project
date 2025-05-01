import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import shap
from xgboost import XGBRegressor
from flask import Flask, request, jsonify

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

