# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.cluster import KMeans

# Load and preprocess dataset
data = pd.read_csv("data.csv")  # Replace with your dataset file

# Print the column names and first few rows to identify the target variable column and data structure
print("Columns in the dataset:")
print(data.columns)
print("\nFirst few rows of the dataset:")
print(data.head())  # Check the first few rows to see if 'Yield' is present

# **Replace 'your_target_column_name' with the actual name of your target column from the printed list**
# Ensure it matches the exact column name in the CSV (case-sensitive)
# For example, if the target column is named "yield_amount", change to:
# target_column = "yield_amount"
target_column = "Yield"  # Update if necessary

# Check if the column exists
if target_column not in data.columns:
    # Handle the error - either exit or adjust the column name
    print(f"Error: Column '{target_column}' not found in the dataset.")

    # Option 1: Exit the program if the column is essential
    # import sys
    # sys.exit(f"Error: Column '{target_column}' not found in the dataset.")
    
    # Option 2: Add the target column (example if it needs to be calculated)
    # ----> Update this section with your yield calculation logic <----
    # For example, assuming a simplified calculation based on Nitrogen and Rainfall:
    # This calculation MUST be done before the check for the column existing.
    # Otherwise, it will always fail.
    data['Yield'] = data['Nitrogen'] * data['Rainfall'] * 0.5  # Replace with your actual logic or obtain a new dataset with 'Yield' data
    # If calculating 'Yield', ensure to validate the results for accuracy.
    # Print intermediate values during the calculation to debug any errors.
    print("\nFirst few rows of the dataset after calculating 'Yield':")
    print(data[['Nitrogen', 'Rainfall', 'Yield']].head())  # Check the calculated values
    
    # If you've obtained a new dataset with 'Yield', reload it:
    # data = pd.read_csv("new_dataset_with_yield.csv")  # Replace with actual file name
    # print("\nColumns in the new dataset:")
    # print(data.columns)
    # print("\nFirst few rows of the new dataset:")
    # print(data.head())  # Check the first few rows to see if 'Yield' is present
    
    # Re-check if the column exists after calculation or reloading
    if target_column not in data.columns:
        import sys
        sys.exit(f"Error: Column '{target_column}' still not found in the dataset after calculation or reloading. Exiting.")

# Proceed with model training if 'Yield' column is found
X = data.drop([target_column], axis=1)  # Drop the target column
y = data[target_column]  # Target variable
print(f"Target column '{target_column}' found. Proceeding with model training.")

# Convert categorical features to numerical using one-hot encoding
X = pd.get_dummies(X, columns=['Crop', 'Soil_Type', 'Variety'], drop_first=True) 

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale numerical features using StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Model Training
lr_model = LinearRegression()
rf_model = RandomForestRegressor(random_state=42)
dt_model = DecisionTreeRegressor(random_state=42)  # Initialize Decision Tree

models = [lr_model, rf_model, dt_model]
model_names = ["Linear Regression", "Random Forest", "Decision Tree"]
predictions = []

for model in models:
    model.fit(X_train, y_train)
    predictions.append(model.predict(X_test))

# Evaluate and Plot Results
plt.figure(figsize=(10, 6))
for i, prediction in enumerate(predictions):
    mae = mean_absolute_error(y_test, prediction)
    mse = mean_squared_error(y_test, prediction)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, prediction)
    
    print(f"{model_names[i]} Model Evaluation:")
    print(f"  MAE: {mae:.2f}")
    print(f"  MSE: {mse:.2f}")
    print(f"  RMSE: {rmse:.2f}")
    print(f"  R-squared: {r2:.2f}")
    print("-" * 30)
    
    # Plot predicted vs actual values in the same chart
    plt.scatter(y_test, prediction, label=model_names[i], alpha=0.7)

plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--')
plt.xlabel("Actual Yield")
plt.ylabel("Predicted Yield")
plt.title("Model Comparison - Predicted vs Actual Yield")
plt.legend()
plt.show()

# Residual Plot Comparison
plt.figure(figsize=(10, 6))
for i, prediction in enumerate(predictions):
    residuals = y_test - prediction
    sns.histplot(residuals, kde=True, label=model_names[i])

plt.xlabel("Residuals")
plt.ylabel("Frequency")
plt.title("Model Comparison - Residuals Distribution")
plt.legend()
plt.show()


# 3D Scatter Plot
fig = px.scatter_3d(data, x="Nitrogen", y="Phosphorus", z="Yield",
                    color="Crop", size="Rainfall",
                    title="3D Scatter Plot of Nitrogen, Phosphorus, and Yield")
fig.show()

# K-Means Clustering
# Select features for clustering
features_for_clustering = ['Nitrogen', 'Phosphorus', 'Potassium']
X_clustering = data[features_for_clustering]

# Determine optimal number of clusters (e.g., using elbow method)
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(X_clustering)
    wcss.append(kmeans.inertia_)

plt.plot(range(1, 11), wcss)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

# Apply K-Means with the chosen number of clusters
num_clusters = 3  # Update based on elbow method results
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
data['Cluster'] = kmeans.fit_predict(X_clustering)

# Visualize clusters (2D scatter plot example)
plt.figure(figsize=(8, 6))
plt.scatter(data['Nitrogen'], data['Phosphorus'], c=data['Cluster'], cmap='viridis')
plt.xlabel("Nitrogen")
plt.ylabel("Phosphorus")
plt.title("K-Means Clustering")
plt.show()