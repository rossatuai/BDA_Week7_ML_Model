import os
import pickle
import mlflow
import pandas as pd

from pandas import read_csv
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer


# Set MLflow tracking server
mlflow.set_tracking_uri("http://localhost:5555")
print("Tracking URI:", mlflow.get_tracking_uri())

# Set experiment
mlflow.set_experiment("salary_prediction_v5")

# Set workspace (GitHub Actions or local fallback)
workspace = os.getenv(
    "GITHUB_WORKSPACE",
    "/opt/mlflow/BDA_Week7_ML_Model"
)

# Define directory paths
model_cleaning_dir = os.path.join(workspace, "ModelCleaning")
csv_file_path = os.path.join(model_cleaning_dir, "cleaned_data.csv")

# Check if file exists
if os.path.exists(csv_file_path):
    print(f"File found: {csv_file_path}")
else:
    print(f"File not found at: {csv_file_path}")
    exit()

# Load cleaned dataset
df = read_csv(csv_file_path)

# Preview data
print(df.head())

# Features and target
X = df["Age"].values.reshape(-1, 1)
y = df["Salary"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Handle missing values
imputer = SimpleImputer(strategy="mean")
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)

# Start MLflow run
with mlflow.start_run():

    # Train model
    mind = LinearRegression()
    mind.fit(X_train, y_train)

    # Predict
    predictions = mind.predict(X_test)

    # Evaluate
    r2 = r2_score(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)

    # Log metrics
    mlflow.log_metric("r2_score", r2)
    mlflow.log_metric("mse", mse)

    # Save model locally
    result = mlflow.sklearn.log_model(
        sk_model=mind,
        name="model",
        input_example=X_train[:5]
    )

    mlflow.register_model(
        model_uri=result.model_uri,
        name="salary_model"
    )



    print(f"Model logged successfully with r2: {r2}")
    print(f"MSE: {mse}")
