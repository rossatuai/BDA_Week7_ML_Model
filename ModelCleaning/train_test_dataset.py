import os
import mlflow
from pandas import read_csv
from joblib import dump
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


mlflow.set_tracking_uri("http://localhost:5555")



workspace = os.getenv('GITHUB_WORKSPACE')

# Define the directory where your Python script is located (ModelCleaning)
model_cleaning_dir = os.path.join(workspace, 'ModelCleaning')

# Define the full path to the cleaned data CSV file
csv_file_path = os.path.join(model_cleaning_dir, 'cleaned_data.csv')

# Check if the file exists (for debugging)
if os.path.exists(csv_file_path):
    print(f"File found: {csv_file_path}")
else:
    print(f"File not found at: {csv_file_path}")


# Read the cleaned data CSV file
df = read_csv(csv_file_path)

# Proceed with your training logic
print(df.head()) 

X= df["Age"].values.reshape(10,1)
y= df["Salary"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


with mlflow.start_run():


    mind = LinearRegression()
    mind.fit(X_train, y_train)


    # Evaluate
    predictions = mind.predict(X_test)
    r2 = r2_score(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)

    # Log Metrics
    mlflow.log_metric("r2_score", r2)
    mlflow.log_metric("mse", mse)

    # Log Model & Register
    result = mlflow.sklearn.log_model(
        sk_model=mind,
        artifact_path="model"
    )

    mlflow.register_model(
        model_uri=result.model_uri,
        name="my_linear_model"
    )

    print(f"Model logged with r2: {r2}")

with open("model.pkl", "wb") as file:
    pickle.dump(mind, file)


dump(mind, "AgeSalaryModel.pkl")
