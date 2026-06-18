import numpy as np
import pandas as pd
import joblib
import os
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


mlflow.set_experiment("GetAround_Pricing_Optimization")
# Phase 1: Loading the dataset

BASE_DIR = os.path.dirname(os.path.abspath(__file__)) 
PROJECT_ROOT = os.path.dirname(os.path.dirname(BASE_DIR)) 

pricing_data_path = os.path.join(PROJECT_ROOT, "api", "src", "get_around_pricing_project.csv")

df_pricing = pd.read_csv(pricing_data_path)

if "Unnamed: 0" in df_pricing.columns:
    df_pricing = df_pricing.drop(columns=["Unnamed: 0"])


# Phase 2: Splitting Target and Features
X = df_pricing.drop(columns=["rental_price_per_day"])
y = df_pricing["rental_price_per_day"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Phase 3: Preprocessing Pipeline

num_features = ["mileage", "engine_power"]
cat_features = [
    "model_key", "fuel", "paint_color", "car_type",
    "private_parking_available", "has_gps", "has_air_conditioning",
    "automatic_car", "has_getaround_connect", "has_speed_regulator", "winter_tires"
]

numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])
categorical_transformer = Pipeline(steps=[('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, num_features),
        ('cat', categorical_transformer, cat_features)
    ]
)

# Phase 4: Training the Baseline Linear Regression Model with MLflow Tracking
with mlflow.start_run(run_name="Baseline_Linear_Regression"):
     
     baseline_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
    ])
     
     baseline_pipeline.fit(X_train, y_train)

     mlflow.log_param("model_type", "LinearRegression")
     mlflow.log_param("test_size", 0.2)

     # Phase 5: Evaluating Model Performance

     y_train_pred = baseline_pipeline.predict(X_train)
     y_test_pred = baseline_pipeline.predict(X_test)

     mae_test = mean_absolute_error(y_test, y_test_pred)
     rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))
     r2_train = r2_score(y_train, y_train_pred)
     r2_test = r2_score(y_test, y_test_pred)

     mlflow.log_metric("r2_train", r2_train)
     mlflow.log_metric("r2_test", r2_test)
     mlflow.log_metric("mae_test", mae_test)
     mlflow.log_metric("rmse_test", rmse_test)
     # Phase 6: Serialization and Export to .joblib
     
     output_model_path = os.path.join(PROJECT_ROOT, "api", "pricing_model.joblib")
     joblib.dump(baseline_pipeline, output_model_path)
    
     mlflow.log_artifact(output_model_path, artifact_path="model")
    
     print(f"🎉 Success ! Experience saved on MLflow et model generated at : '{output_model_path}'")
