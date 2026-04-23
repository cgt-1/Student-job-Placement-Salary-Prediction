import numpy as np
import mlflow
import mlflow.sklearn
import pickle

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score

from src.pipeline_builder import (
    build_classification_pipeline,
    build_regression_pipeline
)


def train_models(df):

    # Targets
    y_class = df['placement_status']
    y_reg = df['salary_lpa']

    # Features
    X = df.drop(columns=[
        'Student_ID',
        'placement_status',
        'salary_lpa',
        'placement_status_encoded',
        'placement_encoded'
    ], errors='ignore')

    # Split
    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
        X, y_class, test_size=0.2, random_state=42, stratify=y_class
    )

    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
        X, y_reg, test_size=0.2, random_state=42
    )

    # Build pipelines
    clf_pipeline = build_classification_pipeline(X)
    reg_pipeline = build_regression_pipeline(X)

    # 🔥 MLflow
    with mlflow.start_run():

        # ---- CLASSIFICATION ----
        clf_pipeline.fit(X_train_c, y_train_c)
        y_pred_c = clf_pipeline.predict(X_test_c)
        acc = accuracy_score(y_test_c, y_pred_c)

        # ---- REGRESSION ----
        reg_pipeline.fit(X_train_r, y_train_r)
        y_pred_r = reg_pipeline.predict(X_test_r)
        rmse = np.sqrt(mean_squared_error(y_test_r, y_pred_r))
        r2 = r2_score(y_test_r, y_pred_r)

        # Print
        print("\nClassification Accuracy:", acc)
        print("Regression RMSE:", rmse)
        print("Regression R2:", r2)

        # Log
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)

        # Save models
        with open("models/classifier.pkl", "wb") as f:
            pickle.dump(clf_pipeline, f)

        with open("models/regressor.pkl", "wb") as f:
            pickle.dump(reg_pipeline, f)

        mlflow.sklearn.log_model(clf_pipeline, "classifier_model")
        mlflow.sklearn.log_model(reg_pipeline, "regressor_model")