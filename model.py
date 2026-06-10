"""Machine learning pipeline utilities for traffic prediction."""

import warnings

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")


class TimeFeatureEngineer(BaseEstimator, TransformerMixin):
    """Create time-based features from a timestamp column."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        if not pd.api.types.is_datetime64_any_dtype(X["timestamp"]):
            X["timestamp"] = pd.to_datetime(X["timestamp"])

        X["hour"] = X["timestamp"].dt.hour
        X["day_of_week"] = X["timestamp"].dt.dayofweek
        X["is_weekend"] = (X["day_of_week"] >= 5).astype(int)
        X["hour_sin"] = np.sin(2 * np.pi * X["hour"] / 24)
        X["hour_cos"] = np.cos(2 * np.pi * X["hour"] / 24)
        X["day_of_week_sin"] = np.sin(2 * np.pi * X["day_of_week"] / 7)
        X["day_of_week_cos"] = np.cos(2 * np.pi * X["day_of_week"] / 7)
        return X


class LagFeatureEngineer(BaseEstimator, TransformerMixin):
    """Create lag features for selected traffic time-series columns."""

    def __init__(self, lag_columns=None, lag_hours=None):
        self.lag_columns = lag_columns or ["congestion_score", "travel_time_mins"]
        self.lag_hours = lag_hours or [1, 24]

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        if "timestamp" in X.columns:
            X = X.sort_values("timestamp").reset_index(drop=True)

        for col in self.lag_columns:
            if col in X.columns:
                for lag in self.lag_hours:
                    X[f"{col}_lag_{lag}"] = X[col].shift(lag)
        return X


def create_feature_engineering_pipeline():
    """Create the feature-engineering-only pipeline."""
    return Pipeline(
        steps=[
            ("time_features", TimeFeatureEngineer()),
            ("lag_features", LagFeatureEngineer()),
        ]
    )


def create_model_pipeline_from_features(model_type="random_forest"):
    """Create a scaler/model pipeline for data that already has features."""
    if model_type == "random_forest":
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        )
    elif model_type == "gradient_boosting":
        from sklearn.ensemble import GradientBoostingRegressor

        model = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42,
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("regressor", model),
        ]
    )


def create_model_pipeline(model_type="random_forest"):
    """Create a complete feature engineering and prediction pipeline."""
    pipeline_steps = [
        ("time_features", TimeFeatureEngineer()),
        ("lag_features", LagFeatureEngineer()),
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("regressor", create_model_pipeline_from_features(model_type)["regressor"]),
    ]
    return Pipeline(steps=pipeline_steps)


def train_model(pipeline, X_train, y_train):
    """Train and return a prediction pipeline."""
    pipeline.fit(X_train, y_train)
    return pipeline


def predict_model(pipeline, X):
    """Generate predictions with a trained pipeline."""
    return pipeline.predict(X)


def evaluate_predictions(y_true, y_pred):
    """Return MAE and RMSE metrics for predictions."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return {"mae": mae, "rmse": rmse}
