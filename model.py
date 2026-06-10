"""
Machine learning model helpers for traffic prediction.

The module provides time and lag feature engineering transformers plus small
helpers for training, prediction, and evaluation.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class TimeFeatureEngineer(BaseEstimator, TransformerMixin):
    """Create time-based features from a ``timestamp`` column."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        """
        Transform input data by adding time-based features.

        Args:
            X (pd.DataFrame): Input data with a ``timestamp`` column.

        Returns:
            pd.DataFrame: Data with additional time features.
        """
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
    """Create lag features for time series data."""

    def __init__(self, lag_columns=None, lag_hours=None):
        """
        Initialize the LagFeatureEngineer.

        Args:
            lag_columns (list): Columns to create lag features for. Defaults to
                ``["congestion_score", "travel_time_mins"]``.
            lag_hours (list): List of hour offsets to lag. Defaults to
                ``[1, 24]``.
        """
        self.lag_columns = lag_columns
        self.lag_hours = lag_hours

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        """
        Transform input data by adding lag features.

        Args:
            X (pd.DataFrame): Input data sorted by timestamp.

        Returns:
            pd.DataFrame: Data with additional lag features.
        """
        X = X.copy()

        if "timestamp" in X.columns:
            X = X.sort_values("timestamp").reset_index(drop=True)

        lag_columns = self.lag_columns or ["congestion_score", "travel_time_mins"]
        lag_hours = self.lag_hours or [1, 24]

        for col in lag_columns:
            if col in X.columns:
                for lag in lag_hours:
                    X[f"{col}_lag_{lag}"] = X[col].shift(lag)

        return X


class NumericFeatureSelector(BaseEstimator, TransformerMixin):
    """Drop non-feature columns and keep numeric model inputs."""

    def __init__(self, drop_columns=None):
        self.drop_columns = drop_columns or []

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X = X.drop(columns=[col for col in self.drop_columns if col in X], errors="ignore")
        return X.select_dtypes(include=[np.number])


def _create_regressor(model_type="random_forest"):
    if model_type == "random_forest":
        return RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        )
    if model_type == "gradient_boosting":
        from sklearn.ensemble import GradientBoostingRegressor

        return GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42,
        )
    raise ValueError(f"Unsupported model type: {model_type}")


def create_feature_engineering_pipeline():
    """
    Create a pipeline for feature engineering only.

    Returns:
        sklearn.pipeline.Pipeline: Time and lag feature engineering pipeline.
    """
    return Pipeline(
        steps=[
            ("time_features", TimeFeatureEngineer()),
            ("lag_features", LagFeatureEngineer()),
        ]
    )


def create_model_pipeline_from_features(model_type="random_forest"):
    """
    Create a machine learning pipeline for already featured numeric data.

    Args:
        model_type (str): Type of model to use (``random_forest`` or
            ``gradient_boosting``).

    Returns:
        sklearn.pipeline.Pipeline: Configured scaler + model pipeline.
    """
    return Pipeline(
        steps=[
            ("feature_selector", NumericFeatureSelector()),
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("regressor", _create_regressor(model_type)),
        ]
    )


def create_model_pipeline(model_type="random_forest", target_column="congestion_score"):
    """
    Create a machine learning pipeline for raw traffic data.

    The pipeline includes time and lag feature engineering, drops common
    non-feature columns, then scales features and trains the requested regressor.
    """
    return Pipeline(
        steps=[
            ("time_features", TimeFeatureEngineer()),
            ("lag_features", LagFeatureEngineer()),
            (
                "feature_selector",
                NumericFeatureSelector(["timestamp", "location_id", target_column]),
            ),
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("regressor", _create_regressor(model_type)),
        ]
    )


def train_model(pipeline, X_train, y_train):
    """
    Train the machine learning pipeline.

    Returns:
        sklearn.pipeline.Pipeline: Trained pipeline.
    """
    pipeline.fit(X_train, y_train)
    return pipeline


def predict_model(pipeline, X):
    """
    Make predictions using a trained pipeline.

    Returns:
        np.ndarray: Predicted values.
    """
    return pipeline.predict(X)


def evaluate_predictions(y_true, y_pred):
    """
    Evaluate predictions using MAE and RMSE.

    Returns:
        dict: Dictionary containing ``mae`` and ``rmse``.
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return {"mae": mae, "rmse": rmse}
