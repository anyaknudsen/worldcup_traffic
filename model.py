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


DEFAULT_LAG_COLUMNS = ["congestion_score", "travel_time_mins"]
DEFAULT_LAG_HOURS = [1, 24]
DEFAULT_UNAVAILABLE_AT_PREDICTION_COLUMNS = [
    "congestion_score",
    "travel_time_mins",
    "speed_kph",
    "incident_count",
]


def _coerce_datetime(series):
    if pd.api.types.is_datetime64_any_dtype(series):
        return series

    try:
        converted = pd.to_datetime(series)
    except (TypeError, ValueError):
        return pd.to_datetime(series, utc=True)

    if pd.api.types.is_datetime64_any_dtype(converted):
        return converted
    return pd.to_datetime(series, utc=True)


def sort_time_series(data, timestamp_column="timestamp"):
    """Return data sorted chronologically with a datetimelike timestamp column."""
    data = data.copy()
    data[timestamp_column] = _coerce_datetime(data[timestamp_column])
    return data.sort_values(timestamp_column, kind="mergesort").reset_index(drop=True)


class TimeFeatureEngineer(BaseEstimator, TransformerMixin):
    """Create time-based features from a ``timestamp`` column."""

    def __init__(self, timestamp_column="timestamp"):
        self.timestamp_column = timestamp_column

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

        X[self.timestamp_column] = _coerce_datetime(X[self.timestamp_column])

        X["hour"] = X[self.timestamp_column].dt.hour
        X["day_of_week"] = X[self.timestamp_column].dt.dayofweek
        X["is_weekend"] = (X["day_of_week"] >= 5).astype(int)
        X["hour_sin"] = np.sin(2 * np.pi * X["hour"] / 24)
        X["hour_cos"] = np.cos(2 * np.pi * X["hour"] / 24)
        X["day_of_week_sin"] = np.sin(2 * np.pi * X["day_of_week"] / 7)
        X["day_of_week_cos"] = np.cos(2 * np.pi * X["day_of_week"] / 7)

        return X


class LagFeatureEngineer(BaseEstimator, TransformerMixin):
    """Create lag features for time series data."""

    def __init__(
        self,
        lag_columns=None,
        lag_hours=None,
        timestamp_column="timestamp",
        group_columns=None,
    ):
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
        self.timestamp_column = timestamp_column
        self.group_columns = group_columns

    def fit(self, X, y=None):
        history = self._prepare_timestamps(X)
        self.group_columns_ = self._resolve_group_columns(history)
        self.history_ = history.copy()
        return self

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y)
        return self._add_lags(self._prepare_timestamps(X))

    def transform(self, X):
        """
        Transform input data by adding lag features.

        Args:
            X (pd.DataFrame): Input data sorted by timestamp.

        Returns:
            pd.DataFrame: Data with additional lag features.
        """
        X = self._prepare_timestamps(X)

        if not hasattr(self, "history_"):
            return self._add_lags(X)

        current_for_lags = X.copy()
        original_values = {}
        for col in self._lag_columns():
            if col in current_for_lags.columns:
                original_values[col] = current_for_lags[col].copy()
                current_for_lags[col] = np.nan

        history = self.history_.copy()
        combined = pd.concat([history, current_for_lags], ignore_index=True, sort=False)
        featured = self._add_lags(combined)
        current = featured.iloc[len(history) :].reset_index(drop=True)

        for col, values in original_values.items():
            current[col] = values.reset_index(drop=True)

        return current

    def _prepare_timestamps(self, X):
        X = X.copy()
        if self.timestamp_column in X.columns:
            X[self.timestamp_column] = _coerce_datetime(X[self.timestamp_column])
        return X

    def _lag_columns(self):
        return self.lag_columns or DEFAULT_LAG_COLUMNS

    def _lag_hours(self):
        return self.lag_hours or DEFAULT_LAG_HOURS

    def _resolve_group_columns(self, X):
        if self.group_columns is not None:
            return [col for col in self.group_columns if col in X.columns]
        if "location_id" in X.columns:
            return ["location_id"]
        return []

    def _add_lags(self, X):
        X = X.copy()
        if self.timestamp_column not in X.columns:
            return self._add_lags_in_current_order(X)

        row_order_col = "__lag_row_order__"
        X[row_order_col] = np.arange(len(X))
        group_columns = getattr(self, "group_columns_", self._resolve_group_columns(X))
        sort_columns = [*group_columns, self.timestamp_column]
        sorted_X = X.sort_values(sort_columns, kind="mergesort")

        for col in self._lag_columns():
            if col in sorted_X.columns:
                for lag in self._lag_hours():
                    lag_col = f"{col}_lag_{lag}"
                    if group_columns:
                        sorted_X[lag_col] = sorted_X.groupby(
                            group_columns, sort=False
                        )[col].shift(lag)
                    else:
                        sorted_X[lag_col] = sorted_X[col].shift(lag)

        return (
            sorted_X.sort_values(row_order_col, kind="mergesort")
            .drop(columns=[row_order_col])
            .reset_index(drop=True)
        )

    def _add_lags_in_current_order(self, X):
        for col in self._lag_columns():
            if col in X.columns:
                for lag in self._lag_hours():
                    X[f"{col}_lag_{lag}"] = X[col].shift(lag)
        return X


class NumericFeatureSelector(BaseEstimator, TransformerMixin):
    """Drop non-feature columns and keep numeric model inputs."""

    def __init__(self, drop_columns=None):
        self.drop_columns = drop_columns or []

    def fit(self, X, y=None):
        X_numeric = self._numeric_features(X)
        self.feature_columns_ = list(X_numeric.columns)
        return self

    def transform(self, X):
        X_numeric = self._numeric_features(X)
        if hasattr(self, "feature_columns_"):
            X_numeric = X_numeric.reindex(columns=self.feature_columns_)
        return X_numeric

    def _numeric_features(self, X):
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


def _create_median_imputer():
    try:
        return SimpleImputer(strategy="median", keep_empty_features=True)
    except TypeError:
        return SimpleImputer(strategy="median")


def create_feature_engineering_pipeline(timestamp_column="timestamp"):
    """
    Create a pipeline for feature engineering only.

    Returns:
        sklearn.pipeline.Pipeline: Time and lag feature engineering pipeline.
    """
    return Pipeline(
        steps=[
            ("time_features", TimeFeatureEngineer(timestamp_column=timestamp_column)),
            ("lag_features", LagFeatureEngineer(timestamp_column=timestamp_column)),
        ]
    )


def create_train_test_feature_sets(
    train_data,
    test_data,
    timestamp_column="timestamp",
):
    """Create train/test features without using test-period lag source values."""
    feature_pipeline = create_feature_engineering_pipeline(
        timestamp_column=timestamp_column
    )
    train_featured = feature_pipeline.fit_transform(train_data)
    test_featured = feature_pipeline.transform(test_data)
    return train_featured.reset_index(drop=True), test_featured.reset_index(drop=True)


def select_forecast_features(
    featured_data,
    target_column="congestion_score",
    timestamp_column="timestamp",
    unavailable_columns=None,
):
    """Drop columns that are not known at forecast time."""
    if unavailable_columns is None:
        unavailable_columns = DEFAULT_UNAVAILABLE_AT_PREDICTION_COLUMNS

    drop_columns = [
        timestamp_column,
        "location_id",
        target_column,
        *unavailable_columns,
    ]
    return featured_data.drop(columns=list(dict.fromkeys(drop_columns)), errors="ignore")


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
            ("imputer", _create_median_imputer()),
            ("scaler", StandardScaler()),
            ("regressor", _create_regressor(model_type)),
        ]
    )


def create_model_pipeline(
    model_type="random_forest",
    target_column="congestion_score",
    timestamp_column="timestamp",
):
    """
    Create a machine learning pipeline for raw traffic data.

    The pipeline includes time and lag feature engineering, drops common
    non-feature columns, then scales features and trains the requested regressor.
    """
    return Pipeline(
        steps=[
            ("time_features", TimeFeatureEngineer(timestamp_column=timestamp_column)),
            ("lag_features", LagFeatureEngineer(timestamp_column=timestamp_column)),
            (
                "feature_selector",
                NumericFeatureSelector(
                    [
                        timestamp_column,
                        "location_id",
                        target_column,
                        *DEFAULT_UNAVAILABLE_AT_PREDICTION_COLUMNS,
                    ]
                ),
            ),
            ("imputer", _create_median_imputer()),
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
