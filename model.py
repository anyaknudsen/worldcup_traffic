

  """
   model.py
  Machine Learning model for traffic prediction.
  Implements a pipeline with feature engineering and prediction capabilities.
  """

  import pandas as pd
  import numpy as np
  from sklearn.base import BaseEstimator, TransformerMixin
  from sklearn.pipeline import Pipeline
  from sklearn.preprocessing import StandardScaler
  from sklearn.ensemble import RandomForestRegressor
  from sklearn.metrics import mean_absolute_error, mean_squared_error
  import warnings
  warnings.filterwarnings('ignore')


  class TimeFeatureEngineer(BaseEstimator, TransformerMixin):
      """
      Custom transformer to create time-based features from timestamps.
      """

      def __init__(self):
          pass

      def fit(self, X, y=None):
          return self

      def transform(self, X):
          """
          Transform input data by adding time-based features.

          Args:
              X (pd.DataFrame): Input data with 'timestamp' column

          Returns:
              pd.DataFrame: Data with additional time features
          """
          # Ensure we're working with a copy
          X = X.copy()

          # Convert timestamp to datetime if it isn't already
          if not pd.api.types.is_datetime64_any_dtype(X['timestamp']):
              X['timestamp'] = pd.to_datetime(X['timestamp'])

          # Extract time-based features
          X['hour'] = X['timestamp'].dt.hour
          X['day_of_week'] = X['timestamp'].dt.dayofweek  # Monday=0, Sunday=6
          X['is_weekend'] = (X['day_of_week'] >= 5).astype(int)

          # Cyclical encoding for hour and day of week
          X['hour_sin'] = np.sin(2 * np.pi * X['hour'] / 24)
          X['hour_cos'] = np.cos(2 * np.pi * X['hour'] / 24)
          X['day_of_week_sin'] = np.sin(2 * np.pi * X['day_of_week'] / 7)
          X['day_of_week_cos'] = np.cos(2 * np.pi * X['day_of_week'] / 7)

          return X


  class LagFeatureEngineer(BaseEstimator, TransformerMixin):
      """
      Custom transformer to create lag features for time series data.
      """

      def __init__(self, lag_columns=None, lag_hours=[1, 24]):
          """
          Initialize the LagFeatureEngineer.

          Args:
              lag_columns (list): Columns to create lag features for.
                                 If None, uses ['congestion_score',
  'travel_time_mins']
              lag_hours (list): List of hours to lag (e.g., [1, 24] for 1 hour
  and 24 hours ago)
          """
          self.lag_columns = lag_columns or ['congestion_score',
  'travel_time_mins']
          self.lag_hours = lag_hours

      def fit(self, X, y=None):
          return self

      def transform(self, X):
          """
          Transform input data by adding lag features.

          Args:
              X (pd.DataFrame): Input data sorted by timestamp

          Returns:
              pd.DataFrame: Data with additional lag features
          """
          # Ensure we're working with a copy
          X = X.copy()

          # Ensure data is sorted by timestamp
          if 'timestamp' in X.columns:
              X = X.sort_values('timestamp').reset_index(drop=True)

          # Create lag features for each specified column
          for col in self.lag_columns:
              if col in X.columns:
                  for lag in self.lag_hours:
                      X[f'{col}_lag_{lag}'] = X[col].shift(lag)

          return X


  def create_feature_engineering_pipeline():
      """
      Create a pipeline for feature engineering only (time and lag features).

      Returns:
          sklearn.pipeline.Pipeline: Feature engineering pipeline
      """
      return Pipeline(steps=[
          ('time_features', TimeFeatureEngineer()),
          ('lag_features', LagFeatureEngineer())
      ])


  def create_model_pipeline_from_features(model_type='random_forest'):
      """
      Create a machine learning pipeline for traffic prediction from already
  featured data.

      Args:
          model_type (str): Type of model to use ('random_forest' or
  'gradient_boosting')

      Returns:
          sklearn.pipeline.Pipeline: Configured machine learning pipeline
  (scaler + model)
      """
      # Define the model
      if model_type == 'random_forest':
          model = RandomForestRegressor(
              n_estimators=100,
              max_depth=10,
              min_samples_split=5,
              min_samples_leaf=2,
              random_state=42,
              n_jobs=-1
          )
      elif model_type == 'gradient_boosting':
          from sklearn.ensemble import GradientBoostingRegressor
          model = GradientBoostingRegressor(
              n_estimators=100,
              learning_rate=0.1,
              max_depth=5,
              random_state=42
          )
      else:
          raise ValueError(f"Unsupported model type: {model_type}")

      # Create the pipeline
      return Pipeline(steps=[
          ('scaler', StandardScaler()),
          ('regressor', model)
      ])


  def create_model_pipeline(model_type='random_forest'):
      """
      Create a machine learning pipeline for traffic prediction (includes
  feature engineering).
      This function is kept for backward compatibility.

      Args:
          model_type (str): Type of model to use ('random_forest' or
  'gradient_boosting')

      Returns:
          sklearn.pipeline.Pipeline: Configured machine learning pipeline
      """
      # Define the preprocessing steps
      preprocessing_steps = [
          ('time_features', TimeFeatureEngineer()),
          ('lag_features', LagFeatureEngineer()),
      ]

      # Define the model
      if model_type == 'random_forest':
          model = RandomForestRegressor(
              n_estimators=100,
              max_depth=10,
              min_samples_split=5,
              min_samples_leaf=2,
              random_state=42,
              n_jobs=-1
          )
      elif model_type == 'gradient_boosting':
          from sklearn.ensemble import GradientBoostingRegressor
          model = GradientBoostingRegressor(
              n_estimators=100,
              learning_rate=0.1,
              max_depth=5,
              random_state=42
          )
      else:
          raise ValueError(f"Unsupported model type: {model_type}")

      # Create the pipeline
      pipeline_steps = preprocessing_steps + [
          ('scaler', StandardScaler()),
          ('regressor', model)
      ]

      return Pipeline(steps=pipeline_steps)


  def train_model(pipeline, X_train, y_train):
      """
      Train the machine learning pipeline.

      Args:
          pipeline (sklearn.pipeline.Pipeline): Pipeline to train
          X_train (pd.DataFrame): Training features
          y_train (pd.Series): Training target

      Returns:
          sklearn.pipeline.Pipeline: Trained pipeline
      """
      # Fit the pipeline
      pipeline.fit(X_train, y_train)
      return pipeline


  def predict_model(pipeline, X):
      """
      Make predictions using the trained pipeline.

      Args:
          pipeline (sklearn.pipeline.Pipeline): Trained pipeline
          X (pd.DataFrame): Features for prediction

      Returns:
          np.array: Predicted values
      """
      return pipeline.predict(X)


  def evaluate_predictions(y_true, y_pred):
      """
      Evaluate predictions using MAE and RMSE.

      Args:
          y_true (np.array): True values
          y_pred (np.array): Predicted values

      Returns:
          dict: Dictionary containing MAE and RMSE
      """
      mae = mean_absolute_error(y_true, y_pred)
      rmse = np.sqrt(mean_squared_error(y_true, y_pred))
      return {'mae': mae, 'rmse': rmse}
