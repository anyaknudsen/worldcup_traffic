

  """
   backtester.py
  Expanding window walk-forward backtesting framework for traffic prediction.
  """

  import pandas as pd
  import numpy as np
  import logging
  from typing import Tuple, List, Dict
  import warnings
  warnings.filterwarnings('ignore')

  from model import create_feature_engineering_pipeline,
  create_model_pipeline_from_features


  # Set up logging
  logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s -
  %(message)s')
  logger = logging.getLogger(__name__)


  class WalkForwardBacktester:
      """
      Implements an expanding window walk-forward backtesting strategy.
      """

      def __init__(self,
                   initial_train_days: int = 30,
                   test_days: int = 7,
                   model_type: str = 'random_forest'):
          """
          Initialize the backtester.

          Args:
              initial_train_days (int): Initial training window size in days
              test_days (int): Number of days to predict in each iteration
              model_type (str): Type of model to use ('random_forest' or
  'gradient_boosting')
          """
          self.initial_train_days = initial_train_days
          self.test_days = test_days
          self.model_type = model_type
          self.history = []  # To store results from each fold

      def backtest(self,
                   data: pd.DataFrame,
                   target_column: str = 'congestion_score',
                   timestamp_column: str = 'timestamp') -> Dict:
          """
          Run expanding window walk-forward backtesting.

          Args:
              data (pd.DataFrame): Complete dataset with timestamp and features
              target_column (str): Name of the target column to predict
              timestamp_column (str): Name of the timestamp column

          Returns:
              dict: Dictionary containing backtesting results and summary
  metrics
          """
          # Ensure data is sorted by timestamp
          data = data.sort_values(timestamp_column).reset_index(drop=True)

          # Convert timestamp to datetime if needed
          if not pd.api.types.is_datetime64_any_dtype(data[timestamp_column]):
              data[timestamp_column] = pd.to_datetime(data[timestamp_column])

          # Calculate total hours for initial train and test windows
          initial_train_hours = self.initial_train_days * 24
          test_hours = self.test_days * 24

          # Check if we have enough data
          if len(data) < initial_train_hours + test_hours:
              raise ValueError(f"Not enough data. Need at least
  {initial_train_hours + test_hours} hours, "
                             f"but got {len(data)} hours.")

          # Initialize tracking variables
          start_idx = 0
          end_train_idx = initial_train_hours
          fold = 0

          # Store results for each fold
          fold_results = []

          # Continue until we run out of data for testing
          while end_train_idx + test_hours <= len(data):
              fold += 1
              logger.info(f"--- Fold {fold} ---")

              # Define indices for the combined data (training + test set for
  this fold)
              # We use data from start_idx to end_train_idx + test_hours
  (exclusive)
              combined_end_idx = end_train_idx + test_hours
              combined_data = data.iloc[start_idx:combined_end_idx].copy()
              logger.info(f"Combined data period:
  {combined_data[timestamp_column].iloc[0]} to
  {combined_data[timestamp_column].iloc[-1]}")

              # Create feature engineering pipeline and transform the combined
  data
              fe_pipeline = create_feature_engineering_pipeline()
              # Fit and transform (though fitting does nothing for our
  transformers, we do it for consistency)
              featured_data = fe_pipeline.fit_transform(combined_data)
              logger.info(f"Featured data shape: {featured_data.shape}")

              # Split the featured data into training and testing sets
              # Training set: from start of combined data to the end of the
  original training set
              train_size = end_train_idx - start_idx
              train_featured = featured_data.iloc[:train_size]
              test_featured = featured_data.iloc[train_size:train_size +
  test_hours]

              logger.info(f"Training period:
  {train_featured[timestamp_column].iloc[0]} to
  {train_featured[timestamp_column].iloc[-1]}")
              logger.info(f"Test period:
  {test_featured[timestamp_column].iloc[0]} to
  {test_featured[timestamp_column].iloc[-1]}")
              logger.info(f"Training samples: {len(train_featured)}, Test
  samples: {len(test_featured)}")

              # Prepare features and target for training
              exclude_cols = [timestamp_column, 'location_id'] if 'location_id'
  in train_featured.columns else [timestamp_column]
              if target_column in train_featured.columns:
                  exclude_cols.append(target_column)
              X_train = train_featured.drop(columns=exclude_cols)
              y_train = train_featured[target_column]

              X_test = test_featured.drop(columns=exclude_cols)
              y_test = test_featured[target_column]

              # Create and train model pipeline (scaler + model) on the training
  featured data
              model_pipeline =
  create_model_pipeline_from_features(self.model_type)
              trained_pipeline = train_model(model_pipeline, X_train, y_train)

              # Make predictions
              y_pred = predict_model(trained_pipeline, X_test)

              # Evaluate predictions
              metrics = evaluate_predictions(y_test.values, y_pred)
              logger.info(f"Fold {fold} - MAE: {metrics['mae']:.2f}, RMSE:
  {metrics['rmse']:.2f}")

              # Store fold results
              fold_result = {
                  'fold': fold,
                  'train_start': train_featured[timestamp_column].iloc[0],
                  'train_end': train_featured[timestamp_column].iloc[-1],
                  'test_start': test_featured[timestamp_column].iloc[0],
                  'test_end': test_featured[timestamp_column].iloc[-1],
                  'train_size': len(train_featured),
                  'test_size': len(test_featured),
                  'mae': metrics['mae'],
                  'rmse': metrics['rmse'],
                  'predictions': y_pred,
                  'actuals': y_test.values
              }
              fold_results.append(fold_result)

              # Expand the training window for next iteration (expanding window)
              # In expanding window, we keep all previous data and add the test
  set to training
              end_train_idx = combined_end_idx  # New end of training window

          # Calculate summary statistics
          if fold_results:
              mae_values = [result['mae'] for result in fold_results]
              rmse_values = [result['rmse'] for result in fold_results]

              summary = {
                  'total_folds': len(fold_results),
                  'mae_mean': np.mean(mae_values),
                  'mae_std': np.std(mae_values),
                  'mae_min': np.min(mae_values),
                  'mae_max': np.max(mae_values),
                  'rmse_mean': np.mean(rmse_values),
                  'rmse_std': np.std(rmse_values),
                  'rmse_min': np.min(rmse_values),
                  'rmse_max': np.max(rmse_values),
                  'fold_results': fold_results
              }
          else:
              summary = {
                  'total_folds': 0,
                  'mae_mean': None,
                  'mae_std': None,
                  'mae_min': None,
                  'mae_max': None,
                  'rmse_mean': None,
                  'rmse_std': None,
                  'rmse_min': None,
                  'rmse_max': None,
                  'fold_results': []
              }

          return summary

      def print_summary(self, summary: Dict):
          """
          Print a formatted summary of the backtesting results.

          Args:
              summary (dict): Summary dictionary from backtest()
          """
          logger.info("\n" + "="*60)
          logger.info("WALK-FORWARD BACKTESTING SUMMARY")
          logger.info("="*60)

          if summary['total_folds'] == 0:
              logger.info("No backtesting folds completed. Not enough data.")
              return

          logger.info(f"Number of folds: {summary['total_folds']}")
          logger.info(f"\nMAE (Mean Absolute Error):")
          logger.info(f"  Mean: {summary['mae_mean']:.2f}")
          logger.info(f"  Std:  {summary['mae_std']:.2f}")
          logger.info(f"  Min:  {summary['mae_min']:.2f}")
          logger.info(f"  Max:  {summary['mae_max']:.2f}")

          logger.info(f"\nRMSE (Root Mean Squared Error):")
          logger.info(f"  Mean: {summary['rmse_mean']:.2f}")
          logger.info(f"  Std:  {summary['rmse_std']:.2f}")
          logger.info(f"  Min:  {summary['rmse_min']:.2f}")
          logger.info(f"  Max:  {summary['rmse_max']:.2f}")

          # Show trend: is performance improving over time?
          if summary['total_folds'] >= 3:
              first_three_mae = [fold['mae'] for fold in
  summary['fold_results'][:3]]
              last_three_mae = [fold['mae'] for fold in
  summary['fold_results'][-3:]]
              first_avg = np.mean(first_three_mae)
              last_avg = np.mean(last_three_mae)
              trend = "improving" if last_avg < first_avg else "degrading" if
  last_avg > first_avg else "stable"
              logger.info(f"\nPerformance trend (first 3 vs last 3 folds):
  {trend}")
              logger.info(f"  First 3 folds avg MAE: {first_avg:.2f}")
              logger.info(f"  Last 3 folds avg MAE:  {last_avg:.2f}")


  def run_backtest_example():
      """
      Example function demonstrating how to use the WalkForwardBacktester.
      """
      from traffic_client import TrafficAPIClient

      # Generate mock data for backtesting example
      logger.info("Generating mock data for backtesting example...")
      client = TrafficAPIClient()
      location = {'lat': 25.4850, 'lng': 51.4475}
      end_time = pd.Timestamp.now()
      start_time = end_time - pd.Timedelta(days=90)  # 90 days of data

      data = client.fetch_traffic_data(location, start_time, end_time)
      logger.info(f"Generated {len(data)} hourly records")

      # Initialize backtester
      backtester = WalkForwardBacktester(
          initial_train_days=30,  # Start with 30 days of training data
          test_days=7,            # Predict 7 days at a time
          model_type='random_forest'
      )

      # Run backtesting
      logger.info("\nStarting expanding window walk-forward backtesting...")
      summary = backtester.backtest(data, target_column='congestion_score')

      # Print summary
      backtester.print_summary(summary)

      return summary

  main.py

  """
  Main script to run the traffic prediction pipeline.
  Orchestrates data fetching, model training, and backtesting.
  """

  import os
  import pandas as pd
  from datetime import datetime, timedelta
  import argparse
  import sys
  import logging

  # Import our custom modules
  from traffic_client import TrafficAPIClient
  from model import create_model_pipeline, train_model, predict_model,
  evaluate_predictions
  from backtester import WalkForwardBacktester


  # Set up logging
  logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s -
  %(message)s')
  logger = logging.getLogger(__name__)


  def fetch_data(location, days=90):
      """
      Fetch traffic data for a given location.

      Args:
          location (dict): Location information
          days (int): Number of days of data to fetch

      Returns:
          pd.DataFrame: Traffic data
      """
      logger.info(f"Fetching {days} days of traffic data...")
      client = TrafficAPIClient()

      end_time = datetime.now()
      start_time = end_time - timedelta(days=days)

      data = client.fetch_traffic_data(location, start_time, end_time)
      logger.info(f"Fetched {len(data)} hourly records")
      return data


  def run_simple_prediction(data, target_column='congestion_score'):
      """
      Run a simple train/test split prediction (not backtesting).

      Args:
          data (pd.DataFrame): Input data
          target_column (str): Target column to predict

      Returns:
          dict: Prediction results and metrics
      """
      logger.info("Running simple train/test split prediction...")

      # Prepare features and target
      exclude_cols = ['timestamp', 'location_id'] if 'location_id' in
  data.columns else ['timestamp']
      if target_column in data.columns:
          exclude_cols.append(target_column)

      X = data.drop(columns=exclude_cols)
      y = data[target_column]

      # Split data (80% train, 20% test)
      split_idx = int(len(X) * 0.8)
      X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
      y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

      logger.info(f"Training samples: {len(X_train)}, Test samples:
  {len(X_test)}")

      # Create and train model
      pipeline = create_model_pipeline('random_forest')
      trained_pipeline = train_model(pipeline, X_train, y_train)

      # Make predictions
      y_pred = predict_model(trained_pipeline, X_test)

      # Evaluate
      metrics = evaluate_predictions(y_test.values, y_pred)

      logger.info(f"Prediction Results:")
      logger.info(f"  MAE: {metrics['mae']:.2f}")
      logger.info(f"  RMSE: {metrics['rmse']:.2f}")

      return {
          'metrics': metrics,
          'predictions': y_pred,
          'actuals': y_test.values,
          'X_test': X_test,
          'y_test': y_test
      }


  def run_backtesting(data, initial_train_days=30, test_days=7):
      """
      Run expanding window walk-forward backtesting.

      Args:
          data (pd.DataFrame): Input data
          initial_train_days (int): Initial training window in days
          test_days (int): Testing window in days

      Returns:
          dict: Backtesting summary
      """
      logger.info("\nRunning expanding window walk-forward backtesting...")
      logger.info(f"Initial training window: {initial_train_days} days")
      logger.info(f"Testing window: {test_days} days")

      # Initialize backtester
      backtester = WalkForwardBacktester(
          initial_train_days=initial_train_days,
          test_days=test_days,
          model_type='random_forest'
      )

      # Run backtesting
      summary = backtester.backtest(data, target_column='congestion_score')

      # Print summary (using logging inside the method)
      backtester.print_summary(summary)

      return summary


  def save_results(data, predictions=None, actuals=None,
  filename='traffic_predictions.csv'):
      """
      Save results to a CSV file.

      Args:
          data (pd.DataFrame): Original data
          predictions (np.array): Predicted values (optional)
          actuals (np.array): Actual values (optional)
          filename (str): Output filename
      """
      try:
          if predictions is not None and actuals is not None:
              # We want to save the predictions and actuals aligned with the
  test period.
              # However, in the context of this function being called from
  main.py,
              # we don't have the test period indices. So we will save the data
  along with
              # the predictions and actuals as new columns, but note that this
  is only
              # meaningful if the indices align.
              # For simplicity, we'll assume that the predictions and actuals
  are for the
              # entire data in order (which is not true in backtesting, but in
  simple
              # prediction they are for the test set which is the last part of
  the data).
              # Since we don't have context, we will save the data and then add
  two columns:
              # 'predictions' and 'actuals', filling with NaN where we don't
  have values.
              # This is a simplified approach. In a real system, we would save
  the
              # predictions and actuals with their timestamps.
              data_copy = data.copy()
              # Initialize new columns with NaN
              data_copy['predictions'] = np.nan
              data_copy['actuals'] = np.nan

              # If we have predictions and actuals, we assume they correspond to
  the last
              # len(predictions) rows of the data (which is true for the simple
  prediction
              # in main.py, but not for backtesting). This is a limitation.
              if len(predictions) == len(actuals) and len(predictions) <=
  len(data_copy):
                  data_copy.iloc[-len(predictions):,
  data_copy.columns.get_loc('predictions')] = predictions
                  data_copy.iloc[-len(predictions):,
  data_copy.columns.get_loc('actuals')] = actuals
              else:
                  logger.warning("Length of predictions/actuals does not match
  data length. Not saving predictions and actuals.")

              logger.info(f"Saving results to {filename}...")
              data_copy.to_csv(filename, index=False)
              logger.info(f"Results saved to {filename}")
          else:
              # If no predictions and actuals, just save the data
              logger.info(f"Saving data to {filename}...")
              data.to_csv(filename, index=False)
              logger.info(f"Data saved to {filename}")
      except Exception as e:
          logger.error(f"Error saving results: {e}")


  def main():
      """
      Main function to run the traffic prediction pipeline.
      """
      parser = argparse.ArgumentParser(description='Traffic Prediction System
  for World Cup Sites')
      parser.add_argument('--location', type=str, default='Lusail',
                          help='Location name (for display purposes)')
      parser.add_argument('--lat', type=float, default=25.4850,
                          help='Latitude of the location')
      parser.add_argument('--lng', type=float, default=51.4475,
                          help='Longitude of the location')
      parser.add_argument('--days', type=int, default=90,
                          help='Number of days of historical data to fetch')
      parser.add_argument('--mode', type=str, choices=['simple', 'backtest',
  'both'],
                          default='both', help='Mode to run: simple train/test,
  backtesting, or both')
      parser.add_argument('--initial-train', type=int, default=30,
                          help='Initial training window in days for
  backtesting')
      parser.add_argument('--test-window', type=int, default=7,
                          help='Testing window in days for backtesting')
      parser.add_argument('--save', action='store_true',
                          help='Save results to CSV file')

      args = parser.parse_args()

      logger.info("="*60)
      logger.info("TRAFFIC PREDICTION SYSTEM FOR WORLD CUP SITES")
      logger.info("="*60)
      logger.info(f"Location: {args.location} ({args.lat}, {args.lng})")
      logger.info(f"Data period: {args.days} days")
      logger.info(f"Mode: {args.mode}")
      logger.info("="*60)

      # Prepare location dictionary
      location = {'lat': args.lat, 'lng': args.lng}

      try:
          # Fetch data
          data = fetch_data(location, days=args.days)

          # Run selected mode(s)
          if args.mode in ['simple', 'both']:
              simple_results = run_simple_prediction(data)
              if args.save:
                  timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                  filename = f"traffic_simple_{args.location}_{timestamp}.csv"
                  # Save the data along with predictions and actuals for the
  test set
                  save_results(data,
                              predictions=simple_results['predictions'],
                              actuals=simple_results['actuals'],
                              filename=filename)

          if args.mode in ['backtest', 'both']:
              backtest_results = run_backtesting(
                  data,
                  initial_train_days=args.initial_train,
                  test_days=args.test_window
              )
              if args.save:
                  timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                  filename = f"traffic_backtest_{args.location}_{timestamp}.csv"
                  # For backtesting, we don't have a single set of predictions
  and actuals to save
                  # in the same way as simple prediction. We'll save the raw
  data.
                  save_results(data, filename=filename)

          logger.info("\n" + "="*60)
          logger.info("PIPELINE COMPLETED SUCCESSFULLY")
          logger.info("="*60)

      except Exception as e:
          logger.error(f"Error running pipeline: {str(e)}")
          import traceback
          logger.error(traceback.format_exc())
          sys.exit(1)


  if __name__ == "__main__":
      main()
