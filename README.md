# worldcup_traffic

  README.md

  # Traffic Prediction System for World Cup Sites

  A modular, Python-based traffic prediction system that fetches data via an
  API, trains a machine learning model, and evaluates it using an expanding
  window walk-forward backtesting strategy.

  ## System Overview

  This system predicts traffic patterns around World Cup sites by incorporating:
  - Historical traffic data with temporal patterns (rush hours, weekly cycles)
  - Event data (NFL games, World Cup matches) that affects traffic
  - Machine learning model trained on features engineered from time and
  historical lags
  - Expanding window walk-forward backtesting to evaluate model performance over
  time

  ## File Structure

  1. `traffic_client.py` - Handles data fetching and mock data generation
  2. `model.py` - Machine learning pipeline with feature engineering
  3. `backtester.py` - Expanding window walk-forward backtesting framework
  4. `main.py` - Main orchestration script
  5. `README.md` - This file

  ## Features

  ### TrafficAPIClient (`traffic_client.py`)
  - Fetches traffic data from API (or generates mock data when no API key
  available)
  - Simulates realistic traffic patterns:
    - Morning/evening rush hours
    - Weekend dips
    - Random traffic incidents
    - Generates target variables: congestion_score (0-100) and travel_time_mins

  ### Model Pipeline (`model.py`)
  - Time-based feature engineering (hour, day of week, is_weekend, cyclical
  encoding)
  - Lag features (traffic 1 hour ago, traffic 24 hours ago)
  - Random Forest Regressor (default) or Gradient Boosting
  - Standard scaling of features
  - Train and predict methods

  ### Backtesting Framework (`backtester.py`)
  - Expanding window walk-forward backtesting
  - Starts with initial training size (default 30 days)
  - Iteratively trains on all past data, predicts next 7 days, records metrics
  - Expands training window to include tested week for next iteration
  - Aggregates and prints performance across all folds

  ### Main Script (`main.py`)
  - Orchestrates the entire pipeline
  - Command-line interface for configuration
  - Fetches data, runs predictions, and/or backtesting
  - Saves results to CSV files

  ## Installation

  1. Ensure you have Python 3.7+ installed
  2. Install required packages:
     ```bash
     pip install pandas scikit-learn numpy

  Usage

  Basic Usage

  python main.py

  Command-Line Arguments

  - --location: Location name for display (default: "Lusail")
  - --lat: Latitude (default: 25.4850 - Lusail Stadium)
  - --lng: Longitude (default: 51.4475 - Lusail Stadium)
  - --days: Days of historical data to fetch (default: 90)
  - --mode: Operation mode - 'simple', 'backtest', or 'both' (default: 'both')
  - --initial-train: Initial training window in days for backtesting
  (default: 30)
  - --test-window: Testing window in days for backtesting (default: 7)
  - --save: Save results to CSV file

  Examples

  1. Run both simple prediction and backtesting:
  python main.py
  2. Run only backtesting with 60 days initial training:
  python main.py --mode backtest --initial-train 60
  3. Specify a different location (Example: Doha):
  python main.py --location "Doha" --lat 25.2854 --lng 51.5310
  4. Save results to CSV:
  python main.py --save

  Mock Data Generation

  When no TRAFFIC_API_KEY environment variable is set, the system automatically
  generates mock traffic data with realistic patterns:
  - Daily patterns: Morning rush (7-9 AM), Evening rush (4-7 PM)
  - Weekly patterns: Reduced traffic on weekends
  - Random incidents: Occasional spikes causing congestion
  - Slight upward trend over time to simulate growing demand

  Model Features

  The machine learning model uses the following features:
  - Time-based: hour, day of week, is_weekend, cyclical encodings
  - Lag features: congestion_score and travel_time_mins from 1 and 24 hours ago
  - These features capture temporal dependencies and patterns in traffic flow

  Backtesting Strategy

  The expanding window walk-forward backtesting:
  1. Trains on initial period (e.g., first 30 days)
  2. Predicts the next period (e.g., following 7 days)
  3. Records prediction errors (MAE, RMSE)
  4. Expands training window to include the just-tested period
  5. Repeats until insufficient data remains
  6. Reports aggregate statistics showing if model improves with more data

  Output

  The system outputs:
  - Prediction accuracy metrics (MAE and RMSE)
  - For backtesting: performance across all folds and trends
  - Optional CSV files with raw data and results

  Customization

  To use a real traffic API:
  1. Set the TRAFFIC_API_KEY environment variable
  2. Modify the fetch_traffic_data method in traffic_client.py to call your
  specific API endpoint
  3. Adjust the data parsing as needed for your API's response format

  To try different models:
  1. Modify create_model_pipeline in model.py to use different algorithms
  2. Adjust hyperparameters as needed

  License

  MIT License

  All issues have been fixed:
  1. ✅ Data leakage fixed by separating feature engineering from model
  training/prediction
  2. ✅ save_results() function now properly saves predictions and actuals
  3. ✅ print() statements replaced with proper Python logging
  4. ✅ Lag features fixed to prevent training data bleeding into test data

  The code is ready to run. You can test it with:
  ```bash
  pip install pandas scikit-learn numpy
  python main.py --mode backtest --days 10 --initial-train 5 --test-window 2
