"""Command-line entry point for the traffic prediction pipeline."""

import argparse
import logging
import sys
from datetime import datetime, timedelta

import numpy as np

from backtester import WalkForwardBacktester
from model import (
    create_train_test_feature_sets,
    create_model_pipeline_from_features,
    evaluate_predictions,
    predict_model,
    select_forecast_features,
    sort_time_series,
    train_model,
)
from traffic_client import TrafficAPIClient


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def fetch_data(location, days=90):
    """
    Fetch traffic data for a given location.

    Args:
        location (dict): Location information.
        days (int): Number of days of data to fetch.

    Returns:
        pd.DataFrame: Traffic data.
    """
    logger.info("Fetching %s days of traffic data...", days)
    client = TrafficAPIClient()

    end_time = datetime.now()
    start_time = end_time - timedelta(days=days)

    data = client.fetch_traffic_data(location, start_time, end_time)
    logger.info("Fetched %s hourly records", len(data))
    return data


def run_simple_prediction(data, target_column="congestion_score"):
    """
    Run a simple train/test split prediction.

    Args:
        data (pd.DataFrame): Input data.
        target_column (str): Target column to predict.

    Returns:
        dict: Prediction results and metrics.
    """
    if len(data) < 2:
        raise ValueError("Need at least two rows for a train/test split.")

    logger.info("Running simple train/test split prediction...")

    data = sort_time_series(data)

    split_idx = int(len(data) * 0.8)
    if split_idx == 0 or split_idx == len(data):
        raise ValueError("Train/test split would be empty; provide more data.")

    train_data = data.iloc[:split_idx].reset_index(drop=True)
    test_data = data.iloc[split_idx:].reset_index(drop=True)
    train_featured, test_featured = create_train_test_feature_sets(
        train_data,
        test_data,
    )

    X_train = select_forecast_features(train_featured, target_column=target_column)
    X_test = select_forecast_features(test_featured, target_column=target_column)
    y_train = train_featured[target_column]
    y_test = test_featured[target_column]

    logger.info("Training samples: %s, Test samples: %s", len(X_train), len(X_test))

    pipeline = create_model_pipeline_from_features("random_forest")
    trained_pipeline = train_model(pipeline, X_train, y_train)
    y_pred = predict_model(trained_pipeline, X_test)
    metrics = evaluate_predictions(y_test.values, y_pred)

    logger.info("Prediction Results:")
    logger.info("  MAE: %.2f", metrics["mae"])
    logger.info("  RMSE: %.2f", metrics["rmse"])

    return {
        "metrics": metrics,
        "predictions": y_pred,
        "actuals": y_test.values,
        "X_test": X_test,
        "y_test": y_test,
    }


def run_backtesting(data, initial_train_days=30, test_days=7):
    """
    Run expanding-window walk-forward backtesting.

    Args:
        data (pd.DataFrame): Input data.
        initial_train_days (int): Initial training window in days.
        test_days (int): Testing window in days.

    Returns:
        dict: Backtesting summary.
    """
    logger.info("\nRunning expanding window walk-forward backtesting...")
    logger.info("Initial training window: %s days", initial_train_days)
    logger.info("Testing window: %s days", test_days)

    backtester = WalkForwardBacktester(
        initial_train_days=initial_train_days,
        test_days=test_days,
        model_type="random_forest",
    )
    summary = backtester.backtest(data, target_column="congestion_score")
    backtester.print_summary(summary)
    return summary


def save_results(data, predictions=None, actuals=None, filename="traffic_predictions.csv"):
    """
    Save traffic data and optional prediction results to a CSV file.

    Prediction arrays are aligned with the final rows of ``data``, which matches
    the simple train/test flow.
    """
    try:
        if predictions is not None and actuals is not None:
            data_copy = data.copy()
            data_copy["predictions"] = np.nan
            data_copy["actuals"] = np.nan

            if len(predictions) == len(actuals) and len(predictions) <= len(data_copy):
                pred_col = data_copy.columns.get_loc("predictions")
                actual_col = data_copy.columns.get_loc("actuals")
                data_copy.iloc[-len(predictions) :, pred_col] = predictions
                data_copy.iloc[-len(predictions) :, actual_col] = actuals
            else:
                logger.warning(
                    "Length of predictions/actuals does not match data length. "
                    "Not saving predictions and actuals."
                )

            logger.info("Saving results to %s...", filename)
            data_copy.to_csv(filename, index=False)
            logger.info("Results saved to %s", filename)
        else:
            logger.info("Saving data to %s...", filename)
            data.to_csv(filename, index=False)
            logger.info("Data saved to %s", filename)
    except Exception as exc:
        logger.error("Error saving results: %s", exc)


def main():
    """Run the traffic prediction pipeline from the command line."""
    parser = argparse.ArgumentParser(
        description="Traffic Prediction System for World Cup Sites"
    )
    parser.add_argument("--location", type=str, default="Lusail")
    parser.add_argument("--lat", type=float, default=25.4850)
    parser.add_argument("--lng", type=float, default=51.4475)
    parser.add_argument("--days", type=int, default=90)
    parser.add_argument(
        "--mode",
        type=str,
        choices=["simple", "backtest", "both"],
        default="both",
    )
    parser.add_argument("--initial-train", type=int, default=30)
    parser.add_argument("--test-window", type=int, default=7)
    parser.add_argument("--save", action="store_true")

    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("TRAFFIC PREDICTION SYSTEM FOR WORLD CUP SITES")
    logger.info("=" * 60)
    logger.info("Location: %s (%s, %s)", args.location, args.lat, args.lng)
    logger.info("Data period: %s days", args.days)
    logger.info("Mode: %s", args.mode)
    logger.info("=" * 60)

    location = {"lat": args.lat, "lng": args.lng}

    try:
        data = fetch_data(location, days=args.days)

        if args.mode in ["simple", "both"]:
            simple_results = run_simple_prediction(data)
            if args.save:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"traffic_simple_{args.location}_{timestamp}.csv"
                save_results(
                    data,
                    predictions=simple_results["predictions"],
                    actuals=simple_results["actuals"],
                    filename=filename,
                )

        if args.mode in ["backtest", "both"]:
            backtest_results = run_backtesting(
                data,
                initial_train_days=args.initial_train,
                test_days=args.test_window,
            )
            if args.save:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"traffic_backtest_{args.location}_{timestamp}.csv"
                save_results(data, filename=filename)

        logger.info("\n%s", "=" * 60)
        logger.info("PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("=" * 60)

    except Exception as exc:
        logger.error("Error running pipeline: %s", exc)
        import traceback

        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
