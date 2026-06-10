"""Command-line entry point for the traffic prediction pipeline."""

import argparse
import logging
import sys
from datetime import datetime, timedelta

import numpy as np

from backtester import WalkForwardBacktester
from model import create_model_pipeline, evaluate_predictions, predict_model, train_model
from traffic_client import TrafficAPIClient

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def fetch_data(location, days=90):
    """Fetch traffic data for a location."""
    client = TrafficAPIClient()
    end_time = datetime.now()
    start_time = end_time - timedelta(days=days)
    return client.fetch_traffic_data(location, start_time, end_time)


def run_simple_prediction(data, target_column="congestion_score"):
    """Run a simple chronological train/test split prediction."""
    exclude_cols = ["timestamp"]
    if "location_id" in data.columns:
        exclude_cols.append("location_id")
    if target_column in data.columns:
        exclude_cols.append(target_column)

    X = data.drop(columns=exclude_cols)
    y = data[target_column]
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    pipeline = create_model_pipeline("random_forest")
    trained_pipeline = train_model(pipeline, X_train, y_train)
    y_pred = predict_model(trained_pipeline, X_test)
    metrics = evaluate_predictions(y_test.values, y_pred)
    return {
        "metrics": metrics,
        "predictions": y_pred,
        "actuals": y_test.values,
        "X_test": X_test,
        "y_test": y_test,
    }


def run_backtesting(data, initial_train_days=30, test_days=7):
    """Run expanding-window walk-forward backtesting."""
    backtester = WalkForwardBacktester(
        initial_train_days=initial_train_days,
        test_days=test_days,
        model_type="random_forest",
    )
    summary = backtester.backtest(data, target_column="congestion_score")
    backtester.print_summary(summary)
    return summary


def save_results(data, predictions=None, actuals=None, filename="traffic_predictions.csv"):
    """Save raw data, optionally with aligned prediction columns."""
    data_copy = data.copy()
    if predictions is not None and actuals is not None:
        data_copy["predictions"] = np.nan
        data_copy["actuals"] = np.nan
        if len(predictions) == len(actuals) and len(predictions) <= len(data_copy):
            data_copy.iloc[-len(predictions) :, data_copy.columns.get_loc("predictions")] = predictions
            data_copy.iloc[-len(predictions) :, data_copy.columns.get_loc("actuals")] = actuals
        else:
            logger.warning("Length of predictions/actuals does not match data length.")
    data_copy.to_csv(filename, index=False)
    logger.info("Results saved to %s", filename)


def main():
    """Parse command-line arguments and run the selected pipeline mode."""
    parser = argparse.ArgumentParser(description="Traffic Prediction System for World Cup Sites")
    parser.add_argument("--location", type=str, default="Lusail")
    parser.add_argument("--lat", type=float, default=25.4850)
    parser.add_argument("--lng", type=float, default=51.4475)
    parser.add_argument("--days", type=int, default=90)
    parser.add_argument("--mode", type=str, choices=["simple", "backtest", "both"], default="both")
    parser.add_argument("--initial-train", type=int, default=30)
    parser.add_argument("--test-window", type=int, default=7)
    parser.add_argument("--save", action="store_true")
    args = parser.parse_args()

    data = fetch_data({"lat": args.lat, "lng": args.lng}, days=args.days)
    try:
        if args.mode in ["simple", "both"]:
            simple_results = run_simple_prediction(data)
            if args.save:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                save_results(
                    data,
                    predictions=simple_results["predictions"],
                    actuals=simple_results["actuals"],
                    filename=f"traffic_simple_{args.location}_{timestamp}.csv",
                )

        if args.mode in ["backtest", "both"]:
            run_backtesting(data, initial_train_days=args.initial_train, test_days=args.test_window)
    except Exception as exc:
        logger.error("Error running pipeline: %s", exc, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
