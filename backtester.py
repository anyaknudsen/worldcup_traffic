"""Expanding-window walk-forward backtesting for traffic prediction."""

import logging
from typing import Dict

import numpy as np
import pandas as pd

from model import (
    create_train_test_feature_sets,
    create_model_pipeline_from_features,
    evaluate_predictions,
    predict_model,
    select_forecast_features,
    sort_time_series,
    train_model,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class WalkForwardBacktester:
    """Run expanding-window walk-forward backtests."""

    def __init__(
        self,
        initial_train_days: int = 30,
        test_days: int = 7,
        model_type: str = "random_forest",
    ):
        """
        Initialize the backtester.

        Args:
            initial_train_days: Initial training window size in days.
            test_days: Number of days to predict in each iteration.
            model_type: Model type to use (``random_forest`` or
                ``gradient_boosting``).
        """
        self.initial_train_days = initial_train_days
        self.test_days = test_days
        self.model_type = model_type
        self.history = []

    def backtest(
        self,
        data: pd.DataFrame,
        target_column: str = "congestion_score",
        timestamp_column: str = "timestamp",
    ) -> Dict:
        """
        Run expanding-window walk-forward backtesting.

        Args:
            data: Complete dataset with timestamp and features.
            target_column: Name of the target column to predict.
            timestamp_column: Name of the timestamp column.

        Returns:
            Dictionary containing fold results and summary metrics.
        """
        data = sort_time_series(data, timestamp_column=timestamp_column)

        initial_train_hours = self.initial_train_days * 24
        test_hours = self.test_days * 24

        if len(data) < initial_train_hours + test_hours:
            raise ValueError(
                "Not enough data. Need at least "
                f"{initial_train_hours + test_hours} hours, but got {len(data)} hours."
            )

        end_train_idx = initial_train_hours
        fold_results = []
        fold = 0

        while end_train_idx + test_hours <= len(data):
            fold += 1
            logger.info("--- Fold %s ---", fold)

            combined_end_idx = end_train_idx + test_hours
            train_data = data.iloc[:end_train_idx].reset_index(drop=True)
            test_data = data.iloc[end_train_idx:combined_end_idx].reset_index(drop=True)
            logger.info(
                "Train/test data period: %s to %s",
                train_data[timestamp_column].iloc[0],
                test_data[timestamp_column].iloc[-1],
            )

            train_featured, test_featured = create_train_test_feature_sets(
                train_data,
                test_data,
                timestamp_column=timestamp_column,
            )
            logger.info(
                "Featured train/test shapes: %s / %s",
                train_featured.shape,
                test_featured.shape,
            )

            logger.info(
                "Training period: %s to %s",
                train_featured[timestamp_column].iloc[0],
                train_featured[timestamp_column].iloc[-1],
            )
            logger.info(
                "Test period: %s to %s",
                test_featured[timestamp_column].iloc[0],
                test_featured[timestamp_column].iloc[-1],
            )
            logger.info(
                "Training samples: %s, Test samples: %s",
                len(train_featured),
                len(test_featured),
            )

            X_train = select_forecast_features(
                train_featured,
                target_column=target_column,
                timestamp_column=timestamp_column,
            )
            y_train = train_featured[target_column]
            X_test = select_forecast_features(
                test_featured,
                target_column=target_column,
                timestamp_column=timestamp_column,
            )
            y_test = test_featured[target_column]

            model_pipeline = create_model_pipeline_from_features(self.model_type)
            trained_pipeline = train_model(model_pipeline, X_train, y_train)
            y_pred = predict_model(trained_pipeline, X_test)

            metrics = evaluate_predictions(y_test.values, y_pred)
            logger.info(
                "Fold %s - MAE: %.2f, RMSE: %.2f",
                fold,
                metrics["mae"],
                metrics["rmse"],
            )

            fold_result = {
                "fold": fold,
                "train_start": train_featured[timestamp_column].iloc[0],
                "train_end": train_featured[timestamp_column].iloc[-1],
                "test_start": test_featured[timestamp_column].iloc[0],
                "test_end": test_featured[timestamp_column].iloc[-1],
                "train_size": len(train_featured),
                "test_size": len(test_featured),
                "mae": metrics["mae"],
                "rmse": metrics["rmse"],
                "predictions": y_pred,
                "actuals": y_test.values,
            }
            fold_results.append(fold_result)
            end_train_idx = combined_end_idx

        self.history = fold_results
        return self._summarize(fold_results)

    def _summarize(self, fold_results):
        if not fold_results:
            return {
                "total_folds": 0,
                "mae_mean": None,
                "mae_std": None,
                "mae_min": None,
                "mae_max": None,
                "rmse_mean": None,
                "rmse_std": None,
                "rmse_min": None,
                "rmse_max": None,
                "fold_results": [],
            }

        mae_values = [result["mae"] for result in fold_results]
        rmse_values = [result["rmse"] for result in fold_results]
        return {
            "total_folds": len(fold_results),
            "mae_mean": np.mean(mae_values),
            "mae_std": np.std(mae_values),
            "mae_min": np.min(mae_values),
            "mae_max": np.max(mae_values),
            "rmse_mean": np.mean(rmse_values),
            "rmse_std": np.std(rmse_values),
            "rmse_min": np.min(rmse_values),
            "rmse_max": np.max(rmse_values),
            "fold_results": fold_results,
        }

    def print_summary(self, summary: Dict):
        """Log a formatted summary of backtesting results."""
        logger.info("\n%s", "=" * 60)
        logger.info("WALK-FORWARD BACKTESTING SUMMARY")
        logger.info("=" * 60)

        if summary["total_folds"] == 0:
            logger.info("No backtesting folds completed. Not enough data.")
            return

        logger.info("Number of folds: %s", summary["total_folds"])
        logger.info("\nMAE (Mean Absolute Error):")
        logger.info("  Mean: %.2f", summary["mae_mean"])
        logger.info("  Std:  %.2f", summary["mae_std"])
        logger.info("  Min:  %.2f", summary["mae_min"])
        logger.info("  Max:  %.2f", summary["mae_max"])

        logger.info("\nRMSE (Root Mean Squared Error):")
        logger.info("  Mean: %.2f", summary["rmse_mean"])
        logger.info("  Std:  %.2f", summary["rmse_std"])
        logger.info("  Min:  %.2f", summary["rmse_min"])
        logger.info("  Max:  %.2f", summary["rmse_max"])

        if summary["total_folds"] >= 3:
            first_three_mae = [fold["mae"] for fold in summary["fold_results"][:3]]
            last_three_mae = [fold["mae"] for fold in summary["fold_results"][-3:]]
            first_avg = np.mean(first_three_mae)
            last_avg = np.mean(last_three_mae)
            trend = (
                "improving"
                if last_avg < first_avg
                else "degrading"
                if last_avg > first_avg
                else "stable"
            )
            logger.info("\nPerformance trend (first 3 vs last 3 folds): %s", trend)
            logger.info("  First 3 folds avg MAE: %.2f", first_avg)
            logger.info("  Last 3 folds avg MAE:  %.2f", last_avg)


def run_backtest_example():
    """Run a demonstration backtest using generated mock traffic data."""
    from traffic_client import TrafficAPIClient

    logger.info("Generating mock data for backtesting example...")
    client = TrafficAPIClient()
    location = {"lat": 25.4850, "lng": 51.4475}
    end_time = pd.Timestamp.now()
    start_time = end_time - pd.Timedelta(days=90)

    data = client.fetch_traffic_data(location, start_time, end_time)
    logger.info("Generated %s hourly records", len(data))

    backtester = WalkForwardBacktester(
        initial_train_days=30,
        test_days=7,
        model_type="random_forest",
    )
    summary = backtester.backtest(data, target_column="congestion_score")
    backtester.print_summary(summary)
    return summary
