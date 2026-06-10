"""Expanding-window walk-forward backtesting for traffic prediction."""

import logging
import warnings
from typing import Dict

import numpy as np
import pandas as pd

from model import (
    create_feature_engineering_pipeline,
    create_model_pipeline_from_features,
    evaluate_predictions,
    predict_model,
    train_model,
)

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


class WalkForwardBacktester:
    """Run an expanding-window walk-forward backtest."""

    def __init__(
        self,
        initial_train_days: int = 30,
        test_days: int = 7,
        model_type: str = "random_forest",
    ):
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
        """Run expanding-window walk-forward backtesting."""
        data = data.sort_values(timestamp_column).reset_index(drop=True)
        if not pd.api.types.is_datetime64_any_dtype(data[timestamp_column]):
            data[timestamp_column] = pd.to_datetime(data[timestamp_column])

        initial_train_hours = self.initial_train_days * 24
        test_hours = self.test_days * 24
        required_hours = initial_train_hours + test_hours
        if len(data) < required_hours:
            raise ValueError(
                f"Not enough data. Need at least {required_hours} hours, "
                f"but got {len(data)} hours."
            )

        start_idx = 0
        end_train_idx = initial_train_hours
        fold = 0
        fold_results = []

        while end_train_idx + test_hours <= len(data):
            fold += 1
            logger.info("--- Fold %s ---", fold)

            combined_end_idx = end_train_idx + test_hours
            combined_data = data.iloc[start_idx:combined_end_idx].copy()
            fe_pipeline = create_feature_engineering_pipeline()
            featured_data = fe_pipeline.fit_transform(combined_data)

            train_size = end_train_idx - start_idx
            train_featured = featured_data.iloc[:train_size]
            test_featured = featured_data.iloc[train_size : train_size + test_hours]

            exclude_cols = [timestamp_column]
            if "location_id" in train_featured.columns:
                exclude_cols.append("location_id")
            if target_column in train_featured.columns:
                exclude_cols.append(target_column)

            X_train = train_featured.drop(columns=exclude_cols)
            y_train = train_featured[target_column]
            X_test = test_featured.drop(columns=exclude_cols)
            y_test = test_featured[target_column]

            model_pipeline = create_model_pipeline_from_features(self.model_type)
            trained_pipeline = train_model(model_pipeline, X_train, y_train)
            y_pred = predict_model(trained_pipeline, X_test)
            metrics = evaluate_predictions(y_test.values, y_pred)

            fold_results.append(
                {
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
            )
            end_train_idx = combined_end_idx

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
        summary = {
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
        self.history = fold_results
        return summary

    def print_summary(self, summary: Dict):
        """Log a formatted summary of backtesting results."""
        logger.info("\n%s", "=" * 60)
        logger.info("WALK-FORWARD BACKTESTING SUMMARY")
        logger.info("%s", "=" * 60)

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
            if last_avg < first_avg:
                trend = "improving"
            elif last_avg > first_avg:
                trend = "degrading"
            else:
                trend = "stable"
            logger.info("\nPerformance trend (first 3 vs last 3 folds): %s", trend)
            logger.info("  First 3 folds avg MAE: %.2f", first_avg)
            logger.info("  Last 3 folds avg MAE:  %.2f", last_avg)

